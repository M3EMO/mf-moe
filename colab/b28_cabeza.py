"""B28: ¿una cabeza de certeza entrenada DENTRO del modelo lee la duda mejor que la misma cabeza sobre un tronco congelado?
Prerregistro en el ticket 10 (mf-moe-generalidad), opción A. La cabeza es lineal sobre el estado del último token de la pregunta
(capas de B27c) y predice la NLL propia del valor real. Entrena con los 255 candidatos fuera del banco; el banco (206) es la prueba.
- congelada: solo la cabeza aprende; el tronco no cambia (es la sonda de B27c entrenada con AdamW).
- conjunta: la cabeza y las capas 6-23 aprenden juntas; un KL contra el modelo de partida frena el cambio de conocimiento.
B28c: el tronco aprende con 5e-6. B28b: la cabeza corre sin dropout y la conjunta arranca de la congelada; B28 falló la integridad por las dos cosas.
Fase 2 (solo si la conjunta hereda, el control pasa y el banco no se movió): B27 sobre el modelo conjunto, brazos cabeza y oráculo.
Uso: python -m colab.b28_cabeza SALIDA.json BANCO.json CANDIDATOS.json
     python -m colab.b28_cabeza d SALIDA.json BANCO.json CANDIDATOS.json            (B28d)
     python -m colab.b28_cabeza f SALIDA.json BANCO.json CANDIDATOS.json HECHOS.json (B28f: anclas y beta 10)
"""

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24
from colab import b27_puerta as b27
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

LAYERS = b27.LAYERS
STEPS, EVAL_EVERY, HEAD_LR, HEAD_WD, BETA, VAL_FRAC = 300, 25, 1e-4, 1.0, 1.0, 5  # piloto en CPU sobre los estados de VH
C1_MIN, I1_RHO, I1_SHIFT, GAIN, BOOT = 0.40, 0.90, 0.15, 0.05, 2000
C1_MIN_D = 0.45  # la ridge antes dio 0,534 en B28, B28b y B28c
TRUNK_LR = 5e-6  # B28c: con el lr de edición (5e-5), B28b movió el banco 0,98 nats
EPISODE, CHECK_D, REPLICAS = (12, 4), 50, 2  # B28d: soporte y consulta de cada lote, cada cuánto mide la integridad, réplicas
# B28f: en B28d el KL valió 0,05 a 0,10 contra un ajuste de 0,2 a 3; beta 10 lo lleva a la escala del ajuste
BETA_F, N_ANCHORS, ANCHOR_BATCH, STEPS_F = 10.0, 512, 16, 150


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


class Head(torch.nn.Module):
    def __init__(self, mu: np.ndarray, sd: np.ndarray):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.lin = torch.nn.Linear(len(mu), 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin((h.float() - self.mu) / self.sd).squeeze(-1)


def snapshot(model, head, trunk: bool) -> dict:
    trunk = {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters() if v.requires_grad} if trunk else {}
    return {"tronco": trunk, "cabeza": {} if head is None else {k: v.detach().to("cpu", copy=True) for k, v in head.state_dict().items()}}


@torch.no_grad()
def ref_logprobs(model, tok, items: list[tuple[str, str]]) -> list[torch.Tensor]:
    """Log-probabilidades del modelo de partida en cada posición real de cada oración, en bf16 en CPU (el KL no necesita otra copia)."""
    model.eval()
    out = []
    for i in range(0, len(items), b17.EVAL_BATCH):
        ids, mask, _ = b17.batch(tok, items[i : i + b17.EVAL_BATCH])
        with amp(DEVICE):
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        out += [lp[j, : int(mask[j].sum())].to(torch.bfloat16).cpu() for j in range(len(ids))]
    model.train()
    return out


def parts(model, tok, items: list[tuple[str, str]]):
    """Una pasada: estado del último token de la pregunta, NLL media del valor y logits (para el KL)."""
    ids, mask, labels = b17.batch(tok, items)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    last = (labels != -100).float().argmax(1) - 1  # la primera etiqueta sigue al último token de la pregunta
    rows = torch.arange(len(items), device=ids.device)
    h = torch.cat([out.hidden_states[l][rows, last].float() for l in LAYERS], -1)
    lab = labels[:, 1:]
    ce = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), lab, ignore_index=-100, reduction="none")
    nll = (ce * (lab != -100)).sum(1) / (lab != -100).sum(1)
    return h, nll, out.logits


def kl(logits: torch.Tensor, refs: list[torch.Tensor]) -> torch.Tensor:
    """KL(partida || actual) por posición real, promediado: frena que el ajuste conjunto cambie lo que el modelo sabe."""
    lp = torch.log_softmax(logits.float(), -1)
    tot = 0.0
    for j, r in enumerate(refs):
        r = r.to(lp.device).float()
        tot = tot + (r.exp() * (r - lp[j, : len(r)])).sum()
    return tot / sum(len(r) for r in refs)


def predict(model, tok, head, facts: list[dict]) -> np.ndarray:
    with torch.no_grad():
        return head(torch.tensor(b27.prompt_states(model, tok, facts, LAYERS), device=DEVICE)).cpu().numpy()


def train_head(model, tok, head, train: list[dict], val: list[dict], bank: list[dict], refs, joint: bool, key: str) -> dict:
    """AdamW sobre la cabeza (y el tronco si `joint`); vuelve al punto con mejor rho en validación interna, nunca elige con el banco."""
    items = [b24.fwd(x) for x in train]
    trunk = [p for p in model.parameters() if p.requires_grad] if joint else []
    groups = [{"params": list(head.parameters()), "lr": HEAD_LR, "weight_decay": HEAD_WD}]
    opt = torch.optim.AdamW(groups + ([{"params": trunk, "lr": TRUNK_LR, "weight_decay": 0.0}] if joint else []))
    y0 = b17.def_losses(model, tok, items)
    ym, ys = float(y0.mean()), float(y0.std()) + 1e-8
    rng, hist, best = random.Random(key), [], None
    torch.manual_seed(0)
    for step in range(1, STEPS + 1):
        pick = rng.sample(range(len(items)), min(b17.BATCH, len(items)))
        model.eval()  # sin dropout de atención (0,1): def_losses y prompt_states dejan el modelo en modo train, y el dropout rompía los rasgos y el KL
        with amp(DEVICE):
            with torch.set_grad_enabled(joint):  # congelada: el tronco ni arma el grafo
                h, nll, logits = parts(model, tok, [items[i] for i in pick])
            loss = ((head(h) - (nll.detach() - ym) / ys) ** 2).mean()
            if joint:
                loss = loss + BETA * kl(logits, [refs[i] for i in pick])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % EVAL_EVERY == 0:
            yv, yb = (b17.def_losses(model, tok, [b24.fwd(x) for x in s]) for s in (val, bank))
            rv, rb = rank_corr(predict(model, tok, head, val), yv), rank_corr(predict(model, tok, head, bank), yb)
            hist.append({"paso": step, "rho_val": round(rv, 4), "rho_banco": round(rb, 4), "perdida": round(float(loss.detach()), 4)})
            if best is None or rv > best[0]:
                best = (rv, step, snapshot(model, head, joint))
    if joint:
        model.load_state_dict(best[2]["tronco"], strict=False)
    head.load_state_dict(best[2]["cabeza"])
    model.train()
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"historia": hist, "paso_elegido": best[1], "rho_val": round(best[0], 4)}


def boot_delta(pf: np.ndarray, yf: np.ndarray, pj: np.ndarray, yj: np.ndarray) -> list[float]:
    """Diferencia pareada rho(conjunta) − rho(congelada) en el banco, con IC90 por bootstrap de hechos."""
    rng = np.random.default_rng(0)
    d = [rank_corr(pj[i], yj[i]) - rank_corr(pf[i], yf[i]) for i in (rng.integers(0, len(pf), len(pf)) for _ in range(BOOT))]
    return [round(rank_corr(pj, yj) - rank_corr(pf, yf), 4), round(float(np.percentile(d, 5)), 4), round(float(np.percentile(d, 95)), 4)]


def decide(delta: list[float]) -> str:
    m, lo, hi = delta
    return "hereda" if m >= GAIN and lo > 0 else "no_hereda" if hi < GAIN else "indeterminado"


def main(path: str, bank_path: str, cand_path: str, model=None, tok=None) -> None:
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    key = lambda x: (x["cabeza"], x["propiedad"])
    in_bank = {key(x) for x in facts}
    rest = [x for x in cands if key(x) not in in_bank]
    rest = random.Random("b28:validacion").sample(rest, len(rest))
    val, train = rest[: len(rest) // VAL_FRAC], rest[len(rest) // VAL_FRAC :]
    n = len(rest)
    X0 = b27.prompt_states(model, tok, rest + facts, LAYERS)
    y0 = b17.def_losses(model, tok, [b24.fwd(x) for x in rest + facts])
    y0_bank = y0[n:]
    f1 = {"n_entrena": len(train), "n_val": len(val), "n_banco": len(facts), "ridge_antes": round(rank_corr(b27.ridge(X0[:n], y0[:n], X0[n:]), y0_bank), 4)}
    Xt = X0[len(val) : n]
    head = Head(Xt.mean(0), Xt.std(0) + 1e-8).to(DEVICE)
    f1["congelada"] = train_head(model, tok, head, train, val, facts, None, False, "b28:cabeza")
    p_frozen = predict(model, tok, head, facts)
    f1["congelada"]["rho_banco"] = round(rank_corr(p_frozen, y0_bank), 4)
    # la conjunta arranca de la cabeza congelada (LP-FT, Kumar et al. 2022): una cabeza al azar deforma el tronco en los primeros pasos
    refs = ref_logprobs(model, tok, [b24.fwd(x) for x in train])
    # shortcut: la fase 1 no se guarda entre VMs; una VM nueva la repite y la fase 2 puede mezclar dos modelos conjuntos casi iguales
    f1["conjunta"] = train_head(model, tok, head, train, val, facts, refs, True, "b28:cabeza")
    p_joint = predict(model, tok, head, facts)
    y1 = b17.def_losses(model, tok, [b24.fwd(x) for x in rest + facts])
    y1_bank = y1[n:]
    f1["conjunta"]["rho_banco"] = round(rank_corr(p_joint, y1_bank), 4)
    X1 = b27.prompt_states(model, tok, rest + facts, LAYERS)
    f1["ridge_despues"] = round(rank_corr(b27.ridge(X1[:n], y1[:n], X1[n:]), y1_bank), 4)
    f1["banco_antes_despues"] = {"spearman": round(rank_corr(y0_bank, y1_bank), 4), "cambio_medio": round(float(np.abs(y1_bank - y0_bank).mean()), 4)}
    f1["delta"] = boot_delta(p_frozen, y0_bank, p_joint, y1_bank)
    f1["C1"] = "pasa" if f1["congelada"]["rho_banco"] >= C1_MIN else "falla"
    moved = f1["banco_antes_despues"]
    f1["I1"] = "pasa" if moved["spearman"] >= I1_RHO and moved["cambio_medio"] <= I1_SHIFT else "falla"
    f1["D1"] = decide(f1["delta"])
    res["config"] = {"capas": list(LAYERS), "pasos": STEPS, "eval_cada": EVAL_EVERY, "lr_cabeza": HEAD_LR, "wd_cabeza": HEAD_WD, "beta_kl": BETA, "lr_tronco": TRUNK_LR,
                     "lr": b17.LR, "batch": b17.BATCH, "dosis": list(b24.DOSES), "conjuntos": b24.SETS, "puerta": [b27.LOW, b27.SPAN]}  # fmt: skip
    res["fase1"] = f1
    res["prediccion"] = {"cabeza_conjunta": p_joint.round(4).tolist(), "cabeza_congelada": p_frozen.round(4).tolist()}
    json.dump(res, open(path, "w"), ensure_ascii=False)
    summary = {k: f1[k] for k in ("ridge_antes", "ridge_despues", "banco_antes_despues", "delta", "C1", "I1", "D1")}
    print(json.dumps(summary | {a: f1[a]["rho_banco"] for a in ("congelada", "conjunta")}), flush=True)
    if (f1["C1"], f1["I1"], f1["D1"]) != ("pasa", "pasa", "hereda"):
        print(json.dumps({"FIN": 0, "sin_fase2": [f1["C1"], f1["I1"], f1["D1"]]}), flush=True)
        return
    base_i, base_v = b24.losses(model, tok, facts)
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    w_head, w_orc = b27.gate(p_joint), b27.gate(base_i[:, 0])
    res["pesos"] = {"cabeza": w_head.round(4).tolist(), "oraculo": w_orc.round(4).tolist()}
    arms = {"cabeza": ("ida", w_head, None), "oraculo": ("ida", w_orc, None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


def episode_ridge(Zs: torch.Tensor, ys: torch.Tensor, Zq: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge en forma dual resuelta sobre el soporte y aplicada a la consulta; diferenciable (R2D2, Bertinetto et al. 2019)."""
    m = Zs.mean(0)
    A, B = Zs - m, Zq - m
    K = A @ A.T + alpha * torch.eye(len(A), device=A.device)
    return ys.mean() + B @ (A.T @ torch.linalg.solve(K, ys - ys.mean()))


def train_ridge_trunk(model, tok, mu: np.ndarray, sd: np.ndarray, alpha: float, train: list[dict], refs, bank: list[dict],
                      y0_bank: np.ndarray, key: str, anchors: tuple | None = None, beta: float = BETA,
                      steps: int = STEPS) -> tuple[list[dict], tuple | None]:
    """El tronco aprende a que una ridge cerrada lea su propia NLL; sin cabeza que aprender. Guarda el último punto que cumple I1.
    `anchors` (B28f): (oraciones, log-probs de partida) de hechos que la lectura no entrena; el KL los suma en cada lote."""
    items = [b24.fwd(x) for x in train]
    trunk = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trunk, lr=TRUNK_LR, weight_decay=0.0)
    y0 = b17.def_losses(model, tok, items)
    ym, ys = float(y0.mean()), float(y0.std()) + 1e-8
    mu_t, sd_t = (torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in (mu, sd))
    rng, hist, keep, s = random.Random(key), [], None, EPISODE[0]
    for step in range(1, steps + 1):
        pick = rng.sample(range(len(items)), sum(EPISODE))
        batch, rs = [items[i] for i in pick], [refs[i] for i in pick]
        if anchors:
            apick = rng.sample(range(len(anchors[0])), ANCHOR_BATCH)
            batch, rs = batch + [anchors[0][j] for j in apick], rs + [anchors[1][j] for j in apick]
        model.eval()  # sin dropout de atención, como B28b
        with amp(DEVICE):
            h, nll, logits = parts(model, tok, batch)
        Z, t = (h[: len(pick)].float() - mu_t) / sd_t, (nll[: len(pick)].detach().float() - ym) / ys
        fit = ((episode_ridge(Z[:s], t[:s], Z[s:], alpha) - t[s:]) ** 2).mean()
        loss = fit + beta * kl(logits, rs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % CHECK_D == 0:
            yb = b17.def_losses(model, tok, [b24.fwd(x) for x in bank])
            moved = {"spearman": round(rank_corr(y0_bank, yb), 4), "cambio_medio": round(float(np.abs(yb - y0_bank).mean()), 4)}
            ok = moved["spearman"] >= I1_RHO and moved["cambio_medio"] <= I1_SHIFT
            hist.append({"paso": step, "ajuste": round(float(fit.detach()), 4), "perdida": round(float(loss.detach()), 4), **moved, "I1": ok})
            if ok:
                keep = (step, snapshot(model, None, True)["tronco"])
    model.train()
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return hist, keep


def decide_replicas(reps: list[dict]) -> str:
    if any(r["I1"] != "pasa" for r in reps):
        return "sin_medida"
    ds = [r["delta"] for r in reps]
    if all(m >= GAIN and lo > 0 for m, lo, _ in ds):
        return "hereda"
    return "no_hereda" if all(hi < GAIN for _, _, hi in ds) else "indeterminado"


def main_d(path: str, bank_path: str, cand_path: str, model=None, tok=None, anchor_path: str | None = None) -> None:
    """B28d: la ridge cerrada lee la duda antes y después de entrenar el tronco contra una ridge por episodio; dos réplicas.
    B28f (con `anchor_path`): KL con beta 10 sobre las oraciones de entrenamiento y sobre hechos ancla fuera de los candidatos; sin fase 2."""
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    key = lambda x: (x["cabeza"], x["propiedad"])
    in_bank = {key(x) for x in facts}
    train = [x for x in cands if key(x) not in in_bank]
    n = len(train)
    X0 = b27.prompt_states(model, tok, train + facts, LAYERS)
    y0 = b17.def_losses(model, tok, [b24.fwd(x) for x in train + facts])
    y0_bank = y0[n:]
    p0, alpha = b27.ridge(X0[:n], y0[:n], X0[n:], return_alpha=True)
    f1 = {"n_entrena": n, "n_banco": len(facts), "alpha": alpha, "ridge_antes": round(rank_corr(p0, y0_bank), 4)}
    f1["C1"] = "pasa" if f1["ridge_antes"] >= C1_MIN_D else "falla"
    refs = ref_logprobs(model, tok, [b24.fwd(x) for x in train])
    anchors, beta, steps = None, BETA, STEPS
    if anchor_path:
        seen = {key(x) for x in cands} | in_bank
        pool = [x for v in json.load(open(anchor_path, encoding="utf-8"))["hechos"].values() for x in v if key(x) not in seen]
        a_items = [b24.fwd(x) for x in random.Random("b28f:anclas").sample(pool, min(N_ANCHORS, len(pool)))]
        anchors, beta, steps = (a_items, ref_logprobs(model, tok, a_items)), BETA_F, STEPS_F
    state0 = snapshot(model, None, True)["tronco"]
    mu, sd = X0[:n].mean(0), X0[:n].std(0) + 1e-8
    reps, keep0 = [], None
    for r in range(REPLICAS):
        model.load_state_dict(state0, strict=False)
        hist, keep = train_ridge_trunk(model, tok, mu, sd, alpha, train, refs, facts, y0_bank, f"b28d:r{r}", anchors, beta, steps)
        rep = {"historia": hist, "I1": "falla" if keep is None else "pasa"}
        if keep is not None:
            model.load_state_dict(keep[1], strict=False)
            X1 = b27.prompt_states(model, tok, train + facts, LAYERS)
            y1 = b17.def_losses(model, tok, [b24.fwd(x) for x in train + facts])
            p1 = b27.ridge(X1[:n], y1[:n], X1[n:])
            rep |= {"paso": keep[0], "ridge_despues": round(rank_corr(p1, y1[n:]), 4), "delta": boot_delta(p0, y0_bank, p1, y1[n:]),
                    "prediccion": p1.round(4).tolist()}  # fmt: skip
            keep0 = keep0 or keep
        reps.append(rep)
    f1["replicas"], f1["D1"] = reps, decide_replicas(reps)
    res["config"] = {"capas": list(LAYERS), "pasos": steps, "episodio": list(EPISODE), "medir_cada": CHECK_D, "replicas": REPLICAS, "beta_kl": beta,
                     "anclas": None if anchors is None else [len(anchors[0]), ANCHOR_BATCH],
                     "lr_tronco": TRUNK_LR, "lr": b17.LR, "batch": b17.BATCH, "dosis": list(b24.DOSES), "conjuntos": b24.SETS, "puerta": [b27.LOW, b27.SPAN]}  # fmt: skip
    res["fase1"], res["prediccion_antes"] = f1, p0.round(4).tolist()
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps({"ridge_antes": f1["ridge_antes"], "C1": f1["C1"], "D1": f1["D1"],
                      "replicas": [{k: r.get(k) for k in ("I1", "paso", "ridge_despues", "delta")} for r in reps]}), flush=True)  # fmt: skip
    if (f1["C1"], f1["D1"]) != ("pasa", "hereda") or anchors:  # B28f: la fase 2 sale del tope de 2 unidades; la decide el usuario
        print(json.dumps({"FIN": 0, "sin_fase2": [f1["C1"], f1["D1"]]}), flush=True)
        return
    model.load_state_dict(keep0[1], strict=False)  # la fase 2 parte de la réplica 0
    base_i, base_v = b24.losses(model, tok, facts)
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    w = b27.gate(np.array(reps[0]["prediccion"]))
    res["pesos"] = {"cabeza": w.round(4).tolist()}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms={"cabeza": ("ida", w, None)})


if __name__ == "__main__":
    if sys.argv[1] in ("d", "f"):
        main_d(*sys.argv[2:5], anchor_path=sys.argv[5] if sys.argv[1] == "f" else None)
    else:
        main(*sys.argv[1:4])
