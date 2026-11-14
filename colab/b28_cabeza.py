"""B28: ¿una cabeza de certeza entrenada DENTRO del modelo lee la duda mejor que la misma cabeza sobre un tronco congelado?
Prerregistro en el ticket 10 (mf-moe-generalidad), opción A. La cabeza es lineal sobre el estado del último token de la pregunta
(capas de B27c) y predice la NLL propia del valor real. Entrena con los 255 candidatos fuera del banco; el banco (206) es la prueba.
- congelada: solo la cabeza aprende; el tronco no cambia (es la sonda de B27c entrenada con AdamW).
- conjunta: la cabeza y las capas 6-23 aprenden juntas; un KL contra el modelo de partida frena el cambio de conocimiento.
Fase 2 (solo si la conjunta hereda, el control pasa y el banco no se movió): B27 sobre el modelo conjunto, brazos cabeza y oráculo.
Uso: python -m colab.b28_cabeza SALIDA.json BANCO.json CANDIDATOS.json
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
    return {"tronco": trunk, "cabeza": {k: v.detach().to("cpu", copy=True) for k, v in head.state_dict().items()}}


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
    opt = torch.optim.AdamW(groups + ([{"params": trunk, "lr": b17.LR, "weight_decay": 0.0}] if joint else []))
    y0 = b17.def_losses(model, tok, items)
    ym, ys = float(y0.mean()), float(y0.std()) + 1e-8
    rng, hist, best = random.Random(key), [], None
    torch.manual_seed(0)
    for step in range(1, STEPS + 1):
        pick = rng.sample(range(len(items)), min(b17.BATCH, len(items)))
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
    init = {k: v.clone() for k, v in head.state_dict().items()}
    f1["congelada"] = train_head(model, tok, head, train, val, facts, None, False, "b28:cabeza")
    p_frozen = predict(model, tok, head, facts)
    f1["congelada"]["rho_banco"] = round(rank_corr(p_frozen, y0_bank), 4)
    head.load_state_dict(init)
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
    res["config"] = {"capas": list(LAYERS), "pasos": STEPS, "eval_cada": EVAL_EVERY, "lr_cabeza": HEAD_LR, "wd_cabeza": HEAD_WD, "beta_kl": BETA,
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


if __name__ == "__main__":
    main(*sys.argv[1:4])
