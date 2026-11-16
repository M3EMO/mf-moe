"""B38: ¿una capacidad nueva y esparsa, como el cuerpo fungiforme de la mosca, protege el resto mejor que el ajuste común y que EWC?
Prerregistro en el ticket 17 de mf-moe-generalidad. Banco, conjuntos (tag «b25»), semillas, lotes y dosis de la ida de B25.
- ewc: el ajuste de B25 más λ/2 · Σ F (θ − θ*)², con la Fisher empírica de candidatos fuera del banco (media 1, el 10 % más alto).
- mosca: modelo base congelado; tras la capa LAYER suma Kenyon: proyección fija binaria a EXPAND·d unidades, ganador por k y salida entrenable.
- mosca_densa: la misma expansión sin ganador por k.
El conjunto 0 elige λ y la tasa; los conjuntos 1 a SETS − 1 miden. comun es la ida de B25 y no corre acá.
Uso: python -m colab.b38_capacidad main SALIDA.json BANCO.json B25_IDA_D40.json CANDIDATOS.json
"""

import json
import os
import random
import sys

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

LAYER, EXPAND, CLAWS, ACTIVE = 16, 40, 40, 0.05
DOSES, LAMBDAS, LRS, FISHER_TOP, MAX_NEW = (10, 40), (1.0, 100.0, 1e4), (1e-4, 1e-3, 1e-2), 0.10, 0.5


class Kenyon(torch.nn.Module):
    """h + salida · kWTA(relu(proyección fija binaria · rmsnorm(h))). Con la salida en cero no cambia nada."""

    def __init__(self, d: int, sparse: bool):
        super().__init__()
        m, g = EXPAND * d, torch.Generator().manual_seed(38)
        idx = torch.rand(m, d, generator=g).argsort(1)[:, : min(CLAWS, d)]  # cada unidad mira CLAWS dimensiones al azar
        self.register_buffer("w_in", torch.zeros(m, d).scatter_(1, idx, 1.0))
        self.out = torch.nn.Parameter(torch.zeros(m, d))
        self.k = max(1, round(ACTIVE * m)) if sparse else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        a = torch.relu(x @ self.w_in.T)
        if self.k:
            a = a * (a >= a.topk(self.k, -1).values[..., -1:])
        return h + (a @ self.out).to(h.dtype)


def attach(model, mod: Kenyon):
    def hook(_, __, out):
        return (mod(out[0]), *out[1:]) if isinstance(out, tuple) else mod(out)

    return model.model.layers[LAYER].register_forward_hook(hook)


def outside(cands: list[dict], facts: list[dict]) -> list[dict]:
    key = lambda x: (x["cabeza"], x["propiedad"], x["valor"])
    taken = {key(x) for x in facts}
    return [x for x in cands if key(x) not in taken]


def fisher(model, tok, items: list[tuple[str, str]], top: float = FISHER_TOP) -> dict:
    """Fisher empírica por parámetro entrenable, normalizada a media 1; guarda solo las entradas sobre el cuantil 1 − top."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    F = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    for it in items:
        ids, mask, labels = b17.batch(tok, [it])
        with amp(DEVICE):
            model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                F[n] += p.grad.float() ** 2
        model.zero_grad(set_to_none=True)
    count = sum(f.numel() for f in F.values())
    mean = sum(float(f.sum()) for f in F.values()) / count
    dev = next(iter(F.values())).device
    g = torch.Generator(device=dev).manual_seed(38)
    # shortcut: el cuantil sale de una muestra de hasta 10 M entradas, proporcional al tamaño de cada tensor
    sample = torch.cat([f.flatten()[torch.randint(f.numel(), (max(1, f.numel() * 10_000_000 // count),), generator=g, device=dev)] for f in F.values()])
    thr = float(torch.quantile(sample[:16_000_000], 1 - top)) if top < 1 else -1.0
    sparse = {}
    for n, f in F.items():
        idx = (f.flatten() > thr).nonzero().squeeze(1)
        sparse[n] = (idx, f.flatten()[idx] / mean, params[n].detach().flatten()[idx].float().clone())
    del F
    return sparse


def ewc_hooks(model, sparse: dict, lam: float) -> list:
    """Suma λ · F · (θ − θ*) al gradiente de las entradas guardadas, antes del recorte de `b24.train`."""
    hooks = []
    for n, p in model.named_parameters():
        if n in sparse:
            idx, f, theta = sparse[n]

            def h(p, idx=idx, f=f, theta=theta):
                p.grad.view(-1).index_add_(0, idx, (lam * f * (p.detach().view(-1)[idx].float() - theta)).to(p.grad.dtype))

            hooks.append(p.register_post_accumulate_grad_hook(h))
    return hooks


def rest_rise(r: dict) -> float:
    return float(np.delete(np.array(r["subida_ida"]), r["sobrescritos"]).mean())


def choose(res: dict, arm: str, values: tuple) -> tuple[float, bool]:
    """Regla del ticket: entre los valores que aprenden (pérdida nueva ≤ MAX_NEW en la dosis mayor), el de menor daño al resto."""
    rows = [(v, res["corridas"][f"s0|{arm}={v}|d{max(DOSES)}"]) for v in values]
    ok = [(rest_rise(r), v) for v, r in rows if np.mean(r["perdida_nuevo"]) <= MAX_NEW]
    return (min(ok)[1], True) if ok else (min((float(np.mean(r["perdida_nuevo"])), v) for v, r in rows)[1], False)


def main(path: str, bank_path: str, b25_path: str, cands_path: str, model=None, tok=None) -> None:
    facts, B, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, b25_path, cands_path))
    B = B["corridas"]
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, _ = b24.losses(model, tok, facts)
    cands = outside(cands, facts)
    res["config"] = {"n_hechos": len(facts), "capa": LAYER, "expansion": EXPAND, "garras": CLAWS, "activas": ACTIVE, "dosis": list(DOSES),
                     "lambdas": list(LAMBDAS), "tasas": list(LRS), "fisher_top": FISHER_TOP, "fisher_hechos": len(cands), "lr": b17.LR}  # fmt: skip
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if n in trainable}
    n = len(facts)
    n_over = n // b24.SETS
    order = random.Random("b25:conjuntos").sample(range(n), n)  # los conjuntos de b24.stage2 con el tag «b25»
    sparse = None

    def run(s: int, name: str, kind: str, value: float) -> None:
        nonlocal sparse
        if f"s{s}|{name}|d{max(DOSES)}" in res["corridas"]:
            return
        over = sorted(order[s * n_over : (s + 1) * n_over])
        assert B[f"s{s}|ida|d40"]["sobrescritos"] == over, "los conjuntos no coinciden con B25"
        new = [b24.fwd(facts[i], facts[i]["valor_nuevo"]) for i in over]
        lr = b17.LR
        if kind == "ewc":
            if sparse is None:
                sparse = fisher(model, tok, [b24.fwd(x) for x in cands])
                res["fisher"] = {"entradas": int(sum(len(v[0]) for v in sparse.values()))}
            undo = ewc_hooks(model, sparse, value)
        else:
            k = Kenyon(model.config.hidden_size, kind == "mosca").to(next(model.parameters()).device)
            model.requires_grad_(False)
            model.kenyon = k  # registrada en el modelo para que `b24.train` encuentre su salida
            undo, lr = [attach(model, k)], value

        def hook(step: int) -> None:
            if step in DOSES:
                L_i, _ = b24.losses(model, tok, facts)
                r = res["corridas"][f"s{s}|{name}|d{step}"] = {"conjunto": s, "brazo": name, "tipo": kind, "valor": value, "sobrescritos": over,
                                                               "pasos": step, "subida_ida": (L_i[:, 0] - base_i[:, 0]).round(4).tolist(),
                                                               "perdida_nuevo": b17.def_losses(model, tok, new).round(4).tolist()}  # fmt: skip
                print(json.dumps({"corrida": f"s{s}|{name}|d{step}", "resto": round(rest_rise(r), 3), "nuevos": round(float(np.mean(r["perdida_nuevo"])), 3)}), flush=True)

        old = b17.LR
        try:
            b17.LR = lr
            torch.manual_seed(1000 + s)
            b24.train(model, tok, new, float("-inf"), max(DOSES), b17.CHECK_EVERY, f"b25:lotes:{s}:ida", hook)
        finally:
            b17.LR = old
            for h in undo:
                h.remove()
            if kind != "ewc":
                del model.kenyon
            for nm, p in model.named_parameters():
                p.requires_grad_(nm in trainable)
            model.load_state_dict(state, strict=False)
        json.dump(res, open(path, "w"), ensure_ascii=False)

    for v in LAMBDAS:
        run(0, f"ewc={v}", "ewc", v)
    for v in LRS:
        run(0, f"mosca={v}", "mosca", v)
    (lam, lam_ok), (lr, lr_ok) = choose(res, "ewc", LAMBDAS), choose(res, "mosca", LRS)
    res["eleccion"] = {"ewc": lam, "ewc_aprende": lam_ok, "mosca": lr, "mosca_aprende": lr_ok}
    print(json.dumps({"eleccion": res["eleccion"]}), flush=True)
    for s in range(1, b24.SETS):
        run(s, "ewc", "ewc", lam)
        run(s, "mosca", "mosca", lr)
        run(s, "mosca_densa", "densa", lr)
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps({"FIN": len(res["corridas"])}), flush=True)


if __name__ == "__main__":
    {"main": main}[sys.argv[1]](*sys.argv[2:])
