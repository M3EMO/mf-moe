"""B12c Fase A (mf-moe-5tq.29): ¿las primitivas se especializan cuando se distinguen más? Hipótesis del usuario: en
B12 se parecían demasiado (mismo vocabulario 0/1, mismo formato). Plan y reglas de decisión: notas del bead.

Dial p: cada primitiva escribe las posiciones propias (las primeras round(16p) de un orden fijo) con su alfabeto;
el resto queda en 0/1 (mf_moe.bits.write). Solo primitivas, 16 expertos. Variantes por semilla:
- base con p ∈ P_GRID;
- sin_balanceo (u = 0) y router_azar (router congelado al azar) con p = 0 y p = 1, con los pasos de su base como tope.
Medidas, por capa y solo en la respuesta:
- uso diferencial y número de padres distintos (los de B12b);
- información mutua experto × primitiva, total y condicionada al token sobre los bits 0/1 (los compartidos), con un
  nulo que permuta la primitiva entre secuencias;
- ablación: caída de exactitud de cada primitiva al sacar cada experto del top-k;
- coseno entre los estados medios de las primitivas (crudo y centrado) y Gini del uso.
Uso: .venv/bin/python -m colab.b12c_distincion SALIDA.json [pasos máximos] [semilla]
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch

from colab.b9_bits import DEVICE, EVAL_EVERY, SINGLES, SIZE, _mi, eval_sets, routing, train
from colab.b12_raices import E0, _usage, parents
from mf_moe.model import build, routers
from mf_moe.train import adamw, encode, scores

P_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTROLS = ("sin_balanceo", "router_azar")
# Tras la primera pasada: con p = 1 `par` no aprende en 20.000 pasos (5 de 5 semillas), y un router congelado al azar
# ya da el 85 % de la IM. Controles agregados: router al azar en las p intermedias, y base sin corte temprano
# ("largo", 20.000 pasos) para separar el largo del entrenamiento de la distinción.
EXTRA = [(p, "router_azar") for p in (0.25, 0.5, 0.75)] + [(p, "largo") for p in (0.0, 0.75)]
SHARED = tuple(b"01")
N_NULL = 200


def _tokens(exs) -> np.ndarray:
    """Token de cada posición de la respuesta, en el orden de `routing`."""
    ids, labels = encode(exs, "cpu")
    return ids.reshape(-1)[(labels != -100).reshape(-1)].numpy()


def _cond_mi(lab: np.ndarray, idx: np.ndarray, tok: np.ndarray) -> float | None:
    """I(experto; primitiva | token) / H(primitiva), promediada sobre los bits 0/1 con su peso."""
    sel = [(tok == s) for s in SHARED if (tok == s).sum()]
    if not sel:
        return None
    return float(sum(m.sum() * _mi(lab[m], idx[m], len(SINGLES)) for m in sel) / sum(m.sum() for m in sel))


def _gini(u: np.ndarray) -> float:
    return float(np.abs(u[:, None] - u[None]).sum() / (2 * len(u) * u.sum()))


def _cos(M: np.ndarray) -> float:
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = M @ M.T
    return float(C[~np.eye(len(C), dtype=bool)].mean())


@torch.no_grad()
def ablation(model, ev: dict) -> list:
    """[capas][expertos][primitivas]: caída de exactitud al excluir el experto del top-k de esa capa."""
    base = {t: scores(model, ev[t], DEVICE)[0] for t in SINGLES}
    out = []
    for r in routers(model):
        rows = []
        for e in range(E0):
            keep = r.bias[e].item()
            r.bias[e] = -1e9
            rows.append([round(base[t] - scores(model, ev[t], DEVICE)[0], 3) for t in SINGLES])
            r.bias[e] = keep
        out.append(rows)
    return out


def measure(model, ev: dict, seed: int) -> dict:
    rt = routing(model, ev)
    n_ans = len(rt[SINGLES[0]]["idx"]) // len(ev[SINGLES[0]])  # tokens de respuesta por secuencia
    assert all(len(rt[t]["idx"]) == n_ans * len(ev[t]) for t in SINGLES)
    lab = np.concatenate([np.full(len(rt[t]["idx"]), i) for i, t in enumerate(SINGLES)])
    idx = np.concatenate([rt[t]["idx"] for t in SINGLES])
    tok = np.concatenate([_tokens(ev[t]) for t in SINGLES])
    seq = lab[::n_ans]
    rng = np.random.default_rng(seed)
    perms = [np.repeat(rng.permutation(seq), n_ans) for _ in range(N_NULL)]
    top, lift = parents(rt, E0)
    out = {"padres": {t: top[t].tolist() for t in SINGLES}, "uso_diferencial": lift, "capas": []}
    for L in range(idx.shape[1]):
        mi, cmi = _mi(lab, idx[:, L], len(SINGLES)), _cond_mi(lab, idx[:, L], tok)
        null_mi = [_mi(q, idx[:, L], len(SINGLES)) for q in perms]
        null_cmi = [_cond_mi(q, idx[:, L], tok) for q in perms] if cmi is not None else None
        H = np.stack([rt[t]["h"][L].mean(0) for t in SINGLES])
        out["capas"].append({
            "padres_distintos": len({int(top[t][L]) for t in SINGLES}),
            "uso_diferencial_medio": round(float(np.mean([lift[t][L] for t in SINGLES])), 4),
            "im": round(mi, 4), "im_nulo_p95": round(float(np.percentile(null_mi, 95)), 4),
            "im_cond_token": None if cmi is None else round(cmi, 4),
            "im_cond_token_nulo_p95": None if cmi is None else round(float(np.percentile(null_cmi, 95)), 4),
            "coseno": round(_cos(H), 4), "coseno_centrado": round(_cos(H - H.mean(0)), 4),
            "gini": round(_gini(_usage(idx, E0)[L]), 4),
        })  # fmt: skip
    out["ablacion"] = ablation(model, ev)
    return out


def run(p: float, variant: str, max_steps: int, seed: int) -> dict:
    ev = eval_sets(SINGLES, p=p)
    model = build(E0, seed=seed, **SIZE).to(DEVICE)
    for r in routers(model):
        if variant == "sin_balanceo":
            r.u = 0.0
        if variant == "router_azar":
            r.weight.requires_grad_(False)
    opt = adamw(model)
    every = max_steps if variant == "largo" else EVAL_EVERY  # largo evalúa solo al final: no corta al dominar
    r = train(model, opt, SINGLES, ev, max_steps, random.Random(seed), f"{variant}:{p}", p=p, every=every)
    return {"p": p, "variante": variant, "pasos": r["pasos"], "domino": r["domino"], "historia": r["historia"],
            **measure(model, ev, seed)}  # fmt: skip


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "pasos_maximos": max_steps, "corridas": []}
    if os.path.exists(path):  # retoma: no repite lo que ya está en la salida
        res = json.load(open(path))
    done = {(c["p"], c["variante"]): c for c in res["corridas"]}
    plan = [(p, "base") for p in P_GRID] + [(p, v) for v in CONTROLS for p in (0.0, 1.0)] + EXTRA
    for p, v in plan:
        if (p, v) in done:
            continue
        # un control entrena a lo sumo los pasos de su base: mismo presupuesto, y sin correr 20.000 pasos si no domina
        out = run(p, v, max_steps if v in ("base", "largo") else done[(p, "base")]["pasos"], seed)
        done[(p, v)] = out
        res["corridas"].append(out)
        print(json.dumps({"p": p, "variante": v, "pasos": out["pasos"],
                          "padres_distintos": [c["padres_distintos"] for c in out["capas"]],
                          "im": [c["im"] for c in out["capas"]], "im_cond": [c["im_cond_token"] for c in out["capas"]]}),
              flush=True)  # fmt: skip
        json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
