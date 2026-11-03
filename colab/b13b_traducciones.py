"""B13b (mf-moe-5tq.27): B13 con la medida de Doshi et al. (arXiv 2610.07168): traducciones, no medias por lengua.

B13 (d0430ee) midió la distancia entre el uso medio de expertos de cada lengua: dio una estrella, sin árbol, también
en los estados. Doshi compara la misma oración latente escrita en dos lenguas, oración por oración. Su teoría: el
parecido aparece cuando el modelo aprendió tantos niveles como la distancia entre las lenguas.
Entrenamiento igual que B13 (oraciones sueltas de lenguas al azar). Evaluación: EVAL_FAM oraciones latentes escritas
en las 8 lenguas (`hierarchy.sample_family`). Por capa y por distancia d = 1..3 en el árbol de lenguas:
- ruteo: expertos compartidos entre tokens alineados de dos traducciones, menos lo mismo con oraciones barajadas
  (no traducciones) de las mismas lenguas;
- kmeans: lo mismo con k-means de 16 sobre los estados que entran al router, top-2 (control «Myth»);
- ruteo_oracion y kmeans_oracion: vecinos mutuos del histograma de expertos de cada oración (como estados);
- estados: vecinos mutuos (k = 10) de la media de tokens de cada oración (la medida de Doshi), menos el barajado.
Uso: .venv/bin/python -m colab.b13b_traducciones SALIDA.json [pasos] [semilla] [variantes separadas por coma] [V M]
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from colab.b9_bits import BATCH, DEVICE, SIZE
from colab.b12_raices import E0
from colab.b13_arbol import probe
from mf_moe import hierarchy as h
from mf_moe.model import build, routers
from mf_moe.train import adamw

CHECKPOINTS = (0, 1000, 3000, 10000, 20000)
EVAL_FAM = 256


def _excess(per_lang: list, score, shuffle: np.ndarray) -> list[float]:
    """score(traducción) − score(barajado) por par de lenguas, promediado por distancia."""
    M = np.zeros((h.N_LANG, h.N_LANG))
    for a, b in itertools.combinations(range(h.N_LANG), 2):
        M[a, b] = M[b, a] = score(per_lang[a], per_lang[b]) - score(per_lang[a], per_lang[b][shuffle])
    return h.by_distance(M)


def _agree(A: np.ndarray, B: np.ndarray) -> float:
    return h.agreement(A.reshape(-1, A.shape[-1]), B.reshape(-1, B.shape[-1]))


def measure(model, ids: torch.Tensor, seed: int) -> dict:
    """`ids` [EVAL_FAM · N_LANG, 1 + SEQ], ordenadas por familia y después por lengua."""
    loss, idx, H = probe(model, ids)
    shuffle = np.random.default_rng(seed).permutation(EVAL_FAM)
    tok = lambda X: X.reshape(EVAL_FAM, h.N_LANG, h.SEQ, -1)  # noqa: E731
    out = {"perdida": round(loss, 4), "capas": []}
    for L in range(len(idx)):
        rt, kt = tok(idx[L]), tok(h.kmeans_topk(H[L], E0, idx.shape[2], seed))
        st = tok(H[L]).mean(2)
        # histograma de expertos de cada oración [familias, lenguas, E0]: el ruteo a la escala de la oración
        hist = lambda X: np.eye(E0)[X].sum((2, 3))  # noqa: E731
        ru, ku = hist(rt), hist(kt)
        out["capas"].append({
            "ruteo": _excess([rt[:, g] for g in range(h.N_LANG)], _agree, shuffle),
            "kmeans": _excess([kt[:, g] for g in range(h.N_LANG)], _agree, shuffle),
            "ruteo_oracion": _excess([ru[:, g] for g in range(h.N_LANG)], h.mknn, shuffle),
            "kmeans_oracion": _excess([ku[:, g] for g in range(h.N_LANG)], h.mknn, shuffle),
            "estados": _excess([st[:, g] for g in range(h.N_LANG)], h.mknn, shuffle),
        })  # fmt: skip
    return out


def run(variant: str, max_steps: int, seed: int) -> dict:
    tabs = h.rules(seed)
    syms, _ = h.sample_family(tabs, EVAL_FAM, np.random.default_rng(10**6 + seed))
    lang = np.tile(np.arange(h.N_LANG), EVAL_FAM)
    ev = torch.tensor(h.ids(syms.reshape(-1, h.SEQ), lang, False), device=DEVICE)
    model = build(E0, seed=seed, **SIZE).to(DEVICE)
    for r in routers(model):
        if variant == "sin_balanceo":
            r.u = 0.0
        if variant == "router_azar":
            r.weight.requires_grad_(False)
    opt, rng, t0 = adamw(model), np.random.default_rng(seed + 1), time.time()
    res = {"variante": variant, "medidas": []}
    for step in range(max_steps + 1):
        if step in CHECKPOINTS or step == max_steps:
            m = measure(model, ev, seed)
            res["medidas"].append({"paso": step, **m})
            print(json.dumps({"variante": variant, "paso": step, "perdida": m["perdida"],
                              "ruteo_d123": [c["ruteo"] for c in m["capas"]],
                              "estados_d123": [c["estados"] for c in m["capas"]],
                              "segundos": round(time.time() - t0)}), flush=True)  # fmt: skip
        if step == max_steps:
            break
        lg = rng.integers(h.N_LANG, size=BATCH)
        x = torch.tensor(h.ids(h.sample(tabs, lg, rng)[0], lg, False), device=DEVICE)
        model(input_ids=x, labels=x).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return res


def main(path: str, max_steps: int = 20000, seed: int = 0, variants: str = "base,router_azar") -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "pasos": max_steps, "tamano": SIZE, "corridas": []}
    if os.path.exists(path):  # retoma: no repite lo que ya está en la salida
        res = json.load(open(path))
    done = {c["variante"] for c in res["corridas"]}
    for v in variants.split(","):
        if v not in done:
            res["corridas"].append(run(v, max_steps, seed))
            json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 4:  # gramática más fácil: V y M (las funciones de hierarchy leen las constantes al llamarse)
        h.V, h.M = int(a[4]), int(a[5])
    main(a[0], *(int(x) for x in a[1:3]), *a[3:4])
