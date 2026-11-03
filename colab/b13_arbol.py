"""B13 (mf-moe-5tq.27): ¿el árbol de ruteo recupera el árbol de los datos? Plan y reglas: ticket 15 del mapa.

Datos: 8 lenguas MRHM en un árbol binario (`mf_moe.hierarchy`), la gramática sale de la semilla. Modelo de B12c
(16 expertos, top-2, d = 64, 4 capas). Variantes por semilla, mismos datos y pasos:
- base: vocabulario terminal compartido (el léxico no delata la rama);
- router_azar: router congelado en su inicialización (parte mecánica, como en B12c);
- sin_balanceo: sesgo de balanceo apagado (Lema 2 de «Myth»: el balanceo borra la dirección compartida);
- vocab_rama: 16 símbolos propios por lengua (control positivo: si acá no aparece el árbol, la medida está rota).
Medidas en los pasos de CHECKPOINTS, por capa, sobre todas las posiciones de la oración:
- ruteo: distancia JSD entre el uso de expertos de cada lengua → Mantel y tripletas contra el árbol (nulo: hojas
  permutadas); IM experto × clado a profundidad 1, 2 y 3, e IM experto × símbolo latente de cada nivel;
- kmeans: lo mismo con k-means de 16 sobre los estados que entran al router, top-2 centroides (control «Myth»);
- estados: distancia coseno entre los estados medios centrados de cada lengua (lo que mide Doshi en un denso).
Uso: .venv/bin/python -m colab.b13_arbol SALIDA.json [pasos] [semilla]
"""

import json
import os
import sys
import time

import numpy as np
import torch

from colab.b9_bits import BATCH, DEVICE, SIZE, _mi
from colab.b12_raices import E0
from mf_moe import hierarchy as h
from mf_moe.model import build, routers
from mf_moe.train import adamw

VARIANTS = ("base", "router_azar", "sin_balanceo", "vocab_rama")
CHECKPOINTS = (0, 1000, 3000, 10000, 20000)
EVAL_PER_LANG = 250


@torch.no_grad()
def probe(model, ids: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray]:
    """Pérdida, top-k [capas, tokens, k] y estados que entran al router [capas, tokens, d], sin el BOS."""
    model.eval()
    rs = routers(model)
    for r in rs:
        r.record = True
    loss = model(input_ids=ids, labels=ids).loss.item()
    keep = (torch.arange(ids.numel(), device=ids.device) % ids.shape[1]) > 0
    idx = np.stack([r.last[2][keep].cpu().numpy() for r in rs])
    H = np.stack([r.last[0][keep].cpu().numpy() for r in rs])
    for r in rs:
        r.record = False
    model.train()
    return loss, idx, H


def _block(ix: np.ndarray, lang: np.ndarray, latent: list, seed: int) -> dict:
    return {**h.tree_recovery(h.jsd_matrix(h.usage(ix, lang, h.N_LANG, E0)), seed=seed),
            "im_clado": [round(_mi(h.clade(lang, d), ix, 2**d), 4) for d in range(1, h.DEPTH + 1)],
            "im_latente": [round(_mi(np.repeat(z, h.SEQ // z.shape[1], 1).reshape(-1), ix, h.V), 4) for z in latent]}  # fmt: skip


def measure(model, ids: torch.Tensor, lang: np.ndarray, latent: list, seed: int) -> dict:
    loss, idx, H = probe(model, ids)
    tok_lang = np.repeat(lang, h.SEQ)
    out = {"perdida": round(loss, 4), "capas": []}
    for L in range(len(idx)):
        Hm = np.stack([H[L][tok_lang == g].mean(0) for g in range(h.N_LANG)])
        Hm = (Hm - Hm.mean(0)) / np.linalg.norm(Hm - Hm.mean(0), axis=1, keepdims=True)
        out["capas"].append({
            "ruteo": _block(idx[L], tok_lang, latent, seed),
            "kmeans": _block(h.kmeans_topk(H[L], E0, idx.shape[2], seed), tok_lang, latent, seed),
            "estados": h.tree_recovery(1 - Hm @ Hm.T, seed=seed),
            "uso_top2": round(float(np.sort(np.bincount(idx[L].reshape(-1), minlength=E0))[::-1][:2].sum() / idx[L].size), 4),
        })  # fmt: skip
    return out


def run(variant: str, max_steps: int, seed: int) -> dict:
    own = variant == "vocab_rama"
    tabs = h.rules(seed)
    ev_rng = np.random.default_rng(10**6 + seed)
    ev_lang = np.repeat(np.arange(h.N_LANG), EVAL_PER_LANG)
    ev_syms, ev_latent = h.sample(tabs, ev_lang, ev_rng)
    ev_ids = torch.tensor(h.ids(ev_syms, ev_lang, own), device=DEVICE)
    model = build(E0, seed=seed, **SIZE).to(DEVICE)
    for r in routers(model):
        if variant == "sin_balanceo":
            r.u = 0.0
        if variant == "router_azar":
            r.weight.requires_grad_(False)
    opt, rng, t0 = adamw(model), np.random.default_rng(seed + 1), time.time()
    res = {"variante": variant, "medidas": [], "perdida_entrenamiento": []}
    for step in range(max_steps + 1):
        if step in CHECKPOINTS or step == max_steps:
            m = measure(model, ev_ids, ev_lang, ev_latent, seed)
            res["medidas"].append({"paso": step, **m})
            print(json.dumps({"variante": variant, "paso": step, "perdida": m["perdida"],
                              "mantel_ruteo": [c["ruteo"]["mantel"] for c in m["capas"]],
                              "mantel_kmeans": [c["kmeans"]["mantel"] for c in m["capas"]],
                              "segundos": round(time.time() - t0)}), flush=True)  # fmt: skip
        if step == max_steps:
            break
        lang = rng.integers(h.N_LANG, size=BATCH)
        ids = torch.tensor(h.ids(h.sample(tabs, lang, rng)[0], lang, own), device=DEVICE)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            res["perdida_entrenamiento"].append([step, round(loss.item(), 4)])
    return res


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "pasos": max_steps, "tamano": SIZE, "corridas": []}
    if os.path.exists(path):  # retoma: no repite lo que ya está en la salida
        res = json.load(open(path))
    done = {c["variante"] for c in res["corridas"]}
    for v in VARIANTS:
        if v in done:
            continue
        res["corridas"].append(run(v, max_steps, seed))
        json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
