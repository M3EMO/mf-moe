"""B33: ¿una sonda mejor acerca la puerta al oráculo, y la caída de la sonda en la dosis 160 viene de la calidad de la señal?
Prerregistro en el ticket 13 (mf-moe-generalidad). Banco, conjuntos (tag «b25»), semillas, lotes y dosis de B27c; solo cambia la señal.
- sonda2: elige entre cuatro juegos de rasgos con validación cruzada sobre los candidatos que no están en el banco (nunca mira el banco).
  Los rasgos salen del modelo de partida: último token de la pregunta en las capas de B27c, en todas las capas pares, o
  además el promedio de los tokens de la pregunta. El juego «+extra» suma hechos de UniEdit al entrenamiento.
- oraculo_ruido: la pérdida real con ruido gaussiano, con el Spearman de la sonda de B27c (0,616) contra la pérdida.
  Si se apaga en la dosis 160 como la sonda, la caída viene de la calidad de la señal y no de errores con estructura.
Uso: python -m colab.b33_sonda2 SALIDA.json BANCO.json CANDIDATOS.json HECHOS_UNIEDIT.json
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
from colab import b27_puerta as b27
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

RHO_B27C, N_EXTRA, FOLDS = 0.616, 3000, 5


@torch.no_grad()
def states(model, tok, facts: list[dict], layers: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Último token y promedio de los tokens de la pregunta de la ida, por capa de `layers`, concatenados: no ven el valor."""
    model.eval()
    last_out, mean_out = [], []
    for i in range(0, len(facts), b17.EVAL_BATCH):
        rows = [tok(b24.fwd(x)[0])["input_ids"] for x in facts[i : i + b17.EVAL_BATCH]]
        L = max(map(len, rows))
        ids = torch.tensor([r + [0] * (L - len(r)) for r in rows], device=DEVICE)
        mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows], device=DEVICE)
        with amp(DEVICE):
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        last, m = torch.tensor([len(r) - 1 for r in rows], device=DEVICE), mask[..., None].float()
        last_out.append(torch.cat([hs[l][torch.arange(len(rows)), last].float() for l in layers], -1).cpu().numpy())
        mean_out.append(torch.cat([(hs[l].float() * m).sum(1) / m.sum(1) for l in layers], -1).cpu().numpy())
    model.train()
    return np.concatenate(last_out), np.concatenate(mean_out)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def select(feats: dict, y: np.ndarray, test: np.ndarray, train_mask: dict) -> dict:
    """Spearman fuera de pliegue sobre las filas `test` (los candidatos fuera del banco).
    `train_mask[nombre]`: filas que pueden entrenar en ese juego (sin los hechos extra, salvo en «+extra»)."""
    idx = np.flatnonzero(test)
    folds = np.array(random.Random("b33:pliegues").choices(range(FOLDS), k=len(idx)))
    score = {}
    for name, X in feats.items():
        pred = np.full(len(idx), np.nan)
        for f in range(FOLDS):
            out = np.zeros(len(y), bool)
            out[idx[folds == f]] = True
            keep = ~out & train_mask[name]
            pred[folds == f] = b27.ridge(X[keep], y[keep], X[out])
        score[name] = round(spearman(pred, y[test]), 4)
    return score


def noisy(signal: np.ndarray, target: float, seed: int = 33) -> tuple[np.ndarray, float]:
    """La señal más un ruido gaussiano fijo, con la escala que deja su Spearman contra la señal en `target` (bisección)."""
    z, lo, hi = np.random.default_rng(seed).standard_normal(len(signal)), 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if spearman(signal + mid * signal.std() * z, signal) > target else (lo, mid)
    out = signal + lo * signal.std() * z
    return out, spearman(out, signal)


def main(path: str, bank_path: str, cand_path: str, uniedit_path: str, model=None, tok=None) -> None:
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    n, key = len(facts), lambda x: (x["cabeza"], x["propiedad"])
    in_bank = {key(x) for x in facts}
    pool = cands + [x for x in facts if key(x) not in {key(c) for c in cands}]  # como b27.probe
    taken = {key(x) for x in pool}
    extra = [x for v in json.load(open(uniedit_path, encoding="utf-8"))["hechos"].values() for x in v if key(x) not in taken]
    extra = random.Random("b33:extra").sample(extra, min(N_EXTRA, len(extra)))
    rows = pool + extra
    y = b17.def_losses(model, tok, [b24.fwd(x) for x in rows])
    every = tuple(range(2, model.config.num_hidden_layers + 1, 2))
    last_all, mean_all = states(model, tok, rows, every)
    d = last_all.shape[1] // len(every)
    b27c_cols = np.concatenate([np.arange(every.index(l) * d, (every.index(l) + 1) * d) for l in b27.LAYERS if l in every])
    both = np.hstack([last_all, mean_all])
    feats = {"b27c": last_all[:, b27c_cols], "pares": last_all, "pares+media": both, "pares+media+extra": both}
    is_pool = np.arange(len(rows)) < len(pool)
    test = is_pool & np.array([key(x) not in in_bank for x in rows])
    train_mask = {k: (np.ones(len(rows), bool) if k.endswith("extra") else is_pool) for k in feats}
    score = select(feats, y, test, train_mask)
    best = max(score, key=score.get)
    X, use = feats[best], train_mask[best]
    # predicción del banco como b27.probe: cada conjunto de stage2 con una sonda ajustada sin sus hechos
    order = random.Random("b25:conjuntos").sample(range(n), n)
    group = {key(facts[i]): min(j // (n // b24.SETS), b24.SETS) for j, i in enumerate(order)}
    g = np.array([group.get(key(x), -1) for x in rows])
    where = {key(x): i for i, x in enumerate(rows[: len(pool)])}
    p_bank = np.zeros(n)
    for s in range(b24.SETS + 1):
        mine = [i for i, x in enumerate(facts) if group[key(x)] == s]
        if mine:
            fit = use & (g != s)
            p_bank[mine] = b27.ridge(X[fit], y[fit], X[[where[key(facts[i])] for i in mine]])
    rho = spearman(p_bank, base_i[:, 0])
    p_noise, rho_noise = noisy(base_i[:, 0], RHO_B27C)
    res["config"] = {"n_hechos": n, "n_sonda": len(pool), "n_extra": len(extra), "capas": list(every), "pliegues": FOLDS, "conjuntos": b24.SETS,
                     "dosis": list(b24.DOSES), "puerta": [b27.LOW, b27.SPAN], "lr": b17.LR, "batch": b17.BATCH, "tag": "b25"}  # fmt: skip
    res["seleccion"] = {"spearman_cv_candidatos": score, "elegido": best}
    res["sonda2"] = {"prediccion": p_bank.round(4).tolist(), "spearman_con_perdida_base": round(rho, 4)}
    res["ruido"] = {"senal": p_noise.round(4).tolist(), "spearman_con_perdida_base": round(rho_noise, 4)}
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"seleccion": score, "elegido": best, "sonda2_spearman": round(rho, 4), "ruido_spearman": round(rho_noise, 4)}), flush=True)
    del last_all, mean_all, both, feats, X
    arms = {"sonda2": ("ida", b27.gate(p_bank), None), "oraculo_ruido": ("ida", b27.gate(p_noise), None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


if __name__ == "__main__":
    main(*sys.argv[1:5])
