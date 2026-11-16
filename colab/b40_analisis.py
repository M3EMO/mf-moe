"""Análisis prerregistrado de B40 (ticket 19). Uso: python -m colab.b40_analisis B40.json B38.json B25_REALES.json"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

SETS, BOOT = range(1, 8), 10_000
BRIER = {"R1": 0.60, "R2": 0.50}


def boot(x: np.ndarray, y: np.ndarray, n: int = BOOT) -> list[float]:
    """Spearman y su IC90 por remuestreo de hechos."""
    rng = np.random.default_rng(0)
    b = [spearmanr(x[i], y[i])[0] for i in rng.integers(0, len(x), (n, len(x)))]
    return [round(float(spearmanr(x, y)[0]), 3), *(round(float(v), 3) for v in np.percentile(b, [5, 95]))]


def main(b40: str, b38: str, b25: str) -> dict:
    R, E, P = (json.load(open(p)) for p in (b40, b38, b25))
    cert = -np.array(P["base"]["perdida_ida"])
    h = {k: np.array(v) for k, v in R["hechos"].items()}
    idx, prot = [], []
    for s in SETS:
        over = P["corridas"][f"s{s}|ida|d10"]["sobrescritos"]
        idx += over
        prot += list(np.array(P["corridas"][f"s{s}|ida|d10"]["subida_ida"])[over] - np.array(E["corridas"][f"s{s}|ewc|d10"]["subida_ida"])[over])
    idx, prot = np.array(idx), np.array(prot)
    out = {"I0": R["fisher"]["entradas"] == E["fisher"]["entradas"], "I0_base_error": round(float(np.abs(np.array(R["base"]["perdida_ida"]) + cert).max()), 4),
           "R1_certeza_r_nuevo": boot(cert, h["r_nuevo"]), "R2_r_nuevo_proteccion": boot(h["r_nuevo"][idx], prot),
           "certeza_r_viejo": boot(cert, h["r_viejo"]), "certeza_g2_nuevo": boot(cert, h["g2_nuevo"]),
           "certeza_proteccion": boot(cert[idx], prot), "n_R2": len(idx)}  # fmt: skip
    out["R1"] = bool(out["R1_certeza_r_nuevo"][1] > 0 and out["R1_certeza_r_nuevo"][0] >= 0.20)
    out["R2"] = bool(out["R2_r_nuevo_proteccion"][1] > 0 and out["R2_r_nuevo_proteccion"][0] >= 0.15)
    out["brier"] = {k: round((p - float(out[k])) ** 2, 3) for k, p in BRIER.items()}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
