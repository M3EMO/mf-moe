"""Análisis prerregistrado de B38 (ticket 17). Uso: python -m colab.b38_analisis B38.json B25_REALES.json"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

from colab.b20_analisis import ci, holds

SETS, MAX_NEW = range(1, 8), 0.5
BRIER = {"L0_mosca": 0.75, "F1": 0.55, "F2": 0.50, "E1": 0.60, "ewc_N2": 0.15}


def rows(R: dict, B: dict, arm: str, d: int) -> list[dict]:
    return [B[f"s{s}|ida|d{d}"] if arm == "comun" else R[f"s{s}|{arm}|d{d}"] for s in SETS]


def damage(r: dict) -> float:
    return float(np.delete(np.array(r["subida_ida"]), r["sobrescritos"]).mean())


def learned(r: dict) -> float:
    return float(np.mean(r["perdida_nuevo"])) if "perdida_nuevo" in r else float(r["nuevos_final"])


def n2(r: dict, cert: np.ndarray) -> float:
    o = np.array(r["sobrescritos"])
    return float(spearmanr(cert[o], np.array(r["subida_ida"])[o])[0])


def main(b38: str, b25: str) -> dict:
    R, P = json.load(open(b38)), json.load(open(b25))
    runs, B = R["corridas"], P["corridas"]
    base = np.array(P["base"]["perdida_ida"])
    cert = -base
    out = {"I0_error": round(float(np.abs(np.array(R["base"]["perdida_ida"]) - base).max()), 4), "eleccion": R.get("eleccion")}
    arms = ("comun", "ewc", "mosca", "mosca_densa")
    for d in (10, 40):
        D = {a: np.array([damage(r) for r in rows(runs, B, a, d)]) for a in arms}
        A = {a: np.array([learned(r) for r in rows(runs, B, a, d)]) for a in arms}
        N = {a: np.array([n2(r, cert) for r in rows(runs, B, a, d)]) for a in arms}
        rel = lambda x, ref: (D[ref] - D[x]) / D[ref]
        out[f"d{d}"] = {"D": {a: round(float(v.mean()), 3) for a, v in D.items()}, "A": {a: round(float(v.mean()), 3) for a, v in A.items()},
                        "N2": {a: [round(x, 3) for x in ci(v)] for a, v in N.items()},
                        "N2_menos_comun": {a: [round(x, 3) for x in ci(N[a] - N["comun"])] for a in arms[1:]},
                        "rel_mosca": [round(x, 3) for x in ci(rel("mosca", "comun"))], "rel_ewc": [round(x, 3) for x in ci(rel("ewc", "comun"))],
                        "rel_mosca_vs_densa": [round(x, 3) for x in ci(rel("mosca", "mosca_densa"))]}  # fmt: skip
        if d == 40:
            L0 = {a: bool(A[a].mean() <= MAX_NEW) for a in arms}
            out["L0"] = L0
            out["F1"] = L0["mosca"] and holds(rel("mosca", "comun"), 0.30)
            out["F2"] = L0["mosca"] and L0["mosca_densa"] and holds(rel("mosca", "mosca_densa"), 0.10)
            out["E1"] = L0["ewc"] and holds(rel("ewc", "comun"), 0.10)
        else:
            m, lo, hi = ci(N["ewc"])
            out["ewc_N2"] = bool(hi < 0 and m <= -0.15)
    outcome = {"L0_mosca": out["L0"]["mosca"], "F1": out["F1"], "F2": out["F2"], "E1": out["E1"], "ewc_N2": out["ewc_N2"]}
    out["brier"] = {k: round((p - float(outcome[k])) ** 2, 3) for k, p in BRIER.items()}
    out["brier_media"] = round(float(np.mean(list(out["brier"].values()))), 3)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
