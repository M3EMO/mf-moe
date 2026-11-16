"""Análisis prerregistrado de B39 (ticket 18). Uso: python -m colab.b39_analisis B39.json B38.json B25_REALES.json B27.json B27C.json"""

import json
import sys

import numpy as np

from colab.b20_analisis import ci, holds
from colab.b38_analisis import damage, learned, n2

SETS, MAX_NEW = range(1, 8), 0.5
BRIER = {"C0": 0.65, "S1": 0.55, "S2": 0.35, "D1": 0.70}
SOURCE = {"comun": (2, "ida"), "ewc": (1, "ewc"), "oraculo": (3, "oraculo"), "azar": (3, "azar"), "sonda": (4, "sonda"),
          "ewc+oraculo": (0, "ewc+oraculo"), "ewc+sonda": (0, "ewc+sonda"), "ewc+azar": (0, "ewc+azar")}  # fmt: skip


def main(*paths: str) -> dict:
    J = [json.load(open(p)) for p in paths]
    base = np.array(J[2]["base"]["perdida_ida"])
    cert = -base
    get = lambda a, d: [J[SOURCE[a][0]]["corridas"][f"s{s}|{SOURCE[a][1]}|d{d}"] for s in SETS]
    r3 = lambda t: [round(x, 3) for x in t]
    out = {"I0_error": round(float(np.abs(np.array(J[0]["base"]["perdida_ida"]) - base).max()), 4)}
    for d in (10, 40):
        N = {a: np.array([n2(r, cert) for r in get(a, d)]) for a in SOURCE}
        D = {a: np.array([damage(r) for r in get(a, d)]) for a in SOURCE}
        A = {a: np.array([learned(r) for r in get(a, d)]) for a in SOURCE}
        out[f"d{d}"] = {"N2": {a: r3(ci(v)) for a, v in N.items()}, "D": {a: round(float(v.mean()), 3) for a, v in D.items()},
                        "A": {a: round(float(v.mean()), 3) for a, v in A.items()},
                        "oraculo_sobre_ewc": r3(ci(N["ewc+oraculo"] - N["ewc"])), "sonda_sobre_ewc": r3(ci(N["ewc+sonda"] - N["ewc"])),
                        "combinada_sobre_sonda": r3(ci(N["ewc+sonda"] - N["sonda"])), "K_azar_sobre_ewc": r3(ci(N["ewc+azar"] - N["ewc"])),
                        "rel_ewc+sonda": r3(ci((D["comun"] - D["ewc+sonda"]) / D["comun"]))}  # fmt: skip
        below = lambda x, m: (lambda t: t[2] < 0 and t[0] <= m)(ci(x))
        if d == 10:
            out["C0"] = below(N["ewc+oraculo"] - N["ewc"], -0.15)
            out["S1"] = below(N["ewc+sonda"] - N["ewc"], -0.10)
            out["S2"] = below(N["ewc+sonda"] - N["sonda"], -0.05)
        else:
            out["L0"] = {a: bool(A[a].mean() <= MAX_NEW) for a in ("ewc+oraculo", "ewc+sonda", "ewc+azar")}
            out["D1"] = holds((D["comun"] - D["ewc+sonda"]) / D["comun"], 0.40)
    out["brier"] = {k: round((p - float(out[k])) ** 2, 3) for k, p in BRIER.items()}
    out["brier_media"] = round(float(np.mean(list(out["brier"].values()))), 3)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
