"""Reglas preregistradas de B22 (mapa mf-moe-generalidad, ticket 02).

Una profundidad es válida si aprende, conserva el conflicto, conserva el gradiente por tema y se parece a B20
al menos tanto como otro conjunto del mismo tema se parece entre sí (ρ medio 0,55 en B20).
Uso: .venv/bin/python -m colab.b22_analisis B22.json B20.json
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

from colab.b17_conflicto import FAR_BAND
from colab.b20_analisis import ci

MIN_RHO, MIN_GRADIENT, LEARNED_SHARE, C0_SHARE, MIN_TOPICS = 0.55, 0.05, 0.9, 0.8, 3


def conflict(r: dict) -> bool:
    far = float(np.mean([x for x, w in zip(r["subida"], r["raiz"]) if w < FAR_BAND]))
    return r["subida_entrenados_verdad"] > max(0.5, 2 * far)


def gradient(r: dict, area: str, area_of: dict) -> float:
    """Daño en el propio tema menos daño fuera de él."""
    y, own = np.array(r["subida"]), np.array([area_of[x] == area for x in r["evaluados"]])
    return float(y[own].mean() - y[~own].mean())


def verdict(b22: dict, b20: dict) -> dict:
    area_of = {x: k for k, v in b22["evaluacion"].items() for x in v}
    out: dict = {}
    for arm in sorted({k.split("|")[2] for k in b22["areas"]}):
        rows = {k.rsplit("|", 1)[0]: r for k, r in b22["areas"].items() if k.endswith(f"|{arm}")}
        learned = float(np.mean([r["nuevos_final"] < 0.5 for r in rows.values()]))
        c0 = float(np.mean([conflict(r) for r in rows.values()]))
        rho = float(np.mean([spearmanr(r["subida"], b20["areas"][k]["subida"])[0] for k, r in rows.items()]))
        g = np.array([gradient(r, k.split("|")[0], area_of) for k, r in rows.items()])
        m, lo, hi = ci(g) if len(g) >= MIN_TOPICS else (float(g.mean()), float("nan"), float("nan"))
        ok = bool(len(g) >= MIN_TOPICS and learned >= LEARNED_SHARE and c0 >= C0_SHARE and rho >= MIN_RHO and lo > 0 and m >= MIN_GRADIENT)
        out[arm] = {"temas": len(g), "aprende": learned, "c0": c0, "rho_b20": rho, "gradiente": (m, lo, hi), "valida": ok}
    return out


def report(b22: dict, b20: dict) -> None:
    print("corridas", len(b22["areas"]), "corte", b22.get("corte"), "gpu", b22.get("gpu"))
    v = verdict(b22, b20)
    for arm, x in v.items():
        rows = [r for k, r in b22["areas"].items() if k.endswith(f"|{arm}")]
        print(f"{arm:4s}", {k: (round(i, 3) if isinstance(i, float) else i) for k, i in x.items() if k != "gradiente"}, "gradiente %+.3f [%+.3f; %+.3f]" % x["gradiente"],
              "| subida %.3f pasos %.0f s %.0f libres %.0fM base %.1f pico %.1f GB" % (np.mean([np.mean(r["subida"]) for r in rows]), np.mean([r["pasos"] for r in rows]), np.median([r["segundos"] for r in rows]),
              rows[0]["parametros_libres"] / 1e6, rows[0]["gb_base"], max(r["gb_pico"] for r in rows)))  # fmt: skip


if __name__ == "__main__":
    report(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
