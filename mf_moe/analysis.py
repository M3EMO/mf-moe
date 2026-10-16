"""Análisis del factorial (ticket 09): lee los result.json y decide la hipótesis.

Prueba principal: diferencia de BWT (fractal principal menos fijo con repetición) en pares con la misma
semilla y escenario; bootstrap pareado con IC 95 %. El fractal gana si el IC queda entero sobre 0.
Secundarias por condición: exactitud media final, olvido, BWT, pasos totales, expertos finales por capa y
transferencia hacia adelante (pasos de cada tarea en fila / pasos de la tarea sola; < 1 = transfiere).

Uso: python -m mf_moe.analysis DIR_RESULTADOS
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

from mf_moe.reference import STEPS_ALONE
from mf_moe.sequential import PRINCIPAL

CONTROL = "fijo+repetición"


def load(root: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/result.json"))]


def transfer(res: dict) -> float:
    """Media geométrica de pasos en fila / pasos sola, sin la primera tarea (no tiene nada antes)."""
    ratios = [s / STEPS_ALONE[t] for t, s in zip(res["orden"][1:], res["pasos"][1:])]
    return statistics.geometric_mean(ratios) if ratios else float("nan")


def pairs(results: list[dict], a: str, b: str, metric: str = "bwt") -> list[tuple[str, float, float]]:
    """(clave, métrica de a, métrica de b) para cada semilla y escenario con las dos condiciones."""
    by = {(r["condicion"]["name"], r["semilla"], r["con_nombre"]): r for r in results}
    out = []
    for (name, seed, named), r in sorted(by.items()):
        if name == a and (b, seed, named) in by:
            out.append((f"s{seed}/{'con' if named else 'sin'}", r[metric], by[(b, seed, named)][metric]))
    return out


def bootstrap_ci(diffs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Media de las diferencias y su IC 95 % por bootstrap pareado (remuestreo de los pares)."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return statistics.fmean(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summary(results: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["condicion"]["name"], []).append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        out[name] = {"corridas": len(rs)} | {
            k: round(statistics.fmean(r[k] for r in rs), 4)
            for k in ("bwt", "exactitud_media", "olvido", "pasos_totales")
        } | {"transferencia": round(statistics.fmean(transfer(r) for r in rs), 3),
             "expertos_finales": round(statistics.fmean(statistics.fmean(r["expertos"][-1]) for r in rs), 2)}  # fmt: skip
    return out


def main(root: str) -> None:
    results = load(root)
    print(json.dumps(summary(results), ensure_ascii=False, indent=1))
    pr = pairs(results, PRINCIPAL.name, CONTROL)
    if pr:
        diffs = [a - b for _, a, b in pr]
        mean, lo, hi = bootstrap_ci(diffs)
        verdict = "gana el fractal" if lo > 0 else "pierde el fractal" if hi < 0 else "sin diferencia clara"
        print(json.dumps({"pares": [(k, round(a, 4), round(b, 4)) for k, a, b in pr], "diferencia_bwt": round(mean, 4),
                          "ic95": [round(lo, 4), round(hi, 4)], "veredicto": verdict}, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1])
