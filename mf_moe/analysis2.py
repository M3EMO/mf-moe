"""Análisis de la fase 2 (ticket 12): lee los result.json de las corridas de Granite y compara pareado por semilla.

Métrica primaria: olvido medio en nats (pérdida final menos pérdida al terminar cada dominio, promedio de los
tres primeros). Las diferencias se reportan como `control − brazo`: positivo = el brazo olvida menos. Secundarias:
pérdida aprendida media (más baja es mejor) y pérdida general final. Preguntas extra: ¿aporta la duda?
(división+padre congelado contra azar+padre congelado), ¿importa qué se congela? (consolidación por uso contra
división+padre congelado), ¿hace falta dividir? (consolidación con y sin división) y ¿importa elegir
por uso? (consolidación por uso contra al azar, sin división). Comprueba además que todas las corridas usaron los mismos datos (sha256).
Uso: python -m mf_moe.analysis2 DIR_RESULTADOS
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scipy.stats import ttest_1samp

from mf_moe.analysis import bootstrap_ci

CONTROL = "control"
QUESTIONS = [(arm, CONTROL) for arm in ("division", "division+fusion_alineada", "division+padre_congelado",
                                        "azar+padre_congelado", "division+consolidacion_uso",
                                        "control+consolidacion_uso", "control+consolidacion_azar")] + [
    ("division+padre_congelado", "azar+padre_congelado"),  # ¿aporta la duda?
    ("division+consolidacion_uso", "division+padre_congelado"),  # ¿importa qué se congela?
    ("division+consolidacion_uso", "control+consolidacion_uso"),  # ¿hace falta dividir?
    ("control+consolidacion_uso", "control+consolidacion_azar"),  # ¿importa elegir por uso?
    ("arbol+consolidacion", "control+consolidacion_uso"),  # H7: ¿el árbol suma sobre la consolidación plana?
    ("arbol+consolidacion", "control_bajo_rango+consolidacion"),  # H7 con los mismos parámetros nuevos (V1)
    ("arbol+consolidacion", "arbol_azar+consolidacion"),  # H9: ¿la duda elige dónde abrir rama?
    ("arbol+consolidacion", "arbol_un_nivel+consolidacion"),  # H10: ¿suma la recursión?
]  # fmt: skip


DEFAULT_ORDER = ["fines", "turco", "indonesio", "suajili"]


def arm(r: dict) -> str:
    """Nombre del brazo; un orden de dominios distinto del original (V4) es otro brazo: `nombre@orden`."""
    doms = r.get("cfg", {}).get("domains", DEFAULT_ORDER)
    return r["condicion"]["name"] + ("" if list(doms) == DEFAULT_ORDER else "@" + "-".join(doms))


def holm(ps: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (V5): p ajustados, en el orden de entrada."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def metrics(r: dict) -> dict[str, float]:
    return {"olvido_medio": r["olvido_medio"], "aprendido_medio": statistics.fmean(r["aprendido"]),
            "general_final": r["general_inicio_fin"][1]}  # fmt: skip


def compare(results: list[dict], a: str, b: str) -> dict:
    by = {(arm(r), r["semilla"]): metrics(r) for r in results}
    seeds = sorted(s for (n, s) in by if n == a and (b, s) in by)
    out = {"brazo": a, "contra": b, "semillas": seeds}
    for k in ("olvido_medio", "aprendido_medio", "general_final"):
        diffs = [by[(b, s)][k] - by[(a, s)][k] for s in seeds]  # las tres bajan cuando mejoran: positivo = `a` mejor
        if not diffs:
            continue
        mean, lo, hi = bootstrap_ci(diffs)
        verdict = "pocos pares" if len(diffs) < 3 else "mejor" if lo > 0 else "peor" if hi < 0 else "sin diferencia clara"
        # t pareada (V5): con 3 pares, el bootstrap tiene pocas remuestras distintas y su IC es optimista
        p = float(ttest_1samp(diffs, 0.0).pvalue) if len(diffs) > 1 and statistics.pstdev(diffs) > 0 else float("nan")
        out[k] = {"diferencias": [round(d, 4) for d in diffs], "media": round(mean, 4),
                  "ic95": [round(lo, 4), round(hi, 4)], "p_t": round(p, 4), "veredicto": verdict}  # fmt: skip
    return out


def main(root: str) -> None:
    results = [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/result.json"))]
    hashes = {json.dumps({d: v["sha256"] for d, v in r["datos"].items()}, sort_keys=True) for r in results if "datos" in r}
    print(json.dumps({"corridas": len(results), "datos_identicos": len(hashes) <= 1}))
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(arm(r), []).append(metrics(r))
    for name, ms in sorted(groups.items()):
        print(json.dumps({"condicion": name, "corridas": len(ms)} | {k: round(statistics.fmean(m[k] for m in ms), 4)
                                                                      for k in ms[0]}, ensure_ascii=False))  # fmt: skip
    rows = [compare(results, a, b) for a, b in QUESTIONS if a in groups and b in groups]
    tested = [r for r in rows if "olvido_medio" in r and r["olvido_medio"]["p_t"] == r["olvido_medio"]["p_t"]]
    for r, p in zip(tested, holm([r["olvido_medio"]["p_t"] for r in tested])):
        r["olvido_medio"]["p_holm"] = round(p, 4)  # sobre la métrica primaria, entre todas las preguntas con datos
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
