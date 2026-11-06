"""Reglas preregistradas de B21 (mapa mf-moe-generalidad, ticket 01). Reducción > 0 significa que el brazo daña menos.

Cada brazo se compara con el control de B20 del mismo conjunto. La unidad de réplica es el tema.
Uso: .venv/bin/python -m colab.b21_analisis B21.json B20.json
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

from colab.b20_analisis import ci

MIN_EFFECT, LEARNED_SHARE, MIN_TOPICS = 0.05, 0.9, 3
RELATIONS = ("total", "tema", "super", "otro", "fuera")


def seen(x: np.ndarray) -> np.ndarray:
    """Los temas que corrieron. Una resta con un tema faltante da NaN: solo quedan los temas de ambos brazos."""
    return x[~np.isnan(x)]


def pvalue(x: np.ndarray) -> float:
    p = float(ttest_1samp(x, 0).pvalue) if len(x) >= MIN_TOPICS else 1.0
    return 1.0 if np.isnan(p) else p  # diferencias idénticas: el t no está definido


def holm(ps: dict) -> dict:
    """Corrección de Holm: el i-ésimo p más chico por (m − i), acumulando el máximo."""
    out, top = {}, 0.0
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        top = max(top, min(1.0, p * (len(ps) - i)))
        out[k] = top
    return out


def table(b21: dict, b20: dict) -> dict:
    """{brazo: {relación: reducción por tema}}, con NaN en los temas sin corrida, para que cada posición sea siempre el mismo tema."""
    R21, R20, tops = b21["areas"], b20["areas"], b21["temas"]
    area_of = {x: k for k, v in b21["evaluacion"].items() for x in v}
    any_run = next(iter(R21.values()))
    topic = np.array([area_of[x] for x in any_run["evaluados"]])
    sup = np.array([tops[x] for x in topic])
    out: dict = {}
    for key, r in R21.items():
        area, s, arm = key.split("|")
        if arm == "control":
            continue
        d = np.array(R20[f"{area}|{s}"]["subida"]) - np.array(r["subida"])
        rel = {"total": np.ones(len(d), bool), "tema": topic == area, "super": (sup == tops[area]) & (topic != area), "otro": sup != tops[area]}
        rel["fuera"] = ~rel["tema"]
        cell = out.setdefault(arm, {}).setdefault(area, [])
        cell.append({k: float(d[m].mean()) for k, m in rel.items()})
    return {arm: {k: np.array([np.mean([c[k] for c in by[t]]) if t in by else np.nan for t in tops]) for k in RELATIONS}
            for arm, by in out.items()}  # fmt: skip


def learned(b21: dict) -> dict:
    """Parte de las corridas de cada brazo que llegó a aprender lo nuevo. Un brazo que no aprende no gana por dañar menos."""
    out: dict = {}
    for key, r in b21["areas"].items():
        out.setdefault(key.split("|")[2], []).append(r["nuevos_final"] < 0.5)
    return {k: float(np.mean(v)) for k, v in out.items()}


def holds(x: np.ndarray, minimum: float = MIN_EFFECT) -> bool:
    if len(x) < MIN_TOPICS:
        return False
    m, lo, _ = ci(x)
    return bool(lo > 0 and m >= minimum)


def verdict(T: dict, ok: dict) -> dict:
    """Seis reglas con Holm. Un brazo que no aprende no gana por dañar menos."""
    fine = lambda arm: ok.get(arm, 0.0) >= LEARNED_SHARE  # noqa: E731
    missing = np.full(len(next(iter(T.values()))["total"]), np.nan)  # un brazo abandonado antes de correr no rompe las otras reglas
    get = lambda arm, k: T[arm][k] if arm in T else missing  # noqa: E731
    rules = {"H_esp": (get("esp8", "fuera"), ("esp8",)), "D_esp": (get("esp8", "total") - get("azar8", "total"), ("esp8", "azar8")),
             "D_pop": (get("cont8", "total") - get("azar8", "total"), ("cont8", "azar8")), "H_cons24": (get("cons24", "total"), ("cons24",)),
             "D_cons": (get("cons24", "total") - get("azar24", "total"), ("cons24", "azar24")),
             "D_tronco": (get("cont8_tronco", "fuera") - get("cont8", "fuera"), ("cont8_tronco", "cont8"))}  # fmt: skip
    rules = {k: (seen(v), arms) for k, (v, arms) in rules.items()}
    p = holm({k: pvalue(v) for k, (v, _) in rules.items()})
    out: dict = {"C_pos": holds(seen(get("cont8_tronco", "fuera"))) and fine("cont8_tronco"), "holm": p, "temas": {k: len(v) for k, (v, _) in rules.items()},
                 "IC": {k: ci(v) for k, (v, _) in rules.items() if len(v)}}  # fmt: skip
    for k, (v, arms) in rules.items():
        s = v if len(v) and float(v.mean()) >= 0 else -v  # las diferencias valen en los dos sentidos: la regla mira el tamaño
        out[k] = bool(holds(s) and p[k] < 0.05 and all(fine(a) for a in arms))
    return out


def report(b21: dict, b20: dict) -> None:
    R = b21["areas"]
    print("corridas", len(R), "corte", b21.get("corte"), "fuga máxima", max(r["fuga"] for r in R.values()))
    for key, r in R.items():
        area, s, arm = key.split("|")
        if arm == "control":
            a, b = np.array(b20["areas"][f"{area}|{s}"]["subida"]), np.array(r["subida"])
            print(f"repetición del control {area}: ρ {spearmanr(a, b)[0]:+.3f}, dif. media {a.mean() - b.mean():+.3f}, pasos {b20['areas'][f'{area}|{s}']['pasos']} y {r['pasos']}")
    T, ok = table(b21, b20), learned(b21)
    for arm, by in T.items():
        rows = [r for k, r in R.items() if k.split("|")[2] == arm]
        print(f"{arm:13s} aprende {ok[arm]:.2f} pasos {np.mean([r['pasos'] for r in rows]):5.1f} nuevos {np.mean([r['nuevos_final'] for r in rows]):.3f} s {np.median([r['segundos'] for r in rows]):4.0f} | "
              + " ".join(f"{k} %+.3f [%+.3f; %+.3f]" % ci(seen(by[k])) for k in ("total", "tema", "super", "otro")))  # fmt: skip
    print(verdict(T, ok))
    if {"prec24", "cons24"} <= set(T):
        print("secundario prec24 − cons24: %+.3f [%+.3f; %+.3f]" % ci(seen(T["prec24"]["total"] - T["cons24"]["total"])))
    for arm in T:
        print(arm, "uso de lo nuevo en lo entrenable %.3f" % np.mean([r["uso_entrenable"] for k, r in R.items() if k.split("|")[2] == arm]))


if __name__ == "__main__":
    report(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
