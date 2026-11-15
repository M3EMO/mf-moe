"""Análisis prerregistrado de B32 (ticket 12). Uso: python -m colab.b32_analisis B32.json B30.json B27C.json"""
import json, sys
import numpy as np
from colab.b20_analisis import ci, holds

R, B, C = (json.load(open(p)) for p in sys.argv[1:4])
base = np.array(R["base"]["perdida_ida"]); n = len(base)
rank = np.argsort(np.argsort(base))  # 0 = el más seguro; los tercios salen de la pérdida real
third = np.where(rank < n / 3, "seguro", np.where(rank >= 2 * n / 3, "dudoso", "medio"))
nt = len(R["config"]["plantillas"])
wins = R["base"]["gana_por_frase"]
H = [h for h, g in zip(("H1", "H2"), wins[nt:]) if g >= 0.6]
out = {"base_gana": {"T0": round(wins[0], 3), "H1": round(wins[nt], 3), "H2": round(wins[nt + 1], 3)}, "H_usadas": H}
out["I0_error_base_contra_b30"] = round(float(np.abs(base - np.array(B["base"]["perdida_ida"])).mean()), 5)
out["I3_error_sonda_contra_b27c"] = round(float(np.abs(np.array(R["sonda"]["prediccion"]) - np.array(C["sonda"]["prediccion"])).mean()), 5)
out["sonda_spearman"] = R["sonda"]["spearman_con_perdida_base"]
out["I0"], out["I3"] = ("pasa" if out[k] < 0.01 else "falla" for k in ("I0_error_base_contra_b30", "I3_error_sonda_contra_b27c"))
if not H:
    out["resultado"] = "sin_medida"; print(json.dumps(out, indent=1)); sys.exit()

def margin(r):
    return np.mean([r["margen"][f] for f in H], 0)

def per_set(X, arm, d, t):
    return np.array([float((margin(r := X["corridas"][f"s{s}|{arm}|d{d}"])[third[np.array(r["sobrescritos"])] == t] > 0).mean()) for s in range(8)])

def pooled(arm, d, t):
    m = np.concatenate([margin(R["corridas"][f"s{s}|{arm}|d{d}"]) for s in range(8)])
    idx = np.concatenate([R["corridas"][f"s{s}|{arm}|d{d}"]["sobrescritos"] for s in range(8)])
    return round(float((m[third[idx] == t] > 0).mean()), 3)

tab = out["resiste_H"] = {f"{a}|d{d}": {t: pooled(a, d, t) for t in ("seguro", "medio", "dudoso")} for a in ("una|umbral", "fuentes|umbral") for d in (10, 40)}
out["C0"] = "pasa" if 1 - tab["fuentes|umbral|d40"]["dudoso"] >= 0.8 else "falla"
v1, v2, v3 = tab["una|umbral|d10"]["seguro"] >= 0.5, 1 - tab["fuentes|umbral|d40"]["seguro"] >= 0.8, 1 - tab["una|umbral|d10"]["dudoso"] >= 0.8
out["R1"] = {"V1": v1, "V2": v2, "V3": v3, "video": bool(v1 and v2 and v3)}
peso, orac = per_set(B, "una|peso", 10, "seguro"), per_set(B, "una|umbral", 10, "seguro")
d = per_set(R, "una|umbral", 10, "seguro") - peso
out["S1_pareada_resiste_seguro"] = [round(x, 3) for x in ci(d)]
out["S1"] = "pasa" if holds(d, 0.5) else "falla"
out["S2_fraccion_del_oraculo"] = round(float(d.mean() / (orac - peso).mean()), 3)
out["dudoso_adopta_una_d10_por_mitad_de_la_sonda"] = {}
p = np.array(R["sonda"]["prediccion"]); sure_half = np.argsort(np.argsort(-p)) / (n - 1) >= 0.5  # tau > 1 posible solo en esta mitad
for half, mask in (("segura", sure_half), ("dudosa", ~sure_half)):
    m = np.concatenate([margin(R["corridas"][f"s{s}|una|umbral|d10"]) for s in range(8)])
    idx = np.concatenate([R["corridas"][f"s{s}|una|umbral|d10"]["sobrescritos"] for s in range(8)])
    sel = (third[idx] == "dudoso") & mask[idx]
    out["dudoso_adopta_una_d10_por_mitad_de_la_sonda"][half] = [int(sel.sum()), round(float((m[sel] <= 0).mean()), 3) if sel.any() else None]
spill = lambda X, a: np.array([np.delete(np.array(X["corridas"][f"s{s}|{a}|d10"]["subida_ida"]), X["corridas"][f"s{s}|{a}|d10"]["sobrescritos"]).mean() for s in range(8)])
out["subida_resto_d10"] = {"sonda_una_umbral": round(float(spill(R, "una|umbral").mean()), 3), "b30_una_peso": round(float(spill(B, "una|peso").mean()), 3),
                           "b30_una_umbral": round(float(spill(B, "una|umbral").mean()), 3)}  # fmt: skip
print(json.dumps(out, indent=1))
