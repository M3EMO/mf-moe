"""Análisis prerregistrado de B34 (ticket 13). Uso: python -m colab.b34_analisis B34.json B30.json"""
import json, sys
import numpy as np

R, B = (json.load(open(p)) for p in sys.argv[1:3])
base = np.array(R["base"]["perdida_ida"]); n = len(base)
rank = np.argsort(np.argsort(base))  # 0 = el más seguro
third = np.where(rank < n / 3, "seguro", np.where(rank >= 2 * n / 3, "dudoso", "medio"))
nt = len(R["config"]["plantillas"])
wins = R["base"]["gana_por_frase"]
H = [h for h, g in zip(("H1", "H2"), wins[nt:]) if g >= 0.6]
out = {"I0_error_base_contra_b30": round(float(np.abs(base - np.array(B["base"]["perdida_ida"])).mean()), 5), "H_usadas": H,
       "similitud": {k: v for k, v in R["similitud"].items() if k != "por_hecho"}}
if not H:
    out["resultado"] = "sin_medida"; print(json.dumps(out, indent=1)); sys.exit()

def pooled(X, arm, d, t, key="margen"):
    rs = [X["corridas"][f"s{s}|{arm}|d{d}"] for s in range(8)]
    idx = np.concatenate([r["sobrescritos"] for r in rs])
    if key == "abiertos":
        return round(float(np.concatenate([r["abiertos"] for r in rs])[third[idx] == t].mean()), 3)
    m = np.concatenate([np.mean([r["margen"][f] for f in H], 0) for r in rs])
    return round(float((m[third[idx] == t] > 0).mean()), 3)

arms = [k.split("|d")[0].split("|", 1)[1] for k in R["corridas"] if k.startswith("s0|") and k.endswith("|d10")]
tab = out["resiste_H"] = {f"{a}|d{d}": {t: pooled(R, a, d, t) for t in ("seguro", "medio", "dudoso")} for a in arms for d in (10, 40)}
for a in ("una|umbral", "fuentes|umbral", "una|peso"):
    tab[f"b30:{a}|d10"] = {t: pooled(B, a, 10, t) for t in ("seguro", "medio", "dudoso")}
tab["b30:fuentes|umbral|d40"] = {t: pooled(B, "fuentes|umbral", 40, t) for t in ("seguro", "medio", "dudoso")}
c = R["similitud"]["contadas_de_10"]
out["C0"] = "pasa" if 1 - tab["contenido|peso|d40"]["dudoso"] >= 0.8 else "falla"
out["C1"] = "pasa" if c["contenido"] - c["fuentes"] >= 1.0 else "falla"
out["K1"] = "pasa" if 1 - tab["contenido|umbral|d40"]["seguro"] >= 0.8 else "falla"
out["K2"] = "pasa" if tab["contenido|peso|d10"]["seguro"] >= 0.5 else "falla"
out["K3"] = "pasa" if 1 - tab["contenido|umbral_sim|d40"]["seguro"] >= 0.8 and 1 - tab["contenido|umbral_sim|d10"]["dudoso"] >= 0.8 else "falla"
out["K4"] = {"seguro_adopta_d40": round(1 - tab["fuentes|umbral_sim|d40"]["seguro"], 3), "seguros_abiertos_d40": pooled(R, "fuentes|umbral_sim", 40, "seguro", "abiertos")}
out["seguros_abiertos_d40"] = {a: pooled(R, a, 40, "seguro", "abiertos") for a in arms}
spill = lambda X, a, d: round(float(np.mean([np.delete(np.array(X["corridas"][f"s{s}|{a}|d{d}"]["subida_ida"]), X["corridas"][f"s{s}|{a}|d{d}"]["sobrescritos"]).mean() for s in range(8)])), 3)
out["subida_resto_d10"] = {**{a: spill(R, a, 10) for a in arms}, "b30:una|umbral": spill(B, "una|umbral", 10), "b30:fuentes|umbral": spill(B, "fuentes|umbral", 10)}
print(json.dumps(out, indent=1))
