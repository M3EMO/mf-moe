"""Análisis preregistrado de B27b (ticket 09). Uso: python -m colab.b27b_analisis B27B.json B27.json"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci

R, P = (json.load(open(p)) for p in sys.argv[1:3])
base = np.array(R["base"]["perdida_ida"]); cert = -base; n = len(base)
terc = np.digitize(cert, np.quantile(cert, [1 / 3, 2 / 3]))  # 0 = dudoso, 2 = seguro
rule = lambda m, lo, hi: "presente" if hi < 0 and m <= -0.2 else "ausente" if lo > -0.1 else "indeterminado"

def stats(runs, arm, d):
    rho, adopt = [], {t: [] for t in range(3)}
    for s in range(8):
        r = runs[f"s{s}|{arm}|d{d}"]; o = np.array(r["sobrescritos"]); up = np.array(r["subida_ida"])[o]
        rho.append(spearmanr(cert[o], up)[0])
        margin = np.array(r["perdida_nuevo"]) - (base[o] + up)  # positivo: lo viejo resiste
        for t in range(3):
            adopt[t] += list(margin[terc[o] == t] < 0)
    m, lo, hi = ci(np.array(rho))
    return {"n2": [round(m, 3), round(lo, 3), round(hi, 3)], "regla": rule(m, lo, hi),
            "adopta_por_tercio": [round(float(np.mean(adopt[t])), 3) for t in range(3)]}

out = {"integridad_base_vs_b27": round(float(np.abs(base - np.array(P["base"]["perdida_ida"])).mean()), 4),
       "senal_spearman_con_certeza": R["senal"]["spearman_con_certeza"]}
for d in (10, 40, 160):
    out[f"d{d}"] = {"senal": stats(R["corridas"], "senal", d), "oraculo_fuerte": stats(R["corridas"], "oraculo_fuerte", d),
                    "oraculo_b27": stats(P["corridas"], "oraculo", d), "azar_b27": stats(P["corridas"], "azar", d)}
s10, o10 = out["d10"]["senal"], out["d10"]["oraculo_b27"]
out["S1"] = "pasa" if s10["regla"] == "presente" else "falla"
out["S1_fraccion_del_oraculo"] = round(s10["n2"][0] / o10["n2"][0], 3)
f10, f160 = out["d10"]["oraculo_fuerte"]["adopta_por_tercio"], out["d160"]["oraculo_fuerte"]["adopta_por_tercio"]
out["R1"] = "pasa" if 1 - f10[2] >= 0.5 and f160[2] >= 0.9 else "falla"
out["R1b"] = "pasa" if f10[0] >= 0.9 else "falla"
out["integridad_ok"] = out["integridad_base_vs_b27"] < 0.01
print(json.dumps(out, indent=1))
