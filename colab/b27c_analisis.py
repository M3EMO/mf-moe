"""Análisis preregistrado de B27c (ticket 09). Uso: python -m colab.b27c_analisis B27C.json B27.json B27B.json"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci

R, P, Q = (json.load(open(p)) for p in sys.argv[1:4])
base = np.array(R["base"]["perdida_ida"]); cert = -base
rule = lambda m, lo, hi: "presente" if hi < 0 and m <= -0.2 else "ausente" if lo > -0.1 else "indeterminado"

def n2(runs, arm, d):
    rho = [spearmanr(cert[o := np.array(runs[f"s{s}|{arm}|d{d}"]["sobrescritos"])], np.array(runs[f"s{s}|{arm}|d{d}"]["subida_ida"])[o])[0] for s in range(8)]
    m, lo, hi = ci(np.array(rho))
    return [round(m, 3), round(lo, 3), round(hi, 3), rule(m, lo, hi)]

out = {"integridad_base_vs_b27": round(float(np.abs(base - np.array(P["base"]["perdida_ida"])).mean()), 4),
       "sonda_spearman_con_perdida_base": R["sonda"]["spearman_con_perdida_base"]}
for d in (10, 40, 160):
    out[f"d{d}"] = {"sonda": n2(R["corridas"], "sonda", d), "oraculo_b27": n2(P["corridas"], "oraculo", d),
                    "senal_b27b": n2(Q["corridas"], "senal", d), "azar_b27": n2(P["corridas"], "azar", d)}
out["P1"] = "pasa" if out["d10"]["sonda"][3] == "presente" else "falla"
out["P2_fraccion"] = round(out["d10"]["sonda"][0] / out["d10"]["oraculo_b27"][0], 3)
out["P2"] = "pasa" if out["P2_fraccion"] >= 0.75 else "falla"
out["integridad_ok"] = out["integridad_base_vs_b27"] < 0.01 and out["sonda_spearman_con_perdida_base"] >= 0.5
print(json.dumps(out, indent=1))
