"""Análisis prerregistrado de B28e (ticket 10). Uso: python -m colab.b28e_analisis B28E.json B27C.json"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci

E, C = (json.load(open(p)) for p in sys.argv[1:3])
base = np.array(E["base"]["perdida_ida"]); cert = -base
rule = lambda m, lo, hi: "presente" if hi < 0 and m <= -0.2 else "ausente" if lo > -0.1 else "indeterminado"
rho = lambda run: spearmanr(cert[o := np.array(run["sobrescritos"])], np.array(run["subida_ida"])[o])[0]
fmt = lambda a: [round(float(v), 3) for v in ci(np.array(a))]
out = {"integridad_base_vs_b27c": round(float(np.abs(base - np.array(C["base"]["perdida_ida"])).mean()), 4),
       "sonda_spearman": E["sonda"]["spearman_con_perdida_base"],
       "peso_paso0_igual_b27c": bool(np.allclose(E["pesos"]["sonda_linea_paso0"], C["pesos"]["sonda"], atol=1e-3))}
for d in (10, 160):
    lin = [rho(E["corridas"][f"s{s}|sonda_linea|d{d}"]) for s in range(8)]
    fij = [rho(C["corridas"][f"s{s}|sonda|d{d}"]) for s in range(8)]
    w_rho = [spearmanr(np.array(E["corridas"][f"s{s}|sonda_linea|d{d}"]["pesos"]), base[E["corridas"][f"s{s}|sonda_linea|d{d}"]["sobrescritos"]])[0] for s in range(8)]
    m = fmt(lin)
    out[f"d{d}"] = {"N2_linea": m + [rule(*m)], "N2_fija_b27c": fmt(fij), "dif_pareada": fmt(np.array(lin) - np.array(fij)),
                    "rho_peso_con_perdida_base": fmt(w_rho),
                    "nuevos_linea": round(float(np.mean([E["corridas"][f"s{s}|sonda_linea|d{d}"]["nuevos_final"] for s in range(8)])), 3),
                    "nuevos_fija": round(float(np.mean([C["corridas"][f"s{s}|sonda|d{d}"]["nuevos_final"] for s in range(8)])), 3)}
out["L2"] = "pasa" if out["d10"]["N2_linea"][3] == "presente" else "falla"
m, lo, hi = out["d160"]["N2_linea"][:3]
out["L1"] = "pasa" if m <= -0.10 and hi < 0 else "falla"
print(json.dumps(out, indent=1))
