"""Análisis prerregistrado de B29 (ticket 11). Uso: python -m colab.b29_analisis B29.json"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci

R = json.load(open(sys.argv[1]))
cert = -np.array(R["base"]["perdida_ida"])
rule = lambda m, lo, hi: "presente" if hi < 0 and m <= -0.2 else "ausente" if lo > -0.1 else "indeterminado"
def n2(arm):
    rho = [spearmanr(cert[o := np.array(R["corridas"][f"s{s}|{arm}|d10"]["sobrescritos"])], np.array(R["corridas"][f"s{s}|{arm}|d10"]["subida_ida"])[o])[0] for s in range(8)]
    m, lo, hi = ci(np.array(rho))
    return [round(m, 3), round(lo, 3), round(hi, 3), rule(m, lo, hi)]
out = {"gana_ida": R["base"]["gana"], "sonda_spearman": R["sonda"]["spearman_con_perdida_base"], **{a: n2(a) for a in ("oraculo", "azar", "sonda")}}
out["integridad"] = "pasa" if out["gana_ida"] >= 0.8 and out["sonda_spearman"] >= 0.5 else "falla"
out["G1"] = "pasa" if out["oraculo"][3] == "presente" and out["azar"][3] != "presente" else "falla"
out["P1"] = "pasa" if out["sonda"][3] == "presente" else "falla"
out["P2_fraccion"] = round(out["sonda"][0] / out["oraculo"][0], 3)
out["P2"] = "pasa" if out["P2_fraccion"] >= 0.75 else "falla"
print(json.dumps(out))
