"""Análisis prerregistrado de B33 (ticket 13). Uso: python -m colab.b33_analisis B33.json B27.json B27C.json"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci, holds

R, P, C = (json.load(open(p)) for p in sys.argv[1:4])
base = np.array(R["base"]["perdida_ida"]); cert = -base

def n2(X, arm, d):
    runs = X["corridas"]
    return np.array([spearmanr(cert[o := np.array(runs[f"s{s}|{arm}|d{d}"]["sobrescritos"])], np.array(runs[f"s{s}|{arm}|d{d}"]["subida_ida"])[o])[0] for s in range(8)])

r3 = lambda t: [round(x, 3) for x in t]
out = {"I0_error_base_contra_b27": round(float(np.abs(base - np.array(P["base"]["perdida_ida"])).mean()), 5),
       "seleccion": R["seleccion"], "sonda2_spearman": R["sonda2"]["spearman_con_perdida_base"], "ruido_spearman": R["ruido"]["spearman_con_perdida_base"]}
out["I0"] = "pasa" if out["I0_error_base_contra_b27"] < 0.01 else "falla"
out["R0"] = "pasa" if abs(out["ruido_spearman"] - 0.616) < 0.01 else "falla"
out["S0"] = "pasa" if out["sonda2_spearman"] >= 0.666 else "falla"
for d in (10, 40, 160):
    arms = {"sonda2": n2(R, "sonda2", d), "oraculo_ruido": n2(R, "oraculo_ruido", d), "sonda_b27c": n2(C, "sonda", d),
            "oraculo_b27": n2(P, "oraculo", d), "azar_b27": n2(P, "azar", d)}
    out[f"d{d}"] = {"n2": {k: r3(ci(v)) for k, v in arms.items()},
                    "pareada_contra_azar": {k: r3(ci(v - arms["azar_b27"])) for k, v in arms.items() if k != "azar_b27"},
                    "fraccion_del_oraculo": {k: round(float((v - arms["azar_b27"]).mean() / (arms["oraculo_b27"] - arms["azar_b27"]).mean()), 3)
                                             for k, v in arms.items() if k not in ("azar_b27", "oraculo_b27")}}  # fmt: skip
    if d == 10:
        out["S1"] = "pasa" if holds(arms["azar_b27"] - arms["sonda2"], 0.15) else "falla"
    if d == 160:
        out["D1"] = "pasa" if holds(arms["azar_b27"] - arms["oraculo_ruido"], 0.10) else "falla"
        out["D2"] = "pasa" if holds(arms["azar_b27"] - arms["sonda2"], 0.10) else "falla"
print(json.dumps(out, indent=1))
