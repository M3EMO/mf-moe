"""Análisis prerregistrado de B37 (ticket 15). Uso: python -m colab.b37_analisis B37.json B25.json"""
import json, sys
import numpy as np
from colab.b20_analisis import ci, holds

R, B = (json.load(open(p)) for p in sys.argv[1:3])
r3 = lambda t: [round(x, 3) for x in t]
base = np.array(R["base"]["perdida_ida"])
out = {"I0_error_base_contra_b25": round(float(np.abs(base - np.array(B["etapa1"]["perdida_ida"] if "etapa1" in B else B["base"]["perdida_ida"])).mean()), 5)}
out["I0"] = "pasa" if out["I0_error_base_contra_b25"] < 0.01 else "falla"
D = lambda a, d: np.array([np.delete(np.array(R["corridas"][f"s{s}|{a}|d{d}"]["subida_ida"]), R["corridas"][f"s{s}|{a}|d{d}"]["sobrescritos"]).mean() for s in range(8)])
new = lambda a, d: np.array([np.mean(R["corridas"][f"s{s}|{a}|d{d}"]["perdida_nuevo"]) for s in range(8)])
for d in (10, 40):
    Da, Dg, Do = D("azar", d), D("guiado", d), D("oraculo", d)
    rg, ro = (Da - Dg) / Da, (Da - Do) / Da
    out[f"d{d}"] = {"D": {"azar": r3(ci(Da)), "guiado": r3(ci(Dg)), "oraculo": r3(ci(Do))}, "reduccion_guiado": r3(ci(rg)), "reduccion_oraculo": r3(ci(ro)),
                    "fraccion_del_techo": round(float(rg.mean() / ro.mean()), 3),
                    "perdida_nuevos": {a: round(float(new(a, d).mean()), 4) for a in ("azar", "guiado", "oraculo")}}  # fmt: skip
    if d == 40:
        out["C0"] = "pasa" if holds(ro, 0.03) else "falla"
        out["L4"] = "pasa" if holds(rg, 0.03) else "falla"
        out["A_marcada"] = bool(abs(new("guiado", d).mean() / new("azar", d).mean() - 1) > 0.10)
print(json.dumps(out, indent=1))
