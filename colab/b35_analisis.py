"""Análisis prerregistrado de B35 y B36 (ticket 14). Uso: python -m colab.b35_analisis SALIDA.json (el archivo que junta las tres etapas)"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from colab.b20_analisis import ci, holds

J = json.load(open(sys.argv[1]))
r3 = lambda t: [round(x, 3) for x in t]
out = {"B0_banco": len(J["banco"])}
out["B0"] = "pasa" if out["B0_banco"] >= 160 else "falla"
if "puerta" in J:
    P = J["puerta"]; cert = -np.array(P["base"]["perdida_ida"])
    n2 = {a: np.array([spearmanr(cert[o := np.array(P["corridas"][f"s{s}|{a}|d10"]["sobrescritos"])], np.array(P["corridas"][f"s{s}|{a}|d10"]["subida_ida"])[o])[0]
                       for s in range(8)]) for a in ("oraculo", "azar", "sonda")}  # fmt: skip
    out["puerta"] = {"gana_ida": round(P["base"]["gana"], 3), "sonda_spearman": P["sonda"]["spearman_con_perdida_base"],
                     "n2": {a: r3(ci(v)) for a, v in n2.items()}, "pareada_contra_azar": {a: r3(ci(n2[a] - n2["azar"])) for a in ("oraculo", "sonda")}}
    out["G1"] = "pasa" if holds(n2["azar"] - n2["oraculo"], 0.15) else "falla"
    out["G2"] = "pasa" if holds(n2["azar"] - n2["sonda"], 0.15) else "falla"
if "umbral" in J:
    U = J["umbral"]; base = np.array(U["base"]["perdida_ida"]); n = len(base)
    rank = np.argsort(np.argsort(base)); third = np.where(rank < n / 3, "seguro", np.where(rank >= 2 * n / 3, "dudoso", "medio"))
    nt = len(U["config"]["plantillas"]); wins = U["base"]["gana_por_frase"]
    H = [h for h, g in zip(("H1", "H2"), wins[nt:]) if g >= 0.6]
    out["H0"] = {"T0": round(wins[0], 3), "H1": round(wins[nt], 3), "H2": round(wins[nt + 1], 3), "usadas": H}
    frames = H or ["T0"]  # sin frases H, la medida cae a T0 y queda marcada
    margin = lambda r: np.mean([r["margen"][f] for f in frames], 0)
    def per_set(a, d, t):
        return np.array([float((margin(r := U["corridas"][f"s{s}|{a}|d{d}"])[third[np.array(r["sobrescritos"])] == t] > 0).mean()) for s in range(8)])
    pooled = lambda a, d, t: round(float(np.mean(np.concatenate([(margin(r := U["corridas"][f"s{s}|{a}|d{d}"]) > 0)[third[np.array(r["sobrescritos"])] == t] for s in range(8)]))), 3)
    tab = out["resiste"] = {f"{a}|d{d}": {t: pooled(a, d, t) for t in ("seguro", "medio", "dudoso")} for a in ("una|peso", "una|umbral", "fuentes|umbral") for d in (10, 40)}
    out["C0"] = "pasa" if 1 - tab["una|peso|d10"]["dudoso"] >= 0.8 else "falla"
    v1, v2, v3 = tab["una|umbral|d10"]["seguro"] >= 0.5, 1 - tab["fuentes|umbral|d40"]["seguro"] >= 0.8, 1 - tab["una|umbral|d10"]["dudoso"] >= 0.8
    out["U1"] = {"V1": v1, "V2": v2, "V3": v3, "pasa": bool(v1 and v2 and v3), "frases": frames}
    d = per_set("una|umbral", 10, "seguro") - per_set("una|peso", 10, "seguro")
    out["P1_pareada"] = r3(ci(d)); out["P1"] = "pasa" if holds(d, 0.30) else "falla"
    spill = lambda a: round(float(np.mean([np.delete(np.array(U["corridas"][f"s{s}|{a}|d10"]["subida_ida"]), U["corridas"][f"s{s}|{a}|d10"]["sobrescritos"]).mean() for s in range(8)])), 3)
    out["subida_resto_d10"] = {a: spill(a) for a in ("una|peso", "una|umbral", "fuentes|umbral")}
print(json.dumps(out, indent=1))
