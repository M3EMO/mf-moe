"""Análisis prerregistrado de B31 (ticket 12). Uso: python -m colab.b31_analisis B31.json B29.json"""
import json, sys
import numpy as np
from colab.b20_analisis import ci, holds

R, Q = (json.load(open(p)) for p in sys.argv[1:3])
base = np.array(R["base"]["perdida_ida"]); n = len(base)
rank = np.argsort(np.argsort(base))  # 0 = el más seguro
third = np.where(rank < n / 3, "seguro", np.where(rank >= 2 * n / 3, "dudoso", "medio"))
nt = len(R["config"]["plantillas"])
wins = R["base"]["gana_por_frase"]
H = [h for h, g in zip(("H1", "H2"), wins[nt:]) if g >= 0.6]
out = {"base_gana": {"T0": round(wins[0], 3), "H1": round(wins[nt], 3), "H2": round(wins[nt + 1], 3)}, "H_usadas": H}
out["I0_error_base_contra_b29"] = round(float(np.abs(base - np.array(Q["base"]["perdida_ida"])).mean()), 5)
out["I0"] = "pasa" if out["I0_error_base_contra_b29"] < 0.01 else "falla"
if not H:
    out["resultado"] = "sin_medida"; print(json.dumps(out, indent=1)); sys.exit()
ARMS = ("una|peso", "una|umbral", "fuentes|umbral")

def margin(r, frame):
    return np.mean([r["margen"][f] for f in frame], 0) if isinstance(frame, list) else np.array(r["margen"][frame])

def per_set(arm, d, frame, t):
    """Fracción que resiste del tercio t, por conjunto."""
    v = []
    for s in range(8):
        r = R["corridas"][f"s{s}|{arm}|d{d}"]
        m, idx = margin(r, frame), np.array(r["sobrescritos"])
        v.append(float((m[third[idx] == t] > 0).mean()))
    return np.array(v)

def pooled(arm, d, frame, t):
    m = np.concatenate([margin(R["corridas"][f"s{s}|{arm}|d{d}"], frame) for s in range(8)])
    idx = np.concatenate([R["corridas"][f"s{s}|{arm}|d{d}"]["sobrescritos"] for s in range(8)])
    return round(float((m[third[idx] == t] > 0).mean()), 3)

out["resiste"] = {f"{a}|d{d}": {f: {t: pooled(a, d, f if f == "T0" else H, t) for t in ("seguro", "medio", "dudoso")} for f in ("T0", "H")}
                  for a in ARMS for d in (10, 40)}
tab = out["resiste"]
out["C0"] = "pasa" if 1 - tab["fuentes|umbral|d40"]["H"]["dudoso"] >= 0.8 else "falla"
v1 = tab["una|umbral|d10"]["H"]["seguro"] >= 0.5
v2 = 1 - tab["fuentes|umbral|d40"]["H"]["seguro"] >= 0.8
v3 = 1 - tab["una|umbral|d10"]["H"]["dudoso"] >= 0.8
out["R1"] = {"V1": v1, "V2": v2, "V3": v3, "video": bool(v1 and v2 and v3)}
dp = per_set("una|umbral", 10, H, "seguro") - per_set("una|peso", 10, H, "seguro")
out["P1_pareada_resiste_seguro"] = [round(x, 3) for x in ci(dp)]
out["P1"] = "pasa" if holds(dp, 0.5) else "falla"

def spill(arm, d):
    return np.array([np.delete(np.array(R["corridas"][f"s{s}|{arm}|d{d}"]["subida_ida"]), R["corridas"][f"s{s}|{arm}|d{d}"]["sobrescritos"]).mean() for s in range(8)])

out["subida_resto"] = {f"{a}|d{d}": round(float(spill(a, d).mean()), 3) for a in ARMS for d in (10, 40)}
dk = spill("una|umbral", 10) - spill("una|peso", 10)
out["K1_pareada_subida_resto"] = [round(x, 3) for x in ci(dk)]
out["K1"] = "pasa" if holds(dk, 0.0) else "falla"
ab = [a for s in range(8) for a, i in zip(R["corridas"][f"s{s}|fuentes|umbral|d40"]["abiertos"], R["corridas"][f"s{s}|fuentes|umbral|d40"]["sobrescritos"]) if third[i] == "seguro"]
out["seguros_abiertos_fuentes_umbral_d40"] = round(float(np.mean(ab)), 3)
print(json.dumps(out, indent=1))
