"""Análisis prerregistrado de B30 (ticket 12). Uso: python -m colab.b30_analisis B30.json B27B.json"""
import json, sys
import numpy as np

R, Q = (json.load(open(p)) for p in sys.argv[1:3])
base = np.array(R["base"]["perdida_ida"]); n = len(base)
rank = np.argsort(np.argsort(base))  # 0 = el más seguro
third = np.where(rank < n / 3, "seguro", np.where(rank >= 2 * n / 3, "dudoso", "medio"))
nt = len(R["config"]["plantillas"])
wins = R["base"]["gana_por_frase"]
H = [h for h, g in zip(("H1", "H2"), wins[nt:]) if g >= 0.6]
out = {"base_gana": {"T0": round(wins[0], 3), "H1": round(wins[nt], 3), "H2": round(wins[nt + 1], 3)}, "H_usadas": H}

def pooled(source, gate, d, frame):
    m, idx = [], []
    for s in range(8):
        r = R["corridas"][f"s{s}|{source}|{gate}|d{d}"]
        mm = np.mean([r["margen"][f] for f in frame], 0) if isinstance(frame, list) else np.array(r["margen"][frame])
        m += list(mm); idx += r["sobrescritos"]
    return np.array(m), np.array(idx)

def resist(source, gate, d, frame, t):
    m, idx = pooled(source, gate, d, frame)
    return round(float((m[third[idx] == t] > 0).mean()), 3)

d_int = [np.abs(np.array(R["corridas"][f"s{s}|una|peso|d10"]["subida_ida"]) - np.array(Q["corridas"][f"s{s}|oraculo_fuerte|d10"]["subida_ida"])).mean() for s in range(8)]
out["integridad1"] = round(float(np.mean(d_int)), 4)
if not H:
    out["resultado"] = "sin_medida"; print(json.dumps(out, indent=1)); sys.exit()
tab = {}
for src in ("una", "fuentes"):
    for g in ("peso", "umbral"):
        for d in (10, 40):
            tab[f"{src}|{g}|d{d}"] = {f: {t: resist(src, g, d, f if f == "T0" else H, t) for t in ("seguro", "medio", "dudoso")} for f in ("T0", "H")}
out["resiste"] = tab
out["C0"] = "pasa" if 1 - tab["fuentes|peso|d40"]["H"]["dudoso"] >= 0.8 else "falla"
for g in ("peso", "umbral"):
    v1 = tab[f"una|{g}|d10"]["H"]["seguro"] >= 0.5
    v2 = 1 - tab[f"fuentes|{g}|d40"]["H"]["seguro"] >= 0.8
    v3 = 1 - tab[f"una|{g}|d10"]["H"]["dudoso"] >= 0.8
    out[g] = {"V1": v1, "V2": v2, "V3": v3, "video": bool(v1 and v2 and v3)}
spill = {}
for k in tab:
    src, g, d = k.split("|")
    rs = [np.delete(np.array(R["corridas"][f"s{s}|{src}|{g}|{d}"]["subida_ida"]), R["corridas"][f"s{s}|{src}|{g}|{d}"]["sobrescritos"]).mean() for s in range(8)]
    spill[k] = round(float(np.mean(rs)), 3)
out["subida_resto"] = spill
ab = [a for s in range(8) for a, i in zip(R["corridas"][f"s{s}|fuentes|umbral|d40"]["abiertos"], R["corridas"][f"s{s}|fuentes|umbral|d40"]["sobrescritos"]) if third[i] == "seguro"]
out["seguros_abiertos_fuentes_umbral_d40"] = round(float(np.mean(ab)), 3)
print(json.dumps(out, indent=1))
