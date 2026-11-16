"""Premisa en FlyWire 783: ¿las sinapsis por par crecen con la importancia de los socios? Prerregistro en el ticket 16 de mf-moe-generalidad.
M1: Spearman de sinapsis por par contra el grado de salida del emisor y el grado de entrada del receptor.
M2: media de sinapsis por par dentro del club rico (grado total > 37) contra el resto.
M3: Spearman entre las dos direcciones de cada par recíproco.
M4: M1 a M3 dentro de cada hemisferio (aristas izquierda-izquierda y derecha-derecha).
Uso: python -m colab.fw_premisa CONEXIONES.feather ANOTACIONES.tsv SALIDA.json
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THRESHOLD, RICH, PERMS = 5, 37, 20


def edges(conn: pd.DataFrame, threshold: int = THRESHOLD) -> pd.DataFrame:
    """Suma las filas por neurópilo en una arista por par, sin autapsis, con al menos `threshold` sinapsis."""
    e = conn.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()
    e = e[(e.pre_pt_root_id != e.post_pt_root_id) & (e.syn_count >= threshold)]
    return e.rename(columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post", "syn_count": "w"}).reset_index(drop=True)


def degrees(e: pd.DataFrame) -> pd.DataFrame:
    out, inn = e.pre.value_counts(), e.post.value_counts()
    ids = out.index.union(inn.index)
    d = pd.DataFrame({"out": out.reindex(ids, fill_value=0), "in": inn.reindex(ids, fill_value=0)})
    d["total"] = d["out"] + d["in"]
    return d


def stats(e: pd.DataFrame, d: pd.DataFrame, seed: int = 0) -> dict:
    w = e.w.to_numpy(float)
    dout, din = d["out"].reindex(e.pre).to_numpy(), d["in"].reindex(e.post).to_numpy()
    rich = (d["total"].reindex(e.pre).to_numpy() > RICH) & (d["total"].reindex(e.post).to_numpy() > RICH)
    rng = np.random.default_rng(seed)
    null = [spearmanr(rng.permutation(w), dout)[0] for _ in range(PERMS)]  # control: el peso sin relación con el grado
    back = e.set_index(["pre", "post"]).w
    rec = e[e.set_index(["post", "pre"]).index.isin(back.index)]
    rec = rec[rec.pre < rec.post]  # cada par recíproco una vez
    w_ji = back.reindex(list(zip(rec.post, rec.pre))).to_numpy()
    return {"aristas": len(e), "M1_pre_salida": float(spearmanr(w, dout)[0]), "M1_post_entrada": float(spearmanr(w, din)[0]),
            "M1_nulo_sd": float(np.std(null)), "M2_rico": float(w[rich].mean()), "M2_resto": float(w[~rich].mean()),
            "M2_razon": float(w[rich].mean() / w[~rich].mean()), "M2_frac_rico": float(rich.mean()),
            "M3_pares": len(rec), "M3_spearman": float(spearmanr(rec.w, w_ji)[0]) if len(rec) > 2 else float("nan")}  # fmt: skip


def verdict(res: dict) -> dict:
    hemis = [res["izquierda"], res["derecha"]]
    m1 = any(all(h[k] >= 0.2 for h in hemis) for k in ("M1_pre_salida", "M1_post_entrada"))
    return {"M1_pasa": m1, "brazo_B38": "proteger pesos" if m1 else "capacidad"}


def main(conn_path: str, ann_path: str, out: str) -> None:
    conn = pd.read_feather(conn_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    side = pd.read_csv(ann_path, sep="\t", usecols=["root_id", "side"]).set_index("root_id")["side"]
    res = {"filas": len(conn)}
    e = edges(conn)
    d = degrees(e)
    res["cerebro"] = stats(e, d)
    for name, s in (("izquierda", "left"), ("derecha", "right")):
        res[name] = stats(e[(side.reindex(e.pre).to_numpy() == s) & (side.reindex(e.post).to_numpy() == s)], d)
    e1 = edges(conn, 1)
    res["umbral_1"] = stats(e1, degrees(e1))
    res["veredicto"] = verdict(res)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
