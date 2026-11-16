"""B37: ¿una intuición de daño (nivel 4 del video) protege mejor que el azar al elegir qué repasar? Prerregistro en el ticket 15.
Banco, conjuntos (tag «b25»), semillas y lotes de B25; solo la ida. Cada edición suma K hechos viejos de repaso con su valor real.
- azar: K hechos no sobrescritos, al azar.
- guiado: los K hechos con más fragilidad prevista. El previo sale de corridas guardadas de otros conjuntos (`previo`, en CPU).
- oraculo: los K hechos que más subieron en B25 (ida, dosis 40, mismo conjunto, sin repaso). Es el techo: mira el resultado.
Con K = 0 el brazo repite la ida de B25 (un test lo verifica).
Uso: python -m colab.b37_repaso previo SALIDA_PREVIO.json DIR_CORRIDAS   (CPU: arma el previo desde los JSON guardados)
     python -m colab.b37_repaso main SALIDA.json BANCO.json PREVIO.json B25.json   (GPU)
"""

import json
import os
import random
import sys

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24

K, DOSES, ARMS = 20, (10, 40), ("azar", "guiado", "oraculo")
EXPERIENCE = ("b27_puerta_gpu/b27_puerta.json", "b27b_senal_gpu/b27b_senal.json", "b27c_sonda_gpu/b27c_sonda.json", "b30_fuentes_gpu/b30_fuentes.json",
              "b32_sonda_gpu/b32_sonda.json", "b33_sonda2_gpu/b33_sonda2.json", "b34_contenido_gpu/b34_contenido.json")  # fmt: skip


def prior_from(rows: np.ndarray, sets: np.ndarray, s: int) -> np.ndarray:
    """Fragilidad prevista de cada hecho para el conjunto s: media de la subida normalizada por fila, solo en filas de otros conjuntos."""
    Z = rows[sets != s]
    Zn = (Z - np.nanmean(Z, 1, keepdims=True)) / np.nanstd(Z, 1, keepdims=True)
    return np.nan_to_num(np.nanmean(Zn, 0), nan=0.0)


def previo(out: str, runs_dir: str) -> None:
    rows, sets = [], []
    for f in EXPERIENCE:
        for key, r in json.load(open(os.path.join(runs_dir, f)))["corridas"].items():
            v = np.array(r["subida_ida"], float)
            v[r["sobrescritos"]] = np.nan  # solo cuenta el daño a los hechos que esa edición no tocó
            rows.append(v)
            sets.append(int(key.split("|")[0][1:]))
    rows, sets = np.array(rows), np.array(sets)
    json.dump({"fuentes": list(EXPERIENCE), "filas": len(rows), **{f"s{s}": prior_from(rows, sets, s).round(5).tolist() for s in range(b24.SETS)}},
              open(out, "w"))  # fmt: skip


def picks(s: int, rest: list[int], prior: list[float], b25_rise: list[float], k: int) -> dict:
    top = lambda score: sorted(sorted(rest, key=lambda i: -score[i])[:k])
    return {"azar": sorted(random.Random(f"b37:azar:{s}").sample(rest, k)), "guiado": top(prior), "oraculo": top(b25_rise)}


def main(path: str, bank_path: str, prior_path: str, b25_path: str, model=None, tok=None, k: int = K, arms: tuple = ARMS) -> None:
    facts, prior, B = (json.load(open(p, encoding="utf-8")) for p in (bank_path, prior_path, b25_path))
    B = B["corridas"]
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    res["config"] = {"n_hechos": len(facts), "k": k, "dosis": list(DOSES), "brazos": list(arms), "tag": "b25", "previo_filas": prior["filas"],
                     "lr": b17.LR, "batch": b17.BATCH}  # fmt: skip
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    state = {n: v.detach().to("cpu", copy=True) for n, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    n = len(facts)
    n_over = n // b24.SETS
    order = random.Random("b25:conjuntos").sample(range(n), n)  # los conjuntos de b24.stage2 con el tag «b25»
    for s in range(b24.SETS):
        over = sorted(order[s * n_over : (s + 1) * n_over])
        ref = B[f"s{s}|ida|d40"]
        assert ref["sobrescritos"] == over, "los conjuntos no coinciden con B25"
        rest = [i for i in range(n) if i not in over]
        chosen = picks(s, rest, prior[f"s{s}"], ref["subida_ida"], k)
        for arm in arms:
            if f"s{s}|{arm}|d{max(DOSES)}" in res["corridas"]:
                continue
            new = [b24.fwd(facts[i], facts[i]["valor_nuevo"]) for i in over]
            items = new + [b24.fwd(facts[i]) for i in chosen[arm]]

            def hook(step: int) -> None:
                if step in DOSES:
                    L_i, _ = b24.losses(model, tok, facts)
                    rise = L_i[:, 0] - base_i[:, 0]
                    res["corridas"][f"s{s}|{arm}|d{step}"] = {"conjunto": s, "brazo": arm, "sobrescritos": over, "repaso": chosen[arm], "pasos": step,
                                                              "subida_ida": rise.round(4).tolist(), "perdida_nuevo": b17.def_losses(model, tok, new).round(4).tolist()}  # fmt: skip
                    others = np.setdiff1d(np.arange(n), over)
                    print(json.dumps({"corrida": f"s{s}|{arm}|d{step}", "resto": round(float(rise[others].mean()), 3),
                                      "nuevos": round(float(np.mean(res["corridas"][f"s{s}|{arm}|d{step}"]["perdida_nuevo"])), 3)}), flush=True)  # fmt: skip

            torch.manual_seed(1000 + s)
            b24.train(model, tok, items, float("-inf"), max(DOSES), b17.CHECK_EVERY, f"b25:lotes:{s}:ida", hook)
            model.load_state_dict(state, strict=False)
            json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps({"FIN": len(res["corridas"])}), flush=True)


if __name__ == "__main__":
    {"previo": previo, "main": main}[sys.argv[1]](*sys.argv[2:])
