"""B18 (ticket 15): el mismo concepto bajo varios conjuntos entrenados. ¿El cambio del predictor sigue al del daño?

B17c mostró que árbol, estado y router separan adentro de afuera del área, pero ese corte transversal mezcla rasgos
propios de cada concepto (definición, frecuencia, largo). Con la evaluación fija y N_SETS conjuntos entrenados por
área, el efecto fijo por concepto cancela todo eso: solo queda lo que cambia con el conjunto entrenado. Una corrida
extra repite el conjunto 0 con otro orden de lotes, para separar el efecto del conjunto del ruido del entrenamiento.
Banco: el de B17c (donantes lejanos, 512 anclas, corte temprano). Áreas nuevas, sin usar en B16 ni en B17.
Uso: .venv/bin/python -m colab.b18_mismo_concepto SALIDA.json [areas separadas por coma]
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # B17c cayó por OOM en la corrida 15

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab.b16_conceptos import DEVICE, TRAIN_FROM, name, predictors, probes, wordnet
from mf_moe import granite
from mf_moe.phase2 import amp, set_trainable

AREAS = ("clothing.n.01", "insect.n.01", "container.n.01", "beverage.n.01", "fungus.n.01", "reptile.n.01")
N_SETS, N_TRAIN, N_FAR, ANCHORS, MAX_IN = 6, 64, 48, 512, 200
FAR_BAND = 0.4


def split(wn, area: str) -> tuple[list, list]:
    """Evaluación fija del área (un tercio de los miembros, más lejanos) y el resto, de donde salen los entrenados."""
    rng, root = random.Random(f"evaluacion:{area}"), wn.synset(area)
    members = sorted(set(root.closure(lambda x: x.hyponyms())), key=lambda s: s.name())
    inside = rng.sample(members, min(MAX_IN, len(members) // 3))
    taken = {s.name() for s in members}
    nouns = rng.sample(sorted(wn.all_synsets("n"), key=lambda s: s.name()), 4 * N_FAR + 200)
    far = [s for s in nouns if s.name() not in taken and (s.wup_similarity(root) or 0.0) < FAR_BAND][:N_FAR]
    chosen = {s.name() for s in inside}
    return inside + far, [s for s in members if s.name() not in chosen]


def fact(s, definition: str) -> tuple[str, str]:
    return f"A {name(s)} is", f" {definition}."


def run_set(model, tok, wn, area: str, ev, pool, base: dict, state: dict, s: int, order: int) -> dict:
    """Entrena el conjunto `s` con el orden de lotes `order`, mide el daño y restaura el modelo."""
    train = random.Random(f"{area}:{s}").sample(pool, N_TRAIN)
    don = b17.donors(wn, area, {x.name() for x in train + ev}, N_TRAIN + ANCHORS, s)
    don, anchors = don[:N_TRAIN], don[N_TRAIN:]
    new = [fact(x, d.definition()) for x, d in zip(train, don)]
    keep = [fact(x, x.definition()) for x in anchors]
    true_tr = [fact(x, x.definition()) for x in train]
    H_tr, E_tr = probes(model, tok, [name(x) for x in train])
    before_tr, before_new = b17.def_losses(model, tok, true_tr), b17.def_losses(model, tok, new)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=b17.LR, weight_decay=0.0)
    rng, t0, hist, step = random.Random(f"orden:{area}:{s}:{order}"), time.time(), [], 0
    torch.manual_seed(1000 * s + order)
    for step in range(1, b17.MAX_STEPS + 1):
        for part in ([new[i] for i in rng.sample(range(N_TRAIN), b17.BATCH)], [keep[i] for i in rng.sample(range(len(keep)), b17.BATCH)]):
            ids, mask, labels = b17.batch(tok, part)
            with amp(DEVICE):
                (model(input_ids=ids, attention_mask=mask, labels=labels).loss / 2).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % b17.CHECK_EVERY == 0:
            cur = float(b17.def_losses(model, tok, new).mean())
            hist.append({"paso": step, "nuevos": round(cur, 4), "segundos": round(time.time() - t0)})
            if cur < b17.STOP_LOSS:
                break
    after = b17.def_losses(model, tok, [fact(x, x.definition()) for x in ev])
    after_tr, learned = b17.def_losses(model, tok, true_tr), b17.def_losses(model, tok, new)
    model.load_state_dict(state, strict=False)  # cada conjunto parte del mismo Granite
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"area": area, "conjunto": s, "orden": order, "entrenados": [x.name() for x in train], "evaluados": [x.name() for x in ev],
            "donantes": [x.name() for x in don], "pasos": step, "raiz": base["raiz"], "antes": base["antes"],
            "subida": (after - np.array(base["antes"])).round(4).tolist(), "subida_entrenados_verdad": float((after_tr - before_tr).mean()),
            "nuevos_antes": float(before_new.mean()), "nuevos_final": float(learned.mean()), "historia": hist,
            "predictores": predictors(train, ev, H_tr, base["H"], E_tr, base["E"]), "covariables": b17.covariates(tok, train, ev),
            "segundos": round(time.time() - t0)}  # fmt: skip


def gate(r: dict) -> str | None:
    """Puerta preregistrada tras la primera corrida (ticket 15)."""
    far = float(np.mean([x for x, w in zip(r["subida"], r["raiz"]) if w < FAR_BAND]))
    if not (r["nuevos_final"] < b17.STOP_LOSS and r["subida_entrenados_verdad"] > max(0.5, 2 * far)):
        return "C0: sin conflicto específico en la primera corrida"
    if r["segundos"] > b17.MAX_AREA_SECONDS:
        return f"tiempo: {r['segundos']} s en la primera corrida"
    return None


def main(path: str, areas: str = ",".join(AREAS), model=None, tok=None) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    res |= {"config": {"n_sets": N_SETS, "n_train": N_TRAIN, "n_far": N_FAR, "anclas": ANCHORS, "max_in": MAX_IN, "train_from": TRAIN_FROM}}
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    # copia en CPU solo lo que cambia (entrenables y buffers del router), una vez: el modelo vuelve a este estado
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    plan = [(s, 0) for s in range(N_SETS)] + [(0, 1)]  # el último repite el conjunto 0 con otro orden de lotes
    for area in areas.split(","):
        ev, pool = split(wn, area)
        H, E = probes(model, tok, [name(x) for x in ev])
        root = wn.synset(area)
        base = {"H": H, "E": E, "antes": b17.def_losses(model, tok, [fact(x, x.definition()) for x in ev]).round(4).tolist(),
                "raiz": [round(x.wup_similarity(root) or 0.0, 4) for x in ev]}  # fmt: skip
        for s, order in plan:
            key = f"{area}|s{s}" + (f"|r{order}" if order else "")
            if key in res["areas"]:
                continue
            res["areas"][key] = r = run_set(model, tok, wn, area, ev, pool, base, state, s, order)
            print(json.dumps({"corrida": key, "entrenados": round(r["subida_entrenados_verdad"], 3), "nuevos": round(r["nuevos_final"], 3),
                              "subida_media": round(float(np.mean(r["subida"])), 3), "pasos": r["pasos"], "segundos": r["segundos"]}), flush=True)  # fmt: skip
            why = gate(r) if len(res["areas"]) == 1 else None
            res["corte"] = why
            json.dump(res, open(path, "w"), ensure_ascii=False)
            if why:
                print(json.dumps({"corte": why}), flush=True)
                return


if __name__ == "__main__":
    main(*sys.argv[1:3])
