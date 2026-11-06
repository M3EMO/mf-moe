"""B20 (ticket 15): la matriz concepto × entrenamiento con temas distintos, para probar un mapa jerárquico aprendido.

B19 tiene 5 conjuntos al azar de un área: 4 dimensiones y poca variedad temática, no puede mostrar una jerarquía. Acá
10 temas nuevos en 5 supertemas y una evaluación fija de 60 conceptos por tema. La primera etapa corre 3 conjuntos por
tema y repite uno por supertema con otro orden de lotes, para medir el ruido. La segunda suma 3 conjuntos por tema, solo
si la señal alcanza: la validación sintética (`colab.b20_analisis`) muestra que por debajo no hay potencia.
Banco: el de B18 (`b18.run_set`). Uso: .venv/bin/python -m colab.b20_matriz SALIDA.json
"""

import json
import os
import random
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab.b16_conceptos import DEVICE, TRAIN_FROM, name, probes, wordnet
from colab.b20_analisis import signal
from mf_moe import granite
from mf_moe.phase2 import set_trainable

TOPICS = {"fish.n.01": "animal", "mammal.n.01": "animal", "tree.n.01": "planta", "herb.n.01": "planta", "vehicle.n.01": "artefacto",
          "tool.n.01": "artefacto", "worker.n.01": "persona", "leader.n.01": "persona", "dish.n.02": "comida", "vegetable.n.01": "comida"}  # fmt: skip
STAGE1, N_SETS, N_EVAL, MIN_SIGNAL = 3, 6, 60, 0.12


def split(wn) -> tuple[list, dict, dict]:
    """Evaluación fija por tema y resto para entrenar. Un concepto en dos temas queda afuera: rompería las etiquetas."""
    members = {a: set(wn.synset(a).closure(lambda x: x.hyponyms())) for a in TOPICS}
    seen = [x for m in members.values() for x in m]
    repeated = {x for x in seen if seen.count(x) > 1}
    ev, chosen, pools = [], {}, {}
    for a, m in members.items():
        clean = sorted(m - repeated, key=lambda s: s.name())
        take = random.Random(f"evaluacion:{a}").sample(clean, N_EVAL)
        ev += take
        chosen[a] = [x.name() for x in take]
        pools[a] = [x for x in clean if x not in take]
    return ev, chosen, pools


def stages() -> tuple[list, list]:
    """Corridas (área, conjunto, orden) de cada etapa. Se repite el primer conjunto del primer tema de cada supertema."""
    first, seen = [], set()
    for a, parent in TOPICS.items():
        first += [(a, s, 0) for s in range(STAGE1)]
        if parent not in seen:
            first.append((a, 0, 1))
            seen.add(parent)
    return first, [(a, s, 0) for a in TOPICS for s in range(STAGE1, N_SETS)]


def main(path: str, model=None, tok=None) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    ev, chosen, pools = split(wn)
    res |= {"temas": TOPICS, "evaluacion": chosen, "config": {"etapa1": STAGE1, "n_sets": N_SETS, "n_eval": N_EVAL, "n_train": b18.N_TRAIN, "anclas": b18.ANCHORS, "train_from": TRAIN_FROM}}
    # copia en CPU solo lo que cambia, una vez: el modelo vuelve a este estado
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    H, E = probes(model, tok, [name(x) for x in ev])
    antes = b17.def_losses(model, tok, [b18.fact(x, x.definition()) for x in ev]).round(4).tolist()

    def run(plan: list) -> bool:
        for area, s, order in plan:
            key = f"{area}|s{s}" + (f"|r{order}" if order else "")
            if key in res["areas"]:
                continue
            root = wn.synset(area)
            base = {"H": H, "E": E, "antes": antes, "raiz": [round(x.wup_similarity(root) or 0.0, 4) for x in ev]}
            res["areas"][key] = r = b18.run_set(model, tok, wn, area, ev, pools[area], base, state, s, order)
            print(json.dumps({"corrida": key, "entrenados": round(r["subida_entrenados_verdad"], 3), "nuevos": round(r["nuevos_final"], 3),
                              "subida_media": round(float(np.mean(r["subida"])), 3), "pasos": r["pasos"], "segundos": r["segundos"]}), flush=True)  # fmt: skip
            res["corte"] = b18.gate(r) if len(res["areas"]) == 1 else None
            json.dump(res, open(path, "w"), ensure_ascii=False)
            if res["corte"]:
                print(json.dumps({"corte": res["corte"]}), flush=True)
                return False
        return True

    first, second = stages()
    if not run(first):
        return
    res["senal"] = signal(res["areas"])
    print(json.dumps({"senal": res["senal"]}), flush=True)
    if res["senal"] < MIN_SIGNAL:
        res["corte"] = f"señal {res['senal']:.3f} < {MIN_SIGNAL}: sin potencia para comparar árbol y plano"
        json.dump(res, open(path, "w"), ensure_ascii=False)
        print(json.dumps({"corte": res["corte"]}), flush=True)
        return
    run(second)


if __name__ == "__main__":
    main(*sys.argv[1:2])
