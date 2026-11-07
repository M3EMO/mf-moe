"""B22 (mapa mf-moe-generalidad, ticket 02): ¿el banco de B20 sobrevive con pocas capas entrenables?

OLMoE no entra en una L4 con 12 de 16 capas entrenables. Antes de descargarlo, esto mide en Granite si 1 o 4 capas finales alcanzan.
Sirve solo la comparación con B20 en el mismo conjunto: congela por `requires_grad` dentro de una sola carga del modelo.
Uso: .venv/bin/python -m colab.b22_profundidad SALIDA.json
"""

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b20_matriz as b20
from colab.b16_conceptos import DEVICE, TRAIN_FROM, name, probes, wordnet
from colab.b21_contencion import given_up
from mf_moe import granite
from mf_moe.phase2 import set_trainable

ARMS = {"p18": 18, "p4": 4, "p1": 1}  # capas finales entrenables; p18 repite B20 y prueba que el congelado no cambia nada
MAX_HOURS = 1.5
GB = 1024**3


class Depth:
    """Se engancha a `b18.run_set` como el probe de B19: congela las capas iniciales y mide la memoria de entrenar."""

    def __init__(self, arm: str):
        self.arm, self.n = arm, ARMS[arm]

    def pre(self, model, tok, new, keep) -> None:
        layers = model.model.layers
        self.off = [p for layer in layers[: max(0, len(layers) - self.n)] for p in layer.parameters() if p.requires_grad]
        for p in self.off:
            p.requires_grad_(False)
        self.free = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.base = self.peak = 0.0
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.base = torch.cuda.memory_allocated() / GB

    def post(self, model, state: dict) -> None:
        if DEVICE == "cuda":
            self.peak = torch.cuda.max_memory_allocated() / GB
        for p in self.off:
            p.requires_grad_(True)

    def final(self, model, tok) -> dict:
        return {"brazo": self.arm, "capas": self.n, "parametros_libres": self.free, "gb_base": round(self.base, 2), "gb_pico": round(self.peak, 2)}


def plan(areas=None) -> list[tuple]:
    areas = list(b20.TOPICS) if areas is None else areas
    return [(areas[0], 0, "p18")] + [(a, 0, arm) for a in areas for arm in ("p4", "p1")]


def main(path: str, model=None, tok=None) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    if DEVICE == "cuda":
        res["gpu"] = {"nombre": torch.cuda.get_device_name(0), "gb_total": round(torch.cuda.get_device_properties(0).total_memory / GB, 2)}
        print(json.dumps(res["gpu"]), flush=True)
    ev, chosen, pools = b20.split(wn)
    res |= {"temas": b20.TOPICS, "evaluacion": chosen, "config": {"brazos": ARMS, "train_from": TRAIN_FROM, "max_horas": MAX_HOURS}}
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    H, E = probes(model, tok, [name(x) for x in ev])
    antes = b17.def_losses(model, tok, [b18.fact(x, x.definition()) for x in ev]).round(4).tolist()
    t0 = time.time()
    for area, s, arm in plan():
        key = f"{area}|s{s}|{arm}"
        if key in res["areas"] or given_up(res["areas"], arm):
            continue
        if time.time() - t0 > MAX_HOURS * 3600:
            res["corte"] = f"tope de {MAX_HOURS} h"
            break
        root = wn.synset(area)
        base = {"H": H, "E": E, "antes": antes, "raiz": [round(x.wup_similarity(root) or 0.0, 4) for x in ev]}
        res["areas"][key] = r = b18.run_set(model, tok, wn, area, ev, pools[area], base, state, s, 0, probe=Depth(arm))
        print(json.dumps({"corrida": key, "subida_media": round(float(np.mean(r["subida"])), 3), "nuevos": round(r["nuevos_final"], 3),
                          "pasos": r["pasos"], "gb_pico": r["gb_pico"], "segundos": r["segundos"]}), flush=True)  # fmt: skip
        res["corte"] = b18.gate(r) if len(res["areas"]) == 1 else None
        json.dump(res, open(path, "w"), ensure_ascii=False)
        if res["corte"]:
            print(json.dumps({"corte": res["corte"]}), flush=True)
            return
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:2])
