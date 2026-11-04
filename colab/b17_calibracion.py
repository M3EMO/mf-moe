"""Calibración de B17 (ticket 15), exploratoria: ¿qué variante da un daño específico (entrenados ≫ franja lejana)?

B17 y B17b cortaron en C0: la franja lejana subía 0,8-1,0 nats, contra 1,4-2,3 de los entrenados. Variantes, sobre
las dos áreas ya vistas (dog, bird), para que la confirmación corra en áreas nuevas:
- base: como B17;
- ancla: cada lote suma 16 definiciones verdaderas de 512 conceptos lejanos (conjunto de preservación);
- ancla_medio: ancla, entrenando solo las capas 6-11 (donde la edición ubica los hechos, ROME).
Uso: .venv/bin/python -m colab.b17_calibracion SALIDA.json
"""

import json
import os
import sys

import numpy as np

from colab.b16_conceptos import wordnet
from colab.b17_conflicto import DEVICE, FAR_BAND, TRAIN_FROM, run_area
from mf_moe import granite
from mf_moe.phase2 import set_trainable

AREAS = ("dog.n.01", "bird.n.01")
VARIANTS = {"base": (0, None), "ancla": (512, None), "ancla_medio": (512, 12)}  # (anclas, primera capa congelada)


def specificity(r: dict) -> dict:
    s, w = np.array(r["subida"]), np.array(r["raiz"])
    far = float(s[w < FAR_BAND].mean())
    return {"entrenados": round(r["subida_entrenados_verdad"], 3), "lejos": round(far, 3),
            "cerca": round(float(s[w >= 0.8].mean()), 3), "cociente": round(r["subida_entrenados_verdad"] / max(far, 1e-3), 2),
            "nuevos": round(r["nuevos_final"], 3), "pasos": r["pasos"]}  # fmt: skip


def main(path: str, model=None, tok=None, areas=AREAS) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {}
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    for variant, (anchor, cut) in VARIANTS.items():
        layers = model.model.layers[cut:] if cut is not None else []
        for layer in layers:
            layer.requires_grad_(False)  # sigue en fp32: solo deja de entrenarse en esta variante
        for area in areas:
            key = f"{area}|{variant}"
            if key not in res:
                res[key] = r = run_area(model, tok, wn, area, 0, anchor)
                print(json.dumps({"clave": key, **specificity(r), "segundos": r["segundos"]}), flush=True)
                json.dump(res, open(path, "w"), ensure_ascii=False)
        for layer in layers:
            layer.requires_grad_(True)


if __name__ == "__main__":
    main(sys.argv[1])
