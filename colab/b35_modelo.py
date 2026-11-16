"""B35 y B36: la puerta (B29) y el umbral (B31) en otro modelo. Prerregistro en el ticket 14 (mf-moe-generalidad).
B35 corre OLMo-2-0425-1B (denso, otra familia, datos abiertos). B36 corre Granite-3.1-3B-A800M (misma familia, otro tamaño).
Una sola VM hace tres etapas con el mismo modelo cargado:
1. banco: el filtro de b29.banco con este modelo (los mismos 1.200 hechos de UniEdit, filtrados por la ida).
2. puerta: b29.main, con los brazos oráculo, azar y sonda en la dosis 10.
3. umbral: b30.main con los brazos una|peso, una|umbral y fuentes|umbral, y el tag «b29».
Las capas que se entrenan y las de la sonda escalan con la profundidad: Granite 1B entrena desde la capa 6 de 24.
Uso: python -m colab.b35_modelo MODELO_HF SALIDA.json
"""

import json
import os
import sys

import torch

from colab import b23_uniedit as b23
from colab import b27_puerta as b27
from colab import b29_replica as b29
from colab import b30_fuentes as b30
from colab.b16_conceptos import DEVICE, TRAIN_FROM
from mf_moe import granite
from mf_moe.phase2 import set_trainable

HECHOS, CANDS, REF_LAYERS = "colab/data/b23_hechos.json", "colab/data/b25_candidatos.json", 24
ARMS_U = (("una", "peso"), ("una", "umbral"), ("fuentes", "umbral"))
STAGES = ("banco", "puerta", "umbral")


def scaled(layers: tuple, n: int) -> tuple:
    """Las capas de Granite 1B (24) llevadas a un modelo de n capas, a la misma profundidad relativa."""
    return tuple(max(1, round(l * n / REF_LAYERS)) for l in layers)


def loader(model_id: str):
    """Reemplazo de b23.load para otro modelo: pesos en fp32, y entrenable el mismo cuarto final de capas que en Granite 1B."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    n = AutoConfig.from_pretrained(model_id).num_hidden_layers

    def load(model, tok, trainable: bool):
        if model is not None:
            return model, tok
        if AutoConfig.from_pretrained(model_id).model_type == "granitemoe":  # la misma carga que granite.load, con otro repo
            model = granite.GraniteMoeForCausalLM.from_pretrained(model_id, dtype=torch.float32, experts_implementation="grouped_mm_bf16")
            model = granite.patch_routers(model.to(DEVICE))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEVICE)
        if trainable:
            set_trainable(model, scaled((TRAIN_FROM,), n)[0])
        return model, AutoTokenizer.from_pretrained(model_id)

    return load, n


def run(out: str, model, tok) -> None:
    """Las tres etapas. Cada etapa escribe su archivo; `out` junta los tres para que el runner los baje y los resuba."""
    parts = {k: out.replace(".json", f"_{k}.json") for k in STAGES}
    if os.path.exists(out):  # retomar en una VM nueva: el runner solo resube `out`
        for k, v in json.load(open(out)).items():
            if not os.path.exists(parts[k]):
                json.dump(v, open(parts[k], "w"), ensure_ascii=False)

    def save() -> None:
        json.dump({k: json.load(open(p)) for k, p in parts.items() if os.path.exists(p)}, open(out, "w"), ensure_ascii=False)

    if not os.path.exists(parts["banco"]):
        b29.banco(HECHOS, CANDS, parts["banco"], model, tok)
    save()
    b29.main(parts["puerta"], parts["banco"], CANDS, model, tok)
    save()
    b30.main(parts["umbral"], parts["banco"], "b29", ARMS_U, model=model, tok=tok)
    save()


def main(model_id: str, out: str) -> None:
    load, n = loader(model_id)
    b23.load = load  # b29 y b30 cargan con b23.load
    b27.LAYERS = scaled(b27.LAYERS, n)
    print(json.dumps({"modelo": model_id, "capas": n, "entrena_desde": scaled((TRAIN_FROM,), n)[0], "capas_sonda": b27.LAYERS}), flush=True)
    model, tok = load(None, None, True)
    run(out, model, tok)


if __name__ == "__main__":
    main(*sys.argv[1:3])
