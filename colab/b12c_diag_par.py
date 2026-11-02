"""B12c, diagnóstico: con p = 1, `par` llega a 1,0 en CPU (semilla 0, paso 4500) y queda en 0 durante 20.000 pasos
en GPU (5 de 5 semillas). La GPU entrena en fp32 y evalúa en bf16. Este script entrena la base con p = 1 en GPU
y cada 500 pasos evalúa `par` en bf16 y en fp32 (copia del modelo en CPU), con el acierto por byte.
Uso: .venv/bin/python -m colab.b12c_diag_par SALIDA.json [pasos] [semilla]
"""

import copy
import json
import random
import sys

from colab.b9_bits import DEVICE, SINGLES, SIZE, eval_sets, train
from colab.b12_raices import E0
from mf_moe.model import build
from mf_moe.train import adamw, scores


def main(path: str, max_steps: int = 12000, seed: int = 0) -> None:
    ev = eval_sets(SINGLES, p=1.0)
    model = build(E0, seed=seed, **SIZE).to(DEVICE)
    opt, rng, hist = adamw(model), random.Random(seed), []
    for step in range(500, max_steps + 1, 500):
        train(model, opt, SINGLES, ev, 500, rng, "diag:1.0", p=1.0, every=500)
        bf16 = scores(model, ev["par"], DEVICE)
        fp32 = scores(copy.deepcopy(model).cpu(), ev["par"], "cpu")
        hist.append({"paso": step, "bf16": bf16[:2], "fp32": fp32[:2], "fp32_por_posicion": [round(x, 3) for x in fp32[2]]})
        print(json.dumps(hist[-1]), flush=True)
        json.dump({"semilla": seed, "historia": hist}, open(path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
