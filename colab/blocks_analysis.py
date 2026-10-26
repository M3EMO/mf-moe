"""B1 (mapa de bloques): árbol de composición de las neuronas de los expertos de Granite, en CPU y solo con pesos.

Uso: .venv/bin/python colab/blocks_analysis.py SALIDA.json [capas separadas por coma]
"""

import glob
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

from mf_moe.blocks import composition, neuron_signatures, shared_atoms
from mf_moe.granite import MODEL_ID


def tensors(layer: int) -> tuple[np.ndarray, np.ndarray]:
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/*.safetensors")
    pre, out = f"model.layers.{layer}.block_sparse_moe.", {}
    for f in glob.glob(snap):
        with safe_open(f, "pt") as s:
            for name in ("input_linear", "output_linear"):
                if pre + name + ".weight" in s.keys():
                    out[name] = s.get_tensor(pre + name + ".weight").float().numpy()
    return out["input_linear"], out["output_linear"]


def main(path: str, layers: list[int]) -> None:
    res = {"modelo": MODEL_ID, "capas": {}}
    for L in layers:
        t = time.time()
        X, lab = neuron_signatures(*tensors(L))
        res["capas"][L] = {"composicion": composition(X, lab), "atomos_compartidos": shared_atoms(X, lab), "segundos": round(time.time() - t)}
        print(json.dumps({"capa": L, "segundos": res["capas"][L]["segundos"]}), flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,6,12,18,23").split(",")])
