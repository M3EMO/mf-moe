"""B1b (mapa de bloques): árbol de composición de las neuronas de los expertos de Granite por función, en CPU.

Firma = activación de cada neurona sobre tokens fijos de 5 dominios (`activation_signatures`). Sin la posición 0
ni el 1 % de tokens con más norma: con ellos, muchas neuronas de expertos distintos comparten un pico atípico
y parecen copias (control: capa 23 con 12 % de copias, sin ellos 2 %).
Uso: .venv/bin/python colab/function_blocks_analysis.py SALIDA.json [documentos por dominio]
"""

import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from colab.expert_tree_analysis import texts
from mf_moe.blocks import (
    activation_signatures,
    composition,
    neuron_signatures,
    shared_atoms,
    weight_function_agreement,
)
from mf_moe.expert_tree import Recorder
from mf_moe.granite import MODEL_ID

DOMAINS, LAYERS, LEN = ("fines", "turco", "espanol", "web", "codigo"), (0, 6, 12, 18, 23), 256


def main(path: str, docs: int = 2) -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    rec, pos = Recorder(model, len(DOMAINS), keep_layers=LAYERS, keep_frac=1.0), []
    for i, d in enumerate(DOMAINS):
        rec.domain = i
        with torch.no_grad():
            for t in texts(d, n_docs=docs):
                ids = tok(t)["input_ids"][:LEN]
                pos += range(len(ids))
                model(torch.tensor([ids]))
        print(json.dumps({"dominio": d, "segundos": round(time.time() - t0)}), flush=True)
    rec.close()
    res = {"modelo": MODEL_ID, "dominios": DOMAINS, "documentos": docs, "largo": LEN, "capas": {}}
    for L in LAYERS:
        t = time.time()
        ex = model.model.layers[L].block_sparse_moe.experts
        w_in, w_out = ex.gate_up_proj.detach().float().numpy(), ex.down_proj.detach().float().numpy()
        H = torch.cat([h for h, _, _ in rec.kept[L]]).numpy()
        n = np.linalg.norm(H, axis=1)
        keep = (np.array(pos) > 0) & (n < np.percentile(n, 99))
        Xf, lab = activation_signatures(H[keep], w_in)
        res["capas"][L] = {"tokens": int(keep.sum()), "tokens_quitados": int((~keep).sum()),
                           "composicion": composition(Xf, lab), "atomos_compartidos": shared_atoms(Xf, lab),
                           "atomos_con_todos_los_tokens": shared_atoms(activation_signatures(H, w_in)[0], lab),
                           "pesos_contra_funcion": weight_function_agreement(neuron_signatures(w_in, w_out)[0], Xf),
                           # control: parte de cada firma en su token más fuerte (1/tokens si se reparte parejo)
                           "peso_token_max_medio": float((Xf**2).max(1).mean()),
                           "segundos": round(time.time() - t)}  # fmt: skip
        print(json.dumps({"capa": L, "tokens": int(keep.sum()), "segundos": res["capas"][L]["segundos"]}), flush=True)
        with open(path, "w") as f:
            json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *[int(x) for x in sys.argv[2:]])
