"""B2 (mapa de bloques): ¿el router de Granite admite direcciones jerárquicas? En CPU, con pocos textos.

Uso: .venv/bin/python colab/routing_tree_analysis.py SALIDA.json
"""

import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from colab.expert_tree_analysis import texts
from mf_moe.blocks import hierarchical_routing
from mf_moe.expert_tree import Recorder
from mf_moe.granite import MODEL_ID

DOMAINS, LAYERS, DOCS, LEN = ("fines", "turco", "espanol", "web", "codigo"), (0, 6, 12, 18, 23), 8, 256


def main(path: str) -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    rec = Recorder(model, len(DOMAINS), keep_layers=LAYERS, keep_frac=0.25)
    for i, d in enumerate(DOMAINS):
        rec.domain = i
        ids = [tok(t)["input_ids"][:LEN] for t in texts(d, n_docs=DOCS)]
        with torch.no_grad():
            for x in ids:
                model(torch.tensor([x]))
        print(json.dumps({"dominio": d, "segundos": round(time.time() - t0)}), flush=True)
    rec.close()
    res = {"modelo": MODEL_ID, "dominios": DOMAINS, "capas": {}}
    for L in LAYERS:
        H = torch.cat([h for h, _, _ in rec.kept[L]]).numpy()
        W = model.model.layers[L].block_sparse_moe.router.layer.weight.float().numpy()
        res["capas"][L] = {"tokens": len(H), "niveles": hierarchical_routing(H, W, model.config.num_experts_per_tok)}
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1])
