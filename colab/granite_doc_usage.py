"""Control positivo de modularidad en Granite (ticket 15, tras mf-moe-5tq.20): ¿cada documento usa un grupo chico
de expertos, como EMO, o los usa todos, como StdMoE? Es el único medidor que separó EMO de StdMoE.

Mismos textos que emo_analysis (`.cache/runs/emo/textos_64.json`), 256 tokens, 32 documentos por dominio. El
Recorder cuenta el uso por "dominio"; acá cada documento es su propio dominio.
Uso: .venv/bin/python colab/granite_doc_usage.py SALIDA.json [textos.json]
"""

import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from colab.emo_analysis import DOMAINS, LEN, fingerprint
from mf_moe.expert_tree import Recorder
from mf_moe.granite import MODEL_ID

DOCS = 32


def main(path: str, corpus: str = ".cache/runs/emo/textos_64.json") -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    texts = json.load(open(corpus))
    full = [[x[:LEN] for x in (tok(t)["input_ids"] for t in texts[d]) if len(x) >= LEN][:DOCS] for d in DOMAINS]
    dom = np.repeat(np.arange(len(DOMAINS)), [len(f) for f in full])
    rec = Recorder(model, len(dom))
    with torch.no_grad():
        for i, x in enumerate(x for f in full for x in f):
            rec.domain = i
            model(torch.tensor([x]))
    rec.close()
    uso = rec.usage.transpose(1, 0, 2)  # (documentos, capas, expertos), como emo_analysis
    np.save(path.replace(".json", "_uso.npy"), uso)
    res = {"modelo": MODEL_ID, "documentos": np.bincount(dom).tolist(), "expertos": rec.n, "k": rec.k,
           "huella": fingerprint({d: uso[dom == i] for i, d in enumerate(DOMAINS)})}  # fmt: skip
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps(res["huella"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
