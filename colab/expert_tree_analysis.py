"""Análisis del árbol natural de expertos (ticket 13): solo inferencia, en una L4.

Para cada modelo: 32 secuencias de 512 tokens por dominio (12 dominios en 4 familias, con el tokenizer del
modelo), ruteo capturado por `mf_moe.expert_tree.Recorder` y medidas de `analyze`. Un modelo que no carga
(sin soporte nativo en transformers; nunca se corre código remoto) o un dominio sin datos se saltean y se
reportan. Uso: python expert_tree_analysis.py OUT
"""

import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mf_moe.expert_tree import DOMAINS, Recorder, analyze

MODELS = [  # (id, cuantizar a 4 bits)
    ("ibm-granite/granite-3.1-1b-a400m-base", False),
    ("allenai/OLMoE-1B-7B-0924", False),
    ("deepseek-ai/DeepSeek-V2-Lite", True),
]
FW2 = {"fines": "fin_Latn", "estonio": "est_Latn", "hungaro": "hun_Latn", "turco": "tur_Latn", "azeri": "azj_Latn",
       "uzbeko": "uzn_Latn", "espanol": "spa_Latn", "portugues": "por_Latn", "italiano": "ita_Latn"}  # fmt: skip
OTHER = {"web": ("HuggingFaceFW/fineweb-edu", "sample-10BT"), "codigo": ("common-pile/stackv2", None),
         "matematica": ("open-web-math/open-web-math", None)}  # fmt: skip
SEQS, LEN, PER_DOC = 32, 512, 2  # a lo sumo 2 secuencias por documento: cada dominio cubre >= 16 documentos


def texts(domain: str, n_docs: int = 120, chars: int = 6000) -> list[str]:
    name, cfg = ("HuggingFaceFW/fineweb-2", FW2[domain]) if domain in FW2 else OTHER[domain]
    out = []
    for row in load_dataset(name, cfg, split="train", streaming=True):
        t = row.get("text") or row.get("content") or ""
        if len(t) > 200:
            out.append(t[:chars])
        if len(out) >= n_docs:
            break
    return out


def seqs(tok, docs: list[str]) -> torch.Tensor:
    out = []
    for d in docs:
        ids = tok(d)["input_ids"]
        out += [ids[i : i + LEN] for i in range(0, len(ids) - LEN + 1, LEN)][:PER_DOC]
        if len(out) >= SEQS:
            break
    return torch.tensor(out[:SEQS])


def main(out: str) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    corpus, res = {}, {}
    for d in DOMAINS:
        try:
            corpus[d] = texts(d)
        except Exception as e:  # noqa: BLE001 - un dominio sin datos se reporta y se saltea
            print(f"== dominio salteado {d}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    doms = [d for d in DOMAINS if corpus.get(d)]
    print(f"== dominios {doms}", flush=True)
    for mid, q4 in MODELS:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=False)
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)} \
                if q4 else {"dtype": torch.bfloat16}  # fmt: skip
            model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=False, device_map="cuda", **kw).eval()
        except Exception as e:  # noqa: BLE001
            res[mid] = {"salteado": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"== modelo salteado {mid}: {res[mid]['salteado']}", flush=True)
            continue
        rec = Recorder(model, len(doms))
        n_seq = {}
        with torch.no_grad():
            for di, d in enumerate(doms):
                rec.domain, x = di, seqs(tok, corpus[d])
                n_seq[d] = len(x)
                for b in x.split(4):
                    model(input_ids=b.to("cuda"))
        rec.close()
        layers = analyze(rec.usage, rec.coact, doms)
        res[mid] = {"expertos": rec.n, "top_k": rec.k, "routers": rec.mods, "secuencias": n_seq, "capas": layers,
                    "uso": rec.usage.tolist(), "segundos": round(time.time() - t0)}  # fmt: skip
        mid_layers = layers[len(layers) // 4 : 3 * len(layers) // 4]
        summary = {k: round(sum(c[k] for c in mid_layers) / len(mid_layers), 4)
                   for k in ("similitud_misma_familia", "similitud_otra_familia", "rand_ajustado_familias",
                             "info_mutua_norm", "cofenetica_expertos")}  # fmt: skip
        res[mid]["capas_del_medio"] = summary
        print(f"== {mid}: {json.dumps(summary)}", flush=True)
        (Path(out) / "arbol.json").write_text(json.dumps(res, ensure_ascii=False))
        del model
        torch.cuda.empty_cache()
    print("== fin del análisis", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1])
