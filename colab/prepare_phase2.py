"""Prepara los datos de la fase 2 en una VM (ticket 12): tokeniza una sola vez los cinco datasets.

Cada dominio queda en `<out>/<dominio>_eval.npy` (las primeras 1.000 secuencias) y `<out>/<dominio>_train.npy`
(las siguientes), uint16, secuencias de 1.024 tokens con EOS entre documentos. La evaluación nunca aparece
en el entrenamiento. Uso: python prepare_phase2.py OUT [tokens_por_dominio]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from mf_moe.granite import MODEL_ID

SEQ, EVAL_SEQS = 1024, 1000
SOURCES = {  # dominio: (dataset, config); licencias en el ticket 12
    "matematica": ("open-web-math/open-web-math", None),
    "codigo": ("common-pile/stackv2", None),
    "biomedico": ("slinusc/PubMedAbstractsSubset", None),
    "legal": ("common-pile/caselaw_access_project", None),
    "general": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
}


def text_of(row: dict) -> str:
    for k in ("text", "abstract", "content"):
        if isinstance(row.get(k), str):
            return row[k]
    return max((v for v in row.values() if isinstance(v, str)), key=len, default="")


def main(out: str, tokens: int = 20_000_000) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    assert tok.vocab_size < 2**16, tok.vocab_size
    Path(out).mkdir(parents=True, exist_ok=True)
    report = {}
    for dom, (name, cfg) in SOURCES.items():
        need = EVAL_SEQS + (tokens // SEQ if dom != "general" else 10_000)
        buf, seqs, docs = [], [], 0
        for row in load_dataset(name, cfg, split="train", streaming=True):
            buf += tok(text_of(row))["input_ids"] + [tok.eos_token_id]
            docs += 1
            while len(buf) >= SEQ and len(seqs) < need:
                seqs.append(buf[:SEQ])
                buf = buf[SEQ:]
            if len(seqs) >= need:
                break
        arr = np.asarray(seqs, dtype=np.uint16)
        np.save(Path(out, f"{dom}_eval.npy"), arr[:EVAL_SEQS])
        np.save(Path(out, f"{dom}_train.npy"), arr[EVAL_SEQS:])
        # el hash deja comprobar que cada VM tokenizó exactamente los mismos datos
        report[dom] = {"dataset": name, "documentos": docs, "secuencias": len(arr),
                       "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}  # fmt: skip
        print(json.dumps({dom: report[dom]}), flush=True)
    Path(out, "datos.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
