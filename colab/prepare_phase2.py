"""Prepara los datos de la fase 2 en una VM (ticket 12): tokeniza una sola vez los cinco datasets.

Cada dominio queda en `<out>/<dominio>_eval.npy` (1.000 secuencias) y `<out>/<dominio>_train.npy`, uint16,
secuencias de 1.024 tokens con EOS entre documentos. Cada documento va entero a la evaluación o al entrenamiento,
así que la evaluación nunca comparte documentos con el entrenamiento. Uso: python prepare_phase2.py OUT [tokens_por_dominio]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from mf_moe.granite import MODEL_ID

SEQ, EVAL_SEQS, EVAL_PER_DOC = 1024, 1000, 8  # tope por documento: la evaluación cubre >= 125 documentos
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
        need = tokens // SEQ if dom != "general" else 10_000
        every = max(2, (need + EVAL_SEQS) // EVAL_SEQS)  # uno de cada `every` documentos va a la evaluación
        side = {"eval": ([], [], EVAL_SEQS), "train": ([], [], need)}  # (buffer, secuencias, meta)
        docs = 0
        for row in load_dataset(name, cfg, split="train", streaming=True):
            ids = tok(text_of(row))["input_ids"] + [tok.eos_token_id]
            # cada documento va entero a un solo lado y la evaluación cubre todo el stream (piloto 4: las primeras
            # 1.000 secuencias de código venían de ~20 documentos y daban 0,18 nats menos que el entrenamiento)
            ev_open = len(side["eval"][1]) < EVAL_SEQS
            to_eval = ev_open and (docs % every == 0 or len(side["train"][1]) >= need)
            docs += 1
            buf, seqs, goal = side["eval" if to_eval else "train"]
            buf += ids[: EVAL_PER_DOC * SEQ] if to_eval else ids
            while len(buf) >= SEQ and len(seqs) < goal:
                seqs.append(buf[:SEQ])
                del buf[:SEQ]
            if len(side["eval"][1]) >= EVAL_SEQS and len(side["train"][1]) >= need:
                break
        seqs = side["eval"][1] + side["train"][1]
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
