"""B27: ¿pesar la pérdida de cada ejemplo por su duda real crea el nivel 2 (certeza contra plasticidad) en Granite?
Prerregistro en el ticket 09 (mf-moe-generalidad). La puerta oráculo usa la pérdida base de la ida: es el techo de cualquier
señal real. Banco, conjuntos, semillas de lotes y dosis son los de B25; solo el brazo ida.
- oraculo: w = 0,25 + 1,5 · rango(duda) / (n − 1), media 1 sobre el banco.
- azar: los mismos pesos, permutados entre hechos.
- uniforme: sin pesos, solo el conjunto 0, para chequear la réplica contra B25.
B27b (main_b): la señal sin etiqueta (probabilidad máxima del primer token, solo con la pregunta) y el oráculo fuerte (pesos 0,02 a 1,98).
Uso: python -m colab.b27_puerta SALIDA.json BANCO.json
     python -m colab.b27_puerta b SALIDA.json BANCO.json
"""

import json
import os
import random
import sys

import numpy as np
import torch

from colab import b23_uniedit as b23
from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

LOW, SPAN = 0.25, 1.5


def gate(doubt: np.ndarray, low: float = LOW, span: float = SPAN) -> np.ndarray:
    rank = np.argsort(np.argsort(doubt))
    return low + span * rank / (len(doubt) - 1)


@torch.no_grad()
def own_confidence(model, tok, facts: list[dict]) -> np.ndarray:
    """Probabilidad máxima del siguiente token al final de la pregunta de la ida: no conoce ningún valor."""
    model.eval()
    out = []
    for i in range(0, len(facts), b17.EVAL_BATCH):
        rows = [tok(b24.fwd(x)[0])["input_ids"] for x in facts[i : i + b17.EVAL_BATCH]]
        L = max(map(len, rows))
        ids = torch.tensor([r + [0] * (L - len(r)) for r in rows], device=DEVICE)
        mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows], device=DEVICE)
        with amp(DEVICE):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        last = logits[torch.arange(len(rows)), torch.tensor([len(r) - 1 for r in rows])]
        out += torch.softmax(last, -1).max(-1).values.tolist()
    model.train()
    return np.array(out)


def main(path: str, bank_path: str, model=None, tok=None) -> None:
    facts = json.load(open(bank_path, encoding="utf-8"))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    w = gate(base_i[:, 0])
    rand = np.array(random.Random("b27:azar").sample(list(w), len(w)))
    res["config"] = {"n_hechos": len(facts), "conjuntos": b24.SETS, "dosis": list(b24.DOSES), "peso_min": LOW, "peso_rango": SPAN,
                     "lr": b24.b17.LR, "batch": b24.b17.BATCH}  # fmt: skip
    res["banco"], res["pesos"] = facts, {"oraculo": w.round(4).tolist(), "azar": rand.round(4).tolist()}
    res["base"] = {"gana": {"ida": float(b23.wins(base_i).mean()), "vuelta": float(b23.wins(base_v).mean())},
                   "perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}  # fmt: skip
    print(json.dumps({"base": res["base"]["gana"]}), flush=True)
    arms = {"uniforme": ("ida", None, {0}), "oraculo": ("ida", w, None), "azar": ("ida", rand, None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


def main_b(path: str, bank_path: str, model=None, tok=None) -> None:
    facts = json.load(open(bank_path, encoding="utf-8"))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    conf = own_confidence(model, tok, facts)
    w_sig, w_strong = gate(-conf), gate(base_i[:, 0], 0.02, 1.96)
    rho = float(np.corrcoef(np.argsort(np.argsort(conf)), np.argsort(np.argsort(-base_i[:, 0])))[0, 1])
    res["config"] = {"n_hechos": len(facts), "conjuntos": b24.SETS, "dosis": list(b24.DOSES), "senal": [LOW, SPAN], "fuerte": [0.02, 1.96],
                     "lr": b24.b17.LR, "batch": b24.b17.BATCH}  # fmt: skip
    res["senal"] = {"confianza": conf.round(5).tolist(), "spearman_con_certeza": round(rho, 4)}
    res["pesos"] = {"senal": w_sig.round(4).tolist(), "oraculo_fuerte": w_strong.round(4).tolist()}
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"senal_spearman_con_certeza": res["senal"]["spearman_con_certeza"]}), flush=True)
    arms = {"senal": ("ida", w_sig, None), "oraculo_fuerte": ("ida", w_strong, None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


if __name__ == "__main__":
    main_b(*sys.argv[2:4]) if sys.argv[1] == "b" else main(*sys.argv[1:3])
