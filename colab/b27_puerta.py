"""B27: ¿pesar la pérdida de cada ejemplo por su duda real crea el nivel 2 (certeza contra plasticidad) en Granite?
Prerregistro en el ticket 09 (mf-moe-generalidad). La puerta oráculo usa la pérdida base de la ida: es el techo de cualquier
señal real. Banco, conjuntos, semillas de lotes y dosis son los de B25; solo el brazo ida.
- oraculo: w = 0,25 + 1,5 · rango(duda) / (n − 1), media 1 sobre el banco.
- azar: los mismos pesos, permutados entre hechos.
- uniforme: sin pesos, solo el conjunto 0, para chequear la réplica contra B25.
Uso: python -m colab.b27_puerta SALIDA.json BANCO.json
"""

import json
import os
import random
import sys

import numpy as np

from colab import b23_uniedit as b23
from colab import b24_inversion as b24

LOW, SPAN = 0.25, 1.5


def gate(doubt: np.ndarray) -> np.ndarray:
    rank = np.argsort(np.argsort(doubt))
    return LOW + SPAN * rank / (len(doubt) - 1)


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


if __name__ == "__main__":
    main(*sys.argv[1:3])
