"""B39: ¿la puerta de certeza suma selectividad sobre EWC sin perder la protección del resto? Prerregistro en el ticket 18 de mf-moe-generalidad.
EWC de B38 (λ = LAM, misma Fisher y corte) con los pesos por hecho de B27 (oráculo, azar) y de B27c (sonda), en los conjuntos 1 a SETS − 1.
Las referencias (comun, ewc, puertas sin EWC) salen de B25, B38, B27 y B27c y no corren acá.
Uso: python -m colab.b39_combinada main SALIDA.json BANCO.json B25_IDA_D40.json CANDIDATOS.json PESOS.json
"""

import json
import os
import random
import sys

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24
from colab import b38_capacidad as b38

LAM, ARMS = 1.0, ("oraculo", "sonda", "azar")


def main(path: str, bank_path: str, b25_path: str, cands_path: str, weights_path: str, model=None, tok=None) -> None:
    facts, B, cands, W = (json.load(open(p, encoding="utf-8")) for p in (bank_path, b25_path, cands_path, weights_path))
    B = B["corridas"]
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, _ = b24.losses(model, tok, facts)
    cands = b38.outside(cands, facts)
    res["config"] = {"n_hechos": len(facts), "lambda": LAM, "fisher_top": b38.FISHER_TOP, "fisher_hechos": len(cands), "dosis": list(b38.DOSES),
                     "brazos": [f"ewc+{a}" for a in ARMS], "lr": b17.LR}  # fmt: skip
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist()}
    state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}
    sparse = b38.fisher(model, tok, [b24.fwd(x) for x in cands])  # antes de los ganchos: la Fisher no lleva penalidad
    res["fisher"] = {"entradas": int(sum(len(v[0]) for v in sparse.values()))}
    hooks = b38.ewc_hooks(model, sparse, LAM)  # todos los brazos usan EWC
    n = len(facts)
    n_over = n // b24.SETS
    order = random.Random("b25:conjuntos").sample(range(n), n)  # los conjuntos de b24.stage2 con el tag «b25»
    for s in range(1, b24.SETS):
        over = sorted(order[s * n_over : (s + 1) * n_over])
        assert B[f"s{s}|ida|d40"]["sobrescritos"] == over, "los conjuntos no coinciden con B25"
        new = [b24.fwd(facts[i], facts[i]["valor_nuevo"]) for i in over]
        for arm in ARMS:
            name = f"ewc+{arm}"
            if f"s{s}|{name}|d{max(b38.DOSES)}" in res["corridas"]:
                continue

            def hook(step: int) -> None:
                if step in b38.DOSES:
                    L_i, _ = b24.losses(model, tok, facts)
                    r = res["corridas"][f"s{s}|{name}|d{step}"] = {"conjunto": s, "brazo": name, "sobrescritos": over, "pasos": step,
                                                                   "subida_ida": (L_i[:, 0] - base_i[:, 0]).round(4).tolist(),
                                                                   "perdida_nuevo": b17.def_losses(model, tok, new).round(4).tolist()}  # fmt: skip
                    print(json.dumps({"corrida": f"s{s}|{name}|d{step}", "resto": round(b38.rest_rise(r), 3), "nuevos": round(float(np.mean(r["perdida_nuevo"])), 3)}), flush=True)

            torch.manual_seed(1000 + s)
            b24.train(model, tok, new, float("-inf"), max(b38.DOSES), b17.CHECK_EVERY, f"b25:lotes:{s}:ida", hook, [float(W[arm][i]) for i in over])
            model.load_state_dict(state, strict=False)
            json.dump(res, open(path, "w"), ensure_ascii=False)
    for h in hooks:
        h.remove()
    print(json.dumps({"FIN": len(res["corridas"])}), flush=True)


if __name__ == "__main__":
    {"main": main}[sys.argv[1]](*sys.argv[2:])
