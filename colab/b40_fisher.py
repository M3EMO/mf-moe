"""B40: ¿el gradiente de sobrescribir un hecho seguro cae más sobre los pesos que EWC frena? Prerregistro en el ticket 19 de mf-moe-generalidad.
Fisher de B38 (b38.fisher). Por hecho del banco: r = Σ F·g² sobre las entradas guardadas / Σ g² sobre todo peso entrenable,
con g el gradiente de la pérdida del valor nuevo (y, como dato, del valor viejo). Sin entrenamiento.
Uso: python -m colab.b40_fisher main SALIDA.json BANCO.json CANDIDATOS.json
"""

import json
import sys

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24
from colab import b38_capacidad as b38
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp


def share(model, tok, item: tuple[str, str], sparse: dict) -> tuple[float, float]:
    """(Σ F·g² en las entradas guardadas / Σ g², Σ g²) para el gradiente de un ítem."""
    ids, mask, labels = b17.batch(tok, [item])
    with amp(DEVICE):
        model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
    num = den = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.float().flatten()
            den += float((g**2).sum())
            if n in sparse:
                idx, f, _ = sparse[n]
                num += float((f * g[idx] ** 2).sum())
    model.zero_grad(set_to_none=True)
    return num / den, den


def main(path: str, bank_path: str, cands_path: str, model=None, tok=None) -> None:
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cands_path))
    model, tok = b23.load(model, tok, True)
    base_i, _ = b24.losses(model, tok, facts)
    cands = b38.outside(cands, facts)
    sparse = b38.fisher(model, tok, [b24.fwd(x) for x in cands])
    new = [share(model, tok, b24.fwd(x, x["valor_nuevo"]), sparse) for x in facts]
    old = [share(model, tok, b24.fwd(x), sparse) for x in facts]
    res = {"config": {"n_hechos": len(facts), "fisher_top": b38.FISHER_TOP, "fisher_hechos": len(cands)},
           "base": {"perdida_ida": base_i[:, 0].round(4).tolist()},
           "fisher": {"entradas": int(sum(len(v[0]) for v in sparse.values()))},
           "hechos": {"r_nuevo": [round(r, 6) for r, _ in new], "r_viejo": [round(r, 6) for r, _ in old], "g2_nuevo": [g for _, g in new]}}  # fmt: skip
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps({"FIN": len(facts), "fisher": res["fisher"]}), flush=True)


if __name__ == "__main__":
    {"main": main}[sys.argv[1]](*sys.argv[2:])
