"""B29: ¿la puerta de B27 y B27c crea el nivel 2 también en otro banco? Prerregistro en el ticket 11 (mf-moe-generalidad).
El banco nuevo sale de los hechos de UniEdit fuera de los 461 candidatos de B25, filtrados solo por la ida: el banco de B27 exigía
las dos direcciones, así que este cubre hechos más dudosos. Brazos oráculo, azar y sonda, solo en la dosis 10.
Uso: python -m colab.b29_replica banco HECHOS.json CANDIDATOS.json SALIDA.json   (CPU: filtro y armado del banco)
     python -m colab.b29_replica main SALIDA.json BANCO.json CANDIDATOS.json      (GPU)
"""

import collections as C
import json
import os
import random
import sys

import numpy as np

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab import b24_inversion as b24
from colab import b25_reales as b25
from colab import b27_puerta as b27

N_SCREEN, N_BANK, DOSES = 1200, 256, (10,)


def banco(hechos_path: str, cand_path: str, out: str, model=None, tok=None) -> None:
    key = lambda x: (x["cabeza"], x["propiedad"])
    cands = json.load(open(cand_path, encoding="utf-8"))
    hechos = json.load(open(hechos_path, encoding="utf-8"))["hechos"]
    pairs = C.defaultdict(set)
    for v in hechos.values():
        for x in v:
            pairs[x["propiedad"]].add((x["cabeza"], x["valor"]))
    seen_k, seen_v = {key(x) for x in cands}, {x["valor"].lower() for x in cands}
    rows = []
    for t in sorted(hechos):
        for x in hechos[t]:
            if key(x) not in seen_k and x["valor"].lower() not in seen_v:  # uno por valor, y ninguno compartido con B25
                seen_v.add(x["valor"].lower())
                rows.append({"tipo": t, "propiedad": x["propiedad"], "cabeza": x["cabeza"], "valor": x["valor"]})
    rows = random.Random("b29:filtro").sample(rows, min(N_SCREEN, len(rows)))
    pool = []
    for x in rows:
        f = b25.donors(x, pairs[x["propiedad"]], b25.K_FILTER, f"b29:filtro:{x['cabeza']}:{x['propiedad']}")
        m = f and b25.donors(x, pairs[x["propiedad"]], b24.K, f"b29:medida:{x['cabeza']}:{x['propiedad']}", (set(f[0]), set(f[1])))
        if m:  # los donantes de la medida no repiten los del filtro, como B25
            pool.append((x, f, m))
    model, tok = b23.load(model, tok, False)
    L = b17.def_losses(model, tok, [b24.fwd(x, v) for x, f, _ in pool for v in [x["valor"], *f[0]]]).reshape(len(pool), -1)
    keep = b23.wins(L)
    facts = [x | {"donantes": m[0], "donantes_inv": m[1]} for (x, _, m), k in zip(pool, keep) if k]
    facts = random.Random("b29:banco").sample(facts, min(N_BANK, len(facts)))
    taken = {s for P in pairs.values() for hv in P for s in hv}
    rng = random.Random("b29:nombres")
    for x, nv, nh in zip(facts, b24.names(len(facts), rng, taken), b24.names(len(facts), rng, taken)):
        x |= {"valor_nuevo": nv, "cabeza_nueva": nh}
    json.dump(facts, open(out, "w"), ensure_ascii=False, indent=0)
    print(json.dumps({"filtrados": len(rows), "con_donantes": len(pool), "ganan_ida": int(keep.sum()), "banco": len(facts),
                      "por_tipo": dict(C.Counter(x["tipo"] for x in facts))}), flush=True)  # fmt: skip


def main(path: str, bank_path: str, cand_path: str, model=None, tok=None) -> None:
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    w = b27.gate(base_i[:, 0])
    rand = np.array(random.Random("b29:azar").sample(list(w), len(w)))
    p_bank, pool, _ = b27.probe(model, tok, facts, cands, tag="b29")
    rho = float(np.corrcoef(np.argsort(np.argsort(p_bank)), np.argsort(np.argsort(base_i[:, 0])))[0, 1])
    res["config"] = {"n_hechos": len(facts), "n_sonda": len(pool), "capas": list(b27.LAYERS), "conjuntos": b24.SETS, "dosis": list(DOSES),
                     "puerta": [b27.LOW, b27.SPAN], "lr": b17.LR, "batch": b17.BATCH}  # fmt: skip
    res["sonda"] = {"prediccion": p_bank.round(4).tolist(), "spearman_con_perdida_base": round(rho, 4)}
    res["pesos"] = {"oraculo": w.round(4).tolist(), "azar": rand.round(4).tolist(), "sonda": b27.gate(p_bank).round(4).tolist()}
    res["base"] = {"gana": float(b23.wins(base_i).mean()), "perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"gana_ida": res["base"]["gana"], "sonda_spearman_con_perdida_base": round(rho, 4)}), flush=True)
    arms = {"oraculo": ("ida", w, None), "azar": ("ida", rand, None), "sonda": ("ida", b27.gate(p_bank), None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b29", doses=DOSES, arms=arms)


if __name__ == "__main__":
    {"banco": banco, "main": main}[sys.argv[1]](*sys.argv[2:])
