"""B25: B24 con hechos reales de preentrenamiento. ¿La vuelta sobrevive a la sobrescritura de la ida cuando las dos direcciones
vienen del preentrenamiento y no de un ajuste fino? Diseño preregistrado en el ticket 08 (mf-moe-generalidad).

- candidatos: une el conteo inverso de B23 con sus hechos y guarda los que pasan F0 en las dos direcciones y no son piso,
  uno por valor (un valor compartido haría que sobrescribir un hecho toque la vuelta de otro).
- main: refiltra en GPU con donantes nuevos (K_FILTER por dirección), mide con otros K donantes como B24 y corre la etapa 2
  de B24 en modo dosis, sin etapa 1: el punto de partida es Granite tal como viene.
Uso: python -m colab.b25_reales candidatos INVERSOS.json HECHOS.json SALIDA.json
     python -m colab.b25_reales main SALIDA.json CANDIDATOS.json HECHOS.json
"""

import collections as C
import json
import os
import random
import sys

import numpy as np

from colab import b23_uniedit as b23
from colab import b24_inversion as b24

K_FILTER, MIN_BANK, GATE0 = 9, 128, 0.8  # con 9 donantes por dirección, un hecho desconocido pasa las dos por azar con p ≈ 0,01


def candidatos(inv_path: str, hechos_path: str, out: str) -> None:
    inv = json.load(open(inv_path, encoding="utf-8"))["tipos"]
    val = {(x["cabeza"], x["propiedad"]): x["valor"] for v in json.load(open(hechos_path, encoding="utf-8"))["hechos"].values() for x in v}
    rows, seen = [], set()
    for t in sorted(inv):
        for r in inv[t]["hechos"]:
            v = val[r["cabeza"], r["propiedad"]]
            if r["inverso"] and not r["piso"] and v.lower() not in seen:
                seen.add(v.lower())
                rows.append({"tipo": t, "propiedad": r["propiedad"], "cabeza": r["cabeza"], "valor": v})
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=0)
    print(json.dumps({"candidatos": len(rows)}))


def donors(x: dict, pairs: set[tuple[str, str]], k: int, seed: str, skip: tuple[set, set] = (set(), set())) -> tuple[list, list] | None:
    """k valores de igual propiedad y otro valor (ida) y k cabezas de igual propiedad y otro valor (vuelta), sin los de `skip`."""
    other = sorted((h, v) for h, v in pairs if v.lower() != x["valor"].lower() and h != x["cabeza"])
    vs, hs = sorted({v for _, v in other} - skip[0]), sorted({h for h, _ in other} - skip[1])
    if min(len(vs), len(hs)) < k:
        return None
    rng = random.Random(seed)
    return rng.sample(vs, k), rng.sample(hs, k)


def strict(model, tok, cands: list[dict]) -> np.ndarray:
    """Gana en la ida y en la vuelta contra los donantes del filtro."""
    L_i, L_v = b24.losses(model, tok, cands)
    return b23.wins(L_i) & b23.wins(L_v)


def main(path: str, cand_path: str, hechos_path: str, model=None, tok=None) -> None:
    cands = json.load(open(cand_path, encoding="utf-8"))
    pairs = C.defaultdict(set)
    for v in json.load(open(hechos_path, encoding="utf-8"))["hechos"].values():
        for x in v:
            pairs[x["propiedad"]].add((x["cabeza"], x["valor"]))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    res["config"] = {"k_filtro": K_FILTER, "k": b24.K, "min_banco": MIN_BANK, "puerta0": GATE0, "conjuntos": b24.SETS, "dosis": list(b24.DOSES),
                     "lr": b24.b17.LR, "batch": b24.b17.BATCH, "stop2": b24.b17.STOP_LOSS, "candidatos": len(cands)}  # fmt: skip
    model, tok = b23.load(model, tok, True)
    pool = []
    for x in cands:
        f = donors(x, pairs[x["propiedad"]], K_FILTER, f"b25:filtro:{x['cabeza']}:{x['propiedad']}")
        m = f and donors(x, pairs[x["propiedad"]], b24.K, f"b25:medida:{x['cabeza']}:{x['propiedad']}", (set(f[0]), set(f[1])))
        if m:  # los donantes de la medida no repiten los del filtro: la selección no infla los aciertos de la línea base
            pool.append((x | {"donantes": f[0], "donantes_inv": f[1]}, m))
    keep = strict(model, tok, [x for x, _ in pool]) if pool else np.zeros(0, bool)
    facts = [x | {"donantes": m[0], "donantes_inv": m[1]} for (x, m), k in zip(pool, keep) if k]
    taken = {s for P in pairs.values() for hv in P for s in hv}
    rng = random.Random("b25:nombres")
    for x, nv, nh in zip(facts, b24.names(len(facts), rng, taken), b24.names(len(facts), rng, taken)):
        x |= {"valor_nuevo": nv, "cabeza_nueva": nh}
    res["filtro"] = {"con_donantes": len(pool), "pasan": len(facts), "por_tipo": dict(C.Counter(x["tipo"] for x in facts))}
    res["banco"] = facts
    print(json.dumps({"filtro": res["filtro"]}), flush=True)
    if len(facts) < MIN_BANK:
        res["corte"] = f"F: el filtro estricto deja {len(facts)} hechos (< {MIN_BANK})"
    else:
        base_i, base_v = b24.losses(model, tok, facts)
        win = {"ida": float(b23.wins(base_i).mean()), "vuelta": float(b23.wins(base_v).mean())}
        res["base"] = {"gana": win, "perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
        print(json.dumps({"base": win}), flush=True)
        res["corte"] = f"E0: la línea base no gana con los donantes de la medida (gana {win})" if min(win.values()) < GATE0 else None
    if res["corte"]:
        json.dump(res, open(path, "w"), ensure_ascii=False)
        print(json.dumps({"corte": res["corte"]}), flush=True)
        return
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25")


if __name__ == "__main__":
    {"candidatos": candidatos, "main": main}[sys.argv[1]](*sys.argv[2:])
