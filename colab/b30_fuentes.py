"""B30: ¿una voz repetida contra fuentes independientes, con puerta de pesos o con umbral de fuentes, reproduce la conducta del video?
Prerregistro en el ticket 12 (mf-moe-generalidad). Banco, conjuntos (tag «b25»), semillas y pesos fuertes de B27b (0,02 a 1,98).
- una: cada hecho se entrena con una sola frase, la de B27 (T0). El brazo una|peso repite el oraculo_fuerte de B27b.
- fuentes: cada exposición elige una de 10 frases distintas; la tasa de exposición por hecho es la misma que en «una».
- peso: el peso del hecho, siempre. umbral: el peso solo cuando el hecho ya vio tau frases distintas; antes, 0.
  tau es 1 en la mitad dudosa del banco y sube hasta 8 en el hecho más seguro.
La creencia se mide en T0 y en dos frases que ningún brazo entrena (H1 y H2): margen = pérdida del valor nuevo − la del viejo.
Uso: python -m colab.b30_fuentes SALIDA.json BANCO.json
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
from colab import b27_puerta as b27

TRAIN_T = ("The {p} of {h} is", "As for {h}, its {p} is", "In the records, the {p} of {h} is listed as", "Fact: the {p} of {h} is",
           "Reports state that the {p} of {h} is", "According to one source, the {p} of {h} is", "Regarding {h}: the {p} is",
           "It is documented that the {p} of {h} is", "{h} has as its {p}", "Sources confirm the {p} of {h}:")  # fmt: skip
HELD_T = ("{h}'s {p} is", "Q: What is the {p} of {h}? A:")
DOSES, TAU_MAX = (10, 40), 8
ARMS = (("una", "peso"), ("fuentes", "peso"), ("una", "umbral"), ("fuentes", "umbral"))


def frame(t: str, x: dict, value: str) -> tuple[str, str]:
    return t.format(p=x["propiedad"], h=x["cabeza"]), f" {value}."


def thresholds(base_loss: np.ndarray) -> np.ndarray:
    """tau por hecho: 1 en la mitad dudosa; en la mitad segura sube con la certeza hasta TAU_MAX."""
    c = np.argsort(np.argsort(-base_loss)) / (len(base_loss) - 1)  # 1 = el más seguro
    return 1 + np.round((TAU_MAX - 1) * np.clip((c - 0.5) / 0.5, 0, None)).astype(int)


def margins(model, tok, facts: list[dict], templates: tuple) -> np.ndarray:
    """[plantilla, hecho]: pérdida del valor nuevo menos la del viejo. Positivo es que lo viejo resiste."""
    rows = [frame(t, x, v) for t in templates for x in facts for v in (x["valor_nuevo"], x["valor"])]
    L = b17.def_losses(model, tok, rows).reshape(len(templates), len(facts), 2)
    return L[..., 0] - L[..., 1]


def schedule(n_items: int, steps: int, key: str) -> list[list[int]]:
    """Los lotes que b24.train va a sacar con la misma clave: el umbral sabe qué frases ve cada hecho antes de cada paso."""
    rng = random.Random(key)
    return [rng.sample(range(n_items), min(b17.BATCH, n_items)) for _ in range(steps)]


def run_arm(model, tok, facts: list[dict], over: list[int], w: np.ndarray, tau: np.ndarray, base: np.ndarray, source: str, gate: str,
            key: str, out: dict, s: int) -> None:  # fmt: skip
    ks = range(1) if source == "una" else range(len(TRAIN_T))
    owner, tmpl = [j for j, _ in enumerate(over) for _ in ks], [k for _ in over for k in ks]
    items = [frame(TRAIN_T[k], facts[over[j]], facts[over[j]]["valor_nuevo"]) for j, k in zip(owner, tmpl)]
    sched, seen = schedule(len(items), max(DOSES), key), [set() for _ in over]
    weights = [0.0] * len(items)

    def gate_for(step: int) -> None:  # step: índice 0 del paso que viene
        for i in sched[step]:
            seen[owner[i]].add(tmpl[i])
        for i in range(len(items)):
            j = owner[i]
            weights[i] = float(w[over[j]]) if gate == "peso" or len(seen[j]) >= tau[over[j]] else 0.0

    def hook(step: int) -> None:
        if step in DOSES:
            m = margins(model, tok, [facts[i] for i in over], (TRAIN_T[0], *HELD_T))
            rise = b17.def_losses(model, tok, [b24.fwd(x) for x in facts]) - base
            out[f"s{s}|{source}|{gate}|d{step}"] = {
                "conjunto": s, "fuente": source, "puerta": gate, "sobrescritos": over, "pasos": step,
                "margen": {n: m[t].round(4).tolist() for t, n in enumerate(("T0", "H1", "H2"))},
                "subida_ida": rise.round(4).tolist(), "frases_vistas": [len(x) for x in seen],
                "abiertos": [int(gate == "peso" or len(seen[j]) >= tau[i]) for j, i in enumerate(over)]}  # fmt: skip
        if step < max(DOSES):
            gate_for(step)

    gate_for(0)
    torch.manual_seed(1000 + s)
    b24.train(model, tok, items, float("-inf"), max(DOSES), b17.CHECK_EVERY, key, hook, weights)


def main(path: str, bank_path: str, model=None, tok=None) -> None:
    facts = json.load(open(bank_path, encoding="utf-8"))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, _ = b24.losses(model, tok, facts)
    base = b17.def_losses(model, tok, [b24.fwd(x) for x in facts])  # igual a base_i[:, 0]; la medida usa la misma llamada
    w, tau = b27.gate(base_i[:, 0], 0.02, 1.96), thresholds(base_i[:, 0])
    m0 = margins(model, tok, facts, (*TRAIN_T, *HELD_T))
    res["config"] = {"n_hechos": len(facts), "conjuntos": b24.SETS, "dosis": list(DOSES), "plantillas": list(TRAIN_T), "fuera": list(HELD_T),
                     "tau_max": TAU_MAX, "fuerte": [0.02, 1.96], "lr": b17.LR, "batch": b17.BATCH}  # fmt: skip
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "tau": tau.tolist(), "pesos": w.round(4).tolist(),
                   "margen": m0.round(4).tolist()}  # fmt: skip
    print(json.dumps({"base_resiste_por_plantilla": (m0 > 0).mean(1).round(3).tolist()}), flush=True)
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    n = len(facts)
    n_over = n // b24.SETS
    order = random.Random("b25:conjuntos").sample(range(n), n)  # los conjuntos de B27 y B27b
    for s in range(b24.SETS):
        over = sorted(order[s * n_over : (s + 1) * n_over])
        for source, gate in ARMS:
            if f"s{s}|{source}|{gate}|d{max(DOSES)}" in res["corridas"]:
                continue
            run_arm(model, tok, facts, over, w, tau, base, source, gate, f"b25:lotes:{s}:ida", res["corridas"], s)
            model.load_state_dict(state, strict=False)
            json.dump(res, open(path, "w"), ensure_ascii=False)
            r = res["corridas"][f"s{s}|{source}|{gate}|d{max(DOSES)}"]
            print(json.dumps({"corrida": f"s{s}|{source}|{gate}", "resiste": {k: round(float(np.mean(np.array(v) > 0)), 3) for k, v in r["margen"].items()}}), flush=True)
    print(json.dumps({"FIN": len(res["corridas"])}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
