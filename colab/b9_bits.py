"""B9b (mf-moe-5tq.15): un MoE chico desde cero sobre las tareas de bits de B9a. ¿Los expertos son las primitivas?

Modelo de la fase 1 (`model.build`): 16 expertos, top-2, d = 64, 4 capas, en CPU. Tareas con nombre («rot+not:
bits =»): sin el nombre, la tarea es ambigua. El ruteo se mide solo en las posiciones de la respuesta: en el
nombre, el ruteo difiere por construcción.
- P1, información mutua experto × primitiva: I(experto elegido; primitiva) / H(primitiva), por capa, sobre las 5
  primitivas solas. Nulos: las etiquetas de primitiva barajadas entre tokens, y el mismo modelo sin entrenar (la
  entrada de la respuesta ya difiere por tarea: en la prueba corta, sin entrenar da 0,02-0,06 contra 0,002).
- P2, composición: ¿«a+b» rutea como la mezcla del ruteo de «a» y de «b»? Fracción de los otros pares «c+d» cuya
  mezcla queda más lejos (JSD) que la de «a+b»: 0,5 = azar, 1 = la composición usa los expertos de sus partes.
- P3, en fila: primero las 5 primitivas, después los 10 pares sin repetición. Olvido de las primitivas, y pasos
  hasta dominar los pares contra un modelo que aprende los pares desde cero (reuso).
- P4, control positivo: un modelo con las 4 variantes recursivas «rec:p» y otro con esas 4 primitivas solas. Si
  `tree_likeness` y `composition` no los separan con datos autosimilares, esas medidas no prueban nada.
Uso: .venv/bin/python -m colab.b9_bits SALIDA.json [pasos máximos por etapa]
"""

import json
import random
import sys
import time

import numpy as np
import torch

from mf_moe.bits import PRIMS, RECURSIVE, make_bits_example
from mf_moe.blocks import composition, neuron_signatures, tree_likeness
from mf_moe.model import build, routers
from mf_moe.train import MASTERY, adamw, encode, scores

SIZE = {"top_k": 2, "d": 64, "layers": 4, "ctx": 64}
N_EXPERTS, BATCH, EVAL_EVERY, EVAL_N = 16, 128, 500, 300
# Sin rev: con 6 primitivas y 12 pares, las 6 tareas con rev seguían en cero a los 10.000 pasos (2026-10-31), y
# sola la aprende en 3000. Con las 6 primitivas y los 30 pares, la media era 0,34 a los 8000 pasos.
SINGLES = [p for p in PRIMS if p != "rev"]
# 10 pares: cada primitiva va dos veces primera y dos segunda; ninguno es otro par con la misma función.
PAIRS = [f"{SINGLES[i]}+{SINGLES[(i + j) % 5]}" for j in (1, 2) for i in range(5)]
RECS = [f"rec:{p}" for p in RECURSIVE if p != "rev"]


def eval_sets(tasks: list[str]) -> dict:
    return {t: [make_bits_example(t, random.Random(f"eval:{t}:{i}"), named=True) for i in range(EVAL_N)] for t in tasks}


def train(model, opt, tasks: list[str], evals: dict, max_steps: int, rng: random.Random, tag: str) -> dict:
    """Lotes parejos entre `tasks` hasta que todas lleguen al 88 % o se acaben los pasos."""
    held = {e.prompt for exs in evals.values() for e in exs}
    hist, t0 = [], time.time()
    for step in range(1, max_steps + 1):
        exs = []
        while len(exs) < BATCH:
            ex = make_bits_example(rng.choice(tasks), rng, named=True)
            if ex.prompt not in held:
                exs.append(ex)
        ids, labels = encode(exs, "cpu")
        model(input_ids=ids, labels=labels).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % EVAL_EVERY == 0:
            accs = {t: round(scores(model, evals[t], "cpu")[0], 3) for t in tasks}
            hist.append({"paso": step, "peor": min(accs.values()), "media": round(float(np.mean(list(accs.values()))), 3),
                         "en_cero": sorted(t for t, a in accs.items() if a < 0.05)})  # fmt: skip
            print(json.dumps({"etapa": tag, **hist[-1], "segundos": round(time.time() - t0)}), flush=True)
            if min(accs.values()) >= MASTERY:
                break
    return {"pasos": step, "domino": hist[-1]["peor"] >= MASTERY, "historia": hist, "exactitud": accs}


@torch.no_grad()
def routing(model, evals: dict) -> dict:
    """Por tarea: elecciones top-k [tokens, capas, k] y estados que entran al router [capas, tokens, d], solo en
    las posiciones de la respuesta."""
    model.eval()
    rs = routers(model)
    for r in rs:
        r.record = True
    out = {}
    for t, exs in evals.items():
        ids, labels = encode(exs, "cpu")
        model(input_ids=ids)
        mask = (labels != -100).reshape(-1)
        out[t] = {"idx": np.stack([r.last[2][mask].numpy() for r in rs], 1),
                  "h": np.stack([r.last[0][mask].numpy() for r in rs])}  # fmt: skip
    for r in rs:
        r.record = False
    model.train()
    return out


def _mi(lab: np.ndarray, exp: np.ndarray, n_lab: int) -> float:
    """I(experto; etiqueta) / H(etiqueta), con cada una de las k elecciones de un token como una observación."""
    lab = np.repeat(lab, exp.shape[1])
    joint = np.zeros((n_lab, N_EXPERTS))
    np.add.at(joint, (lab, exp.reshape(-1)), 1)
    joint /= joint.sum()
    pl, pe = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pl @ pe)[nz])).sum() / -(pl[pl > 0] * np.log(pl[pl > 0])).sum())


def mutual_info(rt: dict, tasks: list[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lab = np.concatenate([np.full(len(rt[t]["idx"]), i) for i, t in enumerate(tasks)])
    idx = np.concatenate([rt[t]["idx"] for t in tasks])
    real = [_mi(lab, idx[:, L], len(tasks)) for L in range(idx.shape[1])]
    null = [np.mean([_mi(rng.permutation(lab), idx[:, L], len(tasks)) for _ in range(10)]) for L in range(idx.shape[1])]
    return {"real": np.round(real, 4).tolist(), "nulo": np.round(null, 4).tolist()}


def _dist(idx: np.ndarray) -> np.ndarray:
    """[capas, expertos]: uso de cada experto por capa."""
    c = np.stack([np.bincount(idx[:, L].reshape(-1), minlength=N_EXPERTS) for L in range(idx.shape[1])]).astype(float)
    return c / c.sum(1, keepdims=True)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))
    return (kl(p) + kl(q)) / 2


def compose_score(rt: dict) -> dict:
    """P2. Por capa: fracción de pares ajenos cuya mezcla queda más lejos de «a+b» que la mezcla de «a» y «b».
    Además, en qué capas «a+b» se parece más a «a» (la primera en aplicarse) que a «b»."""
    P = {t: _dist(rt[t]["idx"]) for t in rt}
    n_layers = next(iter(P.values())).shape[0]
    rank, first = np.zeros((len(PAIRS), n_layers)), np.zeros((len(PAIRS), n_layers))
    for i, ab in enumerate(PAIRS):
        a, b = ab.split("+")
        for L in range(n_layers):
            own = _jsd(P[ab][L], (P[a][L] + P[b][L]) / 2)
            others = [_jsd(P[ab][L], (P[c][L] + P[d][L]) / 2) for cd in PAIRS if not set(cd.split("+")) & {a, b}
                      for c, d in [cd.split("+")]]  # fmt: skip
            rank[i, L] = np.mean([o > own for o in others])
            first[i, L] = _jsd(P[ab][L], P[b][L]) - _jsd(P[ab][L], P[a][L])
    return {"rango_por_capa": rank.mean(0).round(3).tolist(), "rango_medio": round(float(rank.mean()), 3),
            "mas_cerca_de_la_primera_por_capa": first.mean(0).round(4).tolist()}  # fmt: skip


def structure(model, rt: dict) -> dict:
    """P4. Por capa: árbol de los estados de la respuesta y árbol de composición de las neuronas de los expertos."""
    out = {}
    for L, layer in enumerate(model.model.layers):
        H = np.concatenate([rt[t]["h"][L] for t in rt])
        H = H[np.random.default_rng(L).choice(len(H), min(4000, len(H)), replace=False)]
        e = layer.mlp.experts
        Xw, lab = neuron_signatures(e.gate_up_proj.detach().numpy(), e.down_proj.detach().numpy())
        comp = composition(Xw, lab)
        out[L] = {"arbol_estados": tree_likeness(H), "cv_pendientes": comp["real"]["cv_pendientes"],
                  "cv_pendientes_nulo": comp["nulo_gaussiano"]["cv_pendientes"]}  # fmt: skip
    return out


def fresh(seed: int):
    model = build(N_EXPERTS, seed=seed, **SIZE)
    return model, adamw(model)


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "tamano": SIZE, "expertos": N_EXPERTS, "pasos_maximos": max_steps}
    save = lambda: json.dump(res, open(path, "w"), ensure_ascii=False)
    ev = eval_sets(SINGLES + PAIRS + RECS)
    pick = lambda ts: {t: ev[t] for t in ts}

    # P1 y P2: primitivas y pares juntos
    model, opt = fresh(seed)
    rt0 = routing(model, pick(SINGLES + PAIRS))  # línea base: sin entrenar, el ruteo ya difiere por la entrada
    base = {"info_mutua": mutual_info(rt0, SINGLES), "composicion": compose_score(rt0)}
    res["conjunto"] = train(model, opt, SINGLES + PAIRS, pick(SINGLES + PAIRS), max_steps, random.Random(seed), "conjunto")
    rt = routing(model, pick(SINGLES + PAIRS))
    res["conjunto"] |= {"info_mutua": mutual_info(rt, SINGLES), "composicion": compose_score(rt), "sin_entrenar": base}
    save()

    # P3: en fila, contra los pares desde cero
    model, opt = fresh(seed)
    res["fila_primitivas"] = train(model, opt, SINGLES, pick(SINGLES), max_steps, random.Random(seed + 1), "fila:1")
    res["fila_primitivas"]["info_mutua"] = mutual_info(routing(model, pick(SINGLES)), SINGLES)
    res["fila_pares"] = train(model, opt, PAIRS, pick(PAIRS), max_steps, random.Random(seed + 2), "fila:2")
    res["fila_pares"]["primitivas_despues"] = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in SINGLES}
    model, opt = fresh(seed)
    res["pares_desde_cero"] = train(model, opt, PAIRS, pick(PAIRS), max_steps, random.Random(seed + 2), "cero")
    save()

    # P4: control positivo de autosimilitud
    for name, tasks in (("recursivas", RECS), ("planas", [p for p in RECURSIVE if p != "rev"])):
        model, opt = fresh(seed)
        base = structure(model, routing(model, pick(tasks)))
        res[name] = train(model, opt, tasks, pick(tasks), max_steps, random.Random(seed + 3), name)
        res[name] |= {"estructura": structure(model, routing(model, pick(tasks))), "estructura_sin_entrenar": base}
        save()
    res["segundos"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
