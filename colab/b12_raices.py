"""B12 (mf-moe-5tq.24): ¿un experto nuevo necesita raíces? Idea del usuario; la auditoría de novedad con Lacuna
(vault, 2026-11-02) no encontró trabajos que hagan nacer el experto de f∘g desde los expertos de f y g.

Tareas de B10b: 5 primitivas, 10 pares entrenados, 5 pares no vistos sin equivalentes funcionales.
Etapa 1: primitivas con 16 expertos, una vez por semilla; cada brazo parte de una copia de ese estado.
Etapa 2: antes de entrenar los pares, en cada capa nace un hijo por par. Todo token de un par usa a su hijo como
uno de sus k expertos (ruteo forzado por el nombre de la tarea): sin forzar, el hijo recibía 5-19 % del ruteo de
su par al nacer y menos al entrenar, con cualquier fila del router. Así solo cambia de dónde salen sus pesos.
Su fila del router es la media de los estados del par menos la media de todas las tareas. El hijo nace mudo (sesgo
-1): sin forzar, las primitivas lo elegían en el 41 % de sus tokens y un hijo al azar las dañaba al nacer.
- raiz: copia del experto más usado por f (la división de siempre);
- interp: promedio de los expertos más usados por f y por g;
- azar: pesos al azar con la escala de la capa;
- padres_azar: promedio de los expertos de dos primitivas ajenas al par (control de topología);
- fijo: 26 expertos desde el principio, sin crecer (mismos parámetros al final).
El hijo hereda los momentos de AdamW del experto de f en todos los brazos que crecen.
B12b (mf-moe-5tq.28): el padre de cada primitiva es el experto de mayor uso diferencial (uso en ella menos uso
medio en las otras). B12 (commit cc6e6db) usaba el más elegido, que era el mismo para casi todas.
Uso: .venv/bin/python -m colab.b12_raices SALIDA.json [pasos máximos por etapa] [semilla]
"""

import copy
import json
import random
import sys
import time

import numpy as np
import torch

from colab.b9_bits import SINGLES, SIZE, eval_sets, routing, train
from colab.b10b_cobertura import COVERAGE, UNSEEN
from mf_moe.bits import make_bits_example
from mf_moe.division import _grow_param
from mf_moe.model import build, moe, routers
from mf_moe.train import adamw, scores

PAIRS = COVERAGE[10]
E0 = 16
ROW_SCALE = 1.0  # norma de la fila del hijo, en múltiplos de la del padre
ARMS = ("raiz", "interp", "azar", "padres_azar", "fijo")


def _usage(idx: np.ndarray, n: int) -> np.ndarray:
    """Fracción del ruteo de cada experto por capa: [capas, expertos]; `idx` es [tokens, capas, k]."""
    return np.stack([np.bincount(idx[:, L].reshape(-1), minlength=n) / idx[:, L].size for L in range(idx.shape[1])])


def parents(rt: dict, n: int) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Por primitiva y capa: el experto de mayor uso diferencial, y ese uso diferencial."""
    u = {t: _usage(rt[t]["idx"], n) for t in SINGLES}
    lift = {t: u[t] - np.mean([u[o] for o in SINGLES if o != t], 0) for t in SINGLES}
    return {t: lift[t].argmax(1) for t in SINGLES}, {t: lift[t].max(1).round(3).tolist() for t in SINGLES}


@torch.no_grad()
def grow(model, opt, arm: str, rng: random.Random) -> tuple[dict[str, int], dict]:
    """Un hijo por par en cada capa; devuelve el índice del hijo de cada par (el mismo en todas las capas) y los padres."""
    probe = {t: [make_bits_example(t, random.Random(f"init:{t}:{i}"), named=True) for i in range(100)]
             for t in SINGLES + PAIRS}  # fmt: skip
    rt = routing(model, probe)
    top, lift = parents(rt, E0)
    child = {}
    for pair in PAIRS:
        f, g = pair.split("+")
        a, b = (f, g) if arm != "padres_azar" else rng.sample([p for p in SINGLES if p not in (f, g)], 2)
        for L in range(len(routers(model))):
            experts, router = moe(model, L)
            pa, pb = int(top[a][L]), int(top[b][L])
            for name in ("gate_up_proj", "down_proj"):
                W = getattr(experts, name)
                w = {"raiz": W[pa], "azar": torch.randn_like(W[pa]) * W[:E0].std()}.get(arm, (W[pa] + W[pb]) / 2)
                _grow_param(opt, experts, name, pa, w.clone())
            row = torch.tensor(rt[pair]["h"][L].mean(0) - np.mean([rt[t]["h"][L].mean(0) for t in rt], 0))
            _grow_param(opt, router, "weight", pa, row / row.norm() * router.weight[pa].norm() * ROW_SCALE)
            router.bias = torch.cat([router.bias, router.bias[pa : pa + 1] - 1])  # mudo: solo entra si se fuerza
            router.load = torch.cat([router.load, router.load[pa : pa + 1]])
            experts.num_experts += 1
            router.num_experts += 1
        child[pair] = experts.num_experts - 1
    return child, {"padres": {t: top[t].tolist() for t in SINGLES}, "uso_diferencial": lift}


def force_by_task(model, child: dict[str, int]):
    """Hook que fija `force` de cada router según el nombre de la tarea al comienzo de cada fila."""

    def hook(_, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        c = torch.tensor([child.get(bytes(r[:16].tolist()).split(b":")[0].decode(errors="ignore"), -1) for r in ids])
        for r in routers(model):
            r.force = c[:, None].expand(ids.shape).reshape(-1)

    return model.register_forward_pre_hook(hook, with_kwargs=True)


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0 = time.time()
    res = {"semilla": seed, "pasos_maximos": max_steps, "pares": PAIRS, "no_vistos": UNSEEN, "brazos": {}}
    ev = eval_sets(SINGLES + PAIRS + UNSEEN)
    held = {e.prompt for exs in ev.values() for e in exs}
    stage1 = {}
    for n in (E0, E0 + len(PAIRS)):  # el brazo fijo entrena sus primitivas con los 26 expertos
        model = build(n, seed=seed, **SIZE)
        opt = adamw(model)
        r = train(model, opt, SINGLES, {t: ev[t] for t in SINGLES}, max_steps, random.Random(seed), f"primitivas:{n}", held=held)
        stage1[n] = (model, opt, {k: r[k] for k in ("pasos", "domino", "historia")})
    for arm in ARMS:
        model, opt, s1 = copy.deepcopy(stage1[E0 + len(PAIRS) if arm == "fijo" else E0])  # copia modelo y AdamW juntos
        child, par = ({}, None) if arm == "fijo" else grow(model, opt, arm, random.Random(seed))
        hook = force_by_task(model, child)
        after_birth = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in SINGLES}
        r = train(model, opt, PAIRS, {t: ev[t] for t in PAIRS}, max_steps, random.Random(seed + 1), arm,
                  replay=SINGLES, held=held)  # fmt: skip
        acc = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in SINGLES + PAIRS + UNSEEN}
        hook.remove()
        for rt in routers(model):
            rt.force = None
        free = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in PAIRS} if child else None
        out = {"primitivas": s1, "pares": {k: r[k] for k in ("pasos", "domino", "historia")}, "exactitud": acc,
               "primitivas_al_nacer": after_birth, "pares_sin_forzar": free, "padres": par,
               "primitivas_media": round(float(np.mean([acc[t] for t in SINGLES])), 3),
               "pares_media": round(float(np.mean([acc[t] for t in PAIRS])), 3),
               "no_vistos_media": round(float(np.mean([acc[t] for t in UNSEEN])), 3)}  # fmt: skip
        res["brazos"][arm] = out
        print(json.dumps({"brazo": arm, "pasos_pares": r["pasos"], **{k: out[k] for k in ("primitivas_media", "pares_media", "no_vistos_media")}}), flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
