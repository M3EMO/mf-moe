"""B15 (ticket 15): ¿el olvido depende de la distancia en el árbol de los datos, y repetir según el árbol lo baja?

Precedentes (Lacuna, 2026-11-04): el olvido de una clase vieja crece con su parecido a lo nuevo (CLAD, AAAI 2024),
pero el olvido máximo aparece a parecido intermedio (Ramasesh, ICLR 2021; Lee, ICML 2021). No encontré repetición
elegida por distancia en un árbol conocido. Datos: las 8 lenguas MRHM de B13 (`mf_moe.hierarchy`), V = 8, M = 2.
Etapa A: las 7 lenguas viejas juntas, una vez por semilla. Etapa B: la lengua nueva `h` sola (NEW ejemplos por lote)
más REPLAY ejemplos de lenguas viejas según el brazo, sobre copias de la etapa A con su AdamW:
- sin_repeticion: nada (control positivo: ¿el olvido cambia con la distancia d = 1, 2, 3 a `h`?);
- uniforme: cada lengua vieja con la misma probabilidad (las lejanas son 4 de 7: el azar ya repite mayormente lejos);
- cerca: peso 4^-(d-1) por lengua (57 % a la hermana);
- lejos: solo las 4 lenguas a d = 3;
- perdida: peso ∝ subida de pérdida desde el cierre de A, medida cada EVAL_EVERY pasos (Replay on Demand,
  arXiv 2609.40089, con la pérdida de A como referencia).
Medida: pérdida por lengua en un conjunto fijo, cada EVAL_EVERY pasos de B. Olvido = pérdida final − cierre de A.
Uso: .venv/bin/python -m colab.b15_repeticion SALIDA.json SEMILLA [pasos_a pasos_b]
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch

from colab.b9_bits import DEVICE, SIZE
from colab.b12_raices import E0
from mf_moe import hierarchy as h
from mf_moe.model import build
from mf_moe.train import adamw

NEW, REPLAY, BATCH_A = 96, 32, 128
EVAL_EVERY, EVAL_PER_LANG = 250, 256
ARMS = ("sin_repeticion", "uniforme", "cerca", "lejos", "perdida")


def arm_weights(arm: str, dist: np.ndarray, rise: np.ndarray | None = None) -> np.ndarray | None:
    """Probabilidad de cada lengua vieja en la repetición; `dist` = distancia a la nueva (0 en la nueva)."""
    old = dist > 0
    w = {"sin_repeticion": None, "uniforme": old * 1.0, "cerca": old * 4.0 ** -(dist - 1.0), "lejos": dist == 3,
         "perdida": None if rise is None else old * np.maximum(rise, 1e-3)}[arm]  # fmt: skip
    return None if w is None else np.asarray(w, float) / np.sum(w)


@torch.no_grad()
def lang_losses(model, ev: torch.Tensor) -> np.ndarray:
    """Pérdida media de cada lengua; `ev` [N_LANG, n, 1 + SEQ]."""
    model.eval()
    out = np.array([model(input_ids=x, labels=x).loss.item() for x in ev])
    model.train()
    return out


def step(model, opt, tabs, langs: np.ndarray, rng) -> None:
    x = torch.tensor(h.ids(h.sample(tabs, langs, rng)[0], langs, False), device=DEVICE)
    model(input_ids=x, labels=x).loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


def stage_b(model, opt, arm: str, tabs, new: int, ev, base: np.ndarray, steps: int, seed: int) -> dict:
    dist = h.tree_distance()[new]
    rng, w, hist = np.random.default_rng(seed + 2), arm_weights(arm, dist), []
    for s in range(steps + 1):
        if s % EVAL_EVERY == 0 or s == steps:
            L = lang_losses(model, ev)
            hist.append({"paso": s, "perdidas": L.round(4).tolist()})
            if arm == "perdida":
                w = arm_weights(arm, dist, L - base)
        if s == steps:
            break
        langs = np.full(NEW, new)
        if w is not None:
            langs = np.concatenate([langs, rng.choice(h.N_LANG, size=REPLAY, p=w)])
        step(model, opt, tabs, langs, rng)
    olv = np.array(hist[-1]["perdidas"]) - base
    return {"historia": hist, "olvido": olv.round(4).tolist(),
            "olvido_d123": [round(float(olv[dist == d].mean()), 4) for d in (1, 2, 3)],
            "nueva_final": hist[-1]["perdidas"][new]}  # fmt: skip


def main(path: str, seed: int, steps_a: int = 10000, steps_b: int = 4000) -> None:
    torch.set_num_threads(4)
    new = seed % h.N_LANG
    res = json.load(open(path)) if os.path.exists(path) else {}
    res |= {"semilla": seed, "nueva": new, "pasos": [steps_a, steps_b], "V_M": [h.V, h.M], "lote": [NEW, REPLAY]}
    res.setdefault("brazos", {})
    save = lambda: json.dump(res, open(path, "w"), ensure_ascii=False)  # noqa: E731
    tabs = h.rules(seed)
    erng = np.random.default_rng(10**6 + seed)
    ev = torch.stack([torch.tensor(h.ids(h.sample(tabs, np.full(EVAL_PER_LANG, g), erng)[0], np.full(EVAL_PER_LANG, g),
                                         False), device=DEVICE) for g in range(h.N_LANG)])  # fmt: skip
    model = build(E0, seed=seed, **SIZE).to(DEVICE)
    opt, ckpt = adamw(model), path + ".etapa_a.pt"
    if os.path.exists(ckpt):  # retoma: la etapa A es la misma para todos los brazos
        st = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(st["modelo"])
        opt.load_state_dict(st["opt"])
    else:
        res["brazos"] = {}  # una etapa A nueva invalida los brazos de la anterior
        old, rng, t0 = np.delete(np.arange(h.N_LANG), new), np.random.default_rng(seed + 1), time.time()
        for _ in range(steps_a):
            step(model, opt, tabs, rng.choice(old, size=BATCH_A), rng)
        torch.save({"modelo": model.state_dict(), "opt": opt.state_dict()}, ckpt)
        res["segundos_a"] = round(time.time() - t0)
    base = lang_losses(model, ev)
    res["cierre_a"] = base.round(4).tolist()
    save()
    for arm in ARMS:
        if arm in res["brazos"]:
            continue
        t0 = time.time()
        res["brazos"][arm] = stage_b(*copy.deepcopy((model, opt)), arm, tabs, new, ev, base, steps_b, seed)
        res["brazos"][arm]["segundos"] = round(time.time() - t0)
        b = res["brazos"][arm]
        print(json.dumps({"brazo": arm, "olvido_d123": b["olvido_d123"], "nueva": b["nueva_final"],
                          "segundos": b["segundos"]}), flush=True)  # fmt: skip
        save()


if __name__ == "__main__":
    h.V, h.M = 8, 2  # la gramática de B13b (8 2): hierarchy lee las constantes al llamarse; acá no toca los tests
    a = sys.argv[1:]
    main(a[0], int(a[1]), *(int(x) for x in a[2:4]))
