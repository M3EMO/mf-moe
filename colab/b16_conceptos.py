"""B16 (ticket 15): ¿qué distancia predice el olvido de un concepto: el árbol humano, el estado o el router?

Precedente: en edición de conocimiento, el coseno de estados a ~1/3 de la profundidad predice el daño a otros
hechos (CLaRE, arXiv 2603.19297). No apareció el árbol humano de conceptos ni el solapamiento de expertos como
predictor del olvido. Modelo: Granite-3.1-1B-A400M (MoE preentrenado de la fase 2, capas desde TRAIN_FROM en fp32).
Por área de WordNet (AREAS), el modelo aprende hechos nuevos que chocan con lo que sabe: N_TRAIN conceptos del área
con las definiciones permutadas entre ellos («A poodle is <definición de otro perro>»). Después se mide cuánto sube
la pérdida de la definición verdadera de los conceptos de evaluación (los entrenados quedan afuera), estratificados
por su parecido Wu-Palmer a la raíz del área. Predictores, medidos en el modelo antes de entrenar, contra el
concepto entrenado más cercano:
- arbol: similitud Wu-Palmer de WordNet (el árbol humano de conceptos);
- estado: coseno del estado residual en la capa STATE_LAYER, último token del sujeto (la medida de CLaRE);
- router: solapamiento de los top-k expertos del último token del sujeto, medio sobre las capas.
Uso: .venv/bin/python -m colab.b16_conceptos SALIDA.json [areas separadas por coma]
"""

import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from mf_moe import granite
from mf_moe.model import routers
from mf_moe.phase2 import amp, set_trainable

DEVICE = os.environ.get("MFMOE_DEVICE", "cpu")
AREAS = ("dog.n.01", "musical_instrument.n.01", "wheeled_vehicle.n.01", "fish.n.01", "disease.n.01", "tree.n.01")
N_TRAIN, N_BIN, BINS = 64, 96, (0.0, 0.4, 0.6, 0.8, 1.01)
TRAIN_FROM, STATE_LAYER, STEPS, BATCH, LR = 6, 8, 200, 16, 5e-5


def wordnet():
    try:
        import nltk
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=True)
        import nltk
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    return wn


def name(s) -> str:
    return s.lemma_names()[0].replace("_", " ")


def pick(wn, area: str, seed: int) -> tuple[list, list]:
    """Conceptos entrenados (del área) y de evaluación (N_BIN por franja de Wu-Palmer a la raíz, sin entrenados)."""
    rng, root = random.Random(f"{area}:{seed}"), wn.synset(area)
    members = sorted(set(root.closure(lambda x: x.hyponyms())), key=lambda s: s.name())
    train = rng.sample(members, N_TRAIN)
    taken = {s.name() for s in train}
    pool = [s for s in members if s.name() not in taken]
    pool += rng.sample(sorted(wn.all_synsets("n"), key=lambda s: s.name()), 6000)
    sim = {s.name(): s.wup_similarity(root) or 0.0 for s in pool}
    ev = []
    for lo, hi in zip(BINS, BINS[1:]):
        band = sorted({s.name(): s for s in pool if lo <= sim[s.name()] < hi and s.name() not in taken}.values(),
                      key=lambda s: s.name())  # fmt: skip
        ev += rng.sample(band, min(N_BIN, len(band)))
    return train, ev


def encode(tok, prefix: str, rest: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ids y etiquetas: solo la definición cuenta en la pérdida."""
    p, full = tok(prefix)["input_ids"], tok(prefix + rest)["input_ids"]
    labels = [-100] * len(p) + full[len(p) :]
    return torch.tensor(full), torch.tensor(labels)


@torch.no_grad()
def def_losses(model, tok, items: list[tuple[str, str]]) -> np.ndarray:
    model.eval()
    out = []
    for prefix, rest in items:
        ids, labels = encode(tok, prefix, rest)
        with amp(DEVICE):
            out.append(model(input_ids=ids[None].to(DEVICE), labels=labels[None].to(DEVICE)).loss.item())
    model.train()
    return np.array(out)


@torch.no_grad()
def probes(model, tok, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Estado residual en STATE_LAYER y top-k expertos por capa del último token de «A <sujeto>»."""
    model.eval()
    rs = routers(model)
    for r in rs:
        r.record = True
    H, E = [], []
    for s in subjects:
        ids = torch.tensor(tok(f"A {s}")["input_ids"])[None].to(DEVICE)
        with amp(DEVICE):
            out = model(input_ids=ids, output_hidden_states=True)
        H.append(out.hidden_states[STATE_LAYER][0, -1].float().cpu().numpy())
        E.append(np.stack([r.last[2][-1].cpu().numpy() for r in rs]))
    for r in rs:
        r.record = False
    model.train()
    return np.stack(H), np.stack(E)  # [n, d], [n, capas, k]


def predictors(wn_train, wn_ev, H_tr, H_ev, E_tr, E_ev) -> dict[str, list[float]]:
    """Por concepto de evaluación, el máximo sobre los entrenados de cada parecido."""
    norm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
    cos = norm(H_ev) @ norm(H_tr).T
    k = E_ev.shape[2]
    overlap = np.array([[np.mean([len(set(a) & set(b)) / k for a, b in zip(e, t)]) for t in E_tr] for e in E_ev])
    tree = np.array([[c.wup_similarity(t) or 0.0 for t in wn_train] for c in wn_ev])
    return {"arbol": tree.max(1).round(4).tolist(), "estado": cos.max(1).round(4).tolist(),
            "router": overlap.max(1).round(4).tolist()}  # fmt: skip


def run_area(model, tok, wn, area: str, seed: int) -> dict:
    train, ev = pick(wn, area, seed)
    defs = [s.definition() for s in train]
    perm, prng = list(range(N_TRAIN)), random.Random(f"perm:{area}:{seed}")
    while any(i == p for i, p in enumerate(perm)):  # derangement: ningún concepto conserva su definición
        prng.shuffle(perm)
    new = [(f"A {name(s)} is", f" {defs[p]}.") for s, p in zip(train, perm)]
    true_ev = [(f"A {name(s)} is", f" {s.definition()}.") for s in ev]
    true_tr = [(f"A {name(s)} is", f" {d}.") for s, d in zip(train, defs)]
    H_tr, E_tr = probes(model, tok, [name(s) for s in train])
    H_ev, E_ev = probes(model, tok, [name(s) for s in ev])
    before, before_tr = def_losses(model, tok, true_ev), def_losses(model, tok, true_tr)
    # copia en CPU solo lo que cambia (entrenables y buffers del router): en la L4 no entra una segunda copia
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    rng, t0, hist = random.Random(seed), time.time(), []
    for step in range(STEPS):
        batch = [new[i] for i in rng.sample(range(N_TRAIN), BATCH)]
        loss = 0.0
        for prefix, rest in batch:  # de a una oración: sin relleno, las definiciones son cortas
            ids, labels = encode(tok, prefix, rest)
            with amp(DEVICE):
                l = model(input_ids=ids[None].to(DEVICE), labels=labels[None].to(DEVICE)).loss / BATCH  # noqa: E741
            l.backward()
            loss += l.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == STEPS - 1:
            hist.append({"paso": step, "perdida": round(loss, 4)})
    after, after_tr = def_losses(model, tok, true_ev), def_losses(model, tok, true_tr)
    learned = def_losses(model, tok, new)
    model.load_state_dict(state, strict=False)  # cada área parte del mismo Granite
    return {"area": area, "entrenados": [s.name() for s in train], "evaluados": [s.name() for s in ev],
            "subida": (after - before).round(4).tolist(), "antes": before.round(4).tolist(),
            "subida_entrenados_verdad": float((after_tr - before_tr).mean()), "hechos_nuevos_final": float(learned.mean()),
            "historia": hist, "predictores": predictors(train, ev, H_tr, H_ev, E_tr, E_ev),
            "segundos": round(time.time() - t0)}  # fmt: skip


def main(path: str, areas: str = ",".join(AREAS), seed: int = 0, model=None, tok=None) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    res |= {"semilla": seed, "config": {"n_train": N_TRAIN, "n_bin": N_BIN, "train_from": TRAIN_FROM,
                                        "state_layer": STATE_LAYER, "steps": STEPS, "batch": BATCH, "lr": LR}}  # fmt: skip
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    for area in areas.split(","):
        if area in res["areas"]:
            continue
        res["areas"][area] = r = run_area(model, tok, wn, area, seed)
        sub, pr = np.array(r["subida"]), r["predictores"]
        rho = {k: round(float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(sub)))[0, 1]), 3)
               for k, v in pr.items()}  # fmt: skip
        print(json.dumps({"area": area, "spearman": rho, "subida_media": round(float(sub.mean()), 3),
                          "nuevos": round(r["hechos_nuevos_final"], 3), "segundos": r["segundos"]}), flush=True)  # fmt: skip
        json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
