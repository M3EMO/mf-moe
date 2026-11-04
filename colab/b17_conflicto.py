"""B17 (ticket 15): B16 con conflicto real. ¿El router predice qué concepto se daña, más allá del estado y del léxico?

B16 falló su control: las definiciones permutadas seguían en el entrenamiento y el modelo memorizó el conjunto.
Además, 200 pasos (~50 épocas) produjeron un colapso general del formato, como el «estilo de alucinación» que
deja memorizar hechos desconocidos (Gekhman et al., EMNLP 2024). Cambios:
- cada concepto entrenado recibe la definición de un concepto lejano (Wu-Palmer a la raíz < FAR), usada una sola
  vez; los donantes quedan fuera de la evaluación, así la definición verdadera de ningún concepto se entrena;
- el entrenamiento corta cuando la pérdida media de los hechos nuevos baja de STOP_LOSS (defensa de Gekhman);
- lotes con relleno, y covariables para el control: pérdida inicial, tokens del sujeto, frecuencia (SemCor) y
  solapamiento léxico con el entrenado más parecido (sujetos que comparten el último token comparten ruteo).
Puertas en el código: tras la primera área, C0 (conflicto real y específico) y el tiempo; tras la segunda, el
control positivo del estado. Precedente del ruteo: el fine-tuning de un MoE cambia el ruteo de las capas
siguientes y eso rompe hechos (MoEEdit, ICLR 2026); nadie usó el solapamiento previo para predecir qué hecho cae.
Uso: .venv/bin/python -m colab.b17_conflicto SALIDA.json [areas separadas por coma]
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from colab.b16_conceptos import DEVICE, TRAIN_FROM, name, pick, predictors, probes, wordnet
from mf_moe import granite
from mf_moe.phase2 import amp, set_trainable

FAR, STOP_LOSS, MAX_STEPS, CHECK_EVERY, BATCH, EVAL_BATCH, LR = 0.3, 0.5, 300, 10, 16, 32, 5e-5
# B17b: dog.n.01 quedó vista en el primer intento de B17 (cortado por un C0 mal calibrado); bird la reemplaza
AREAS = ("bird.n.01", "musical_instrument.n.01", "wheeled_vehicle.n.01", "fish.n.01", "disease.n.01", "tree.n.01")
FAR_BAND = 0.4  # franja lejana de la evaluación (Wu-Palmer a la raíz): línea de base del derrame general
MAX_AREA_SECONDS = 900  # puerta de tiempo: B16 estimó 5 min por área sin medir y tardó 22


def donors(wn, area: str, taken: set[str], n: int, seed: int) -> list:
    """n conceptos lejanos del área cuyas definiciones serán los hechos nuevos."""
    rng, root = random.Random(f"donantes:{area}:{seed}"), wn.synset(area)
    pool = rng.sample(sorted(wn.all_synsets("n"), key=lambda s: s.name()), 4 * n + 200)
    far = [s for s in pool if s.name() not in taken and (s.wup_similarity(root) or 0.0) < FAR]
    assert len(far) >= n, f"{area}: solo {len(far)} donantes lejanos"
    return far[:n]


def batch(tok, items: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ids, máscara y etiquetas con relleno a la derecha; solo la definición cuenta en la pérdida."""
    rows = []
    for prefix, rest in items:
        p, full = tok(prefix)["input_ids"], tok(prefix + rest)["input_ids"]
        rows.append((full, [-100] * len(p) + full[len(p) :]))
    L = max(len(f) for f, _ in rows)
    ids = torch.tensor([f + [0] * (L - len(f)) for f, _ in rows])
    mask = torch.tensor([[1] * len(f) + [0] * (L - len(f)) for f, _ in rows])
    labels = torch.tensor([lab + [-100] * (L - len(lab)) for _, lab in rows])
    return ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)


@torch.no_grad()
def def_losses(model, tok, items: list[tuple[str, str]]) -> np.ndarray:
    """Pérdida media por oración (nats por token de la definición)."""
    model.eval()
    out = []
    for i in range(0, len(items), EVAL_BATCH):
        ids, mask, labels = batch(tok, items[i : i + EVAL_BATCH])
        with amp(DEVICE):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        lab = labels[:, 1:]
        ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), lab, ignore_index=-100, reduction="none")
        out += ((ce * (lab != -100)).sum(1) / (lab != -100).sum(1)).tolist()
    model.train()
    return np.array(out)


def covariates(tok, train, ev) -> dict[str, list[float]]:
    toks = lambda s: tok(" " + name(s))["input_ids"]  # noqa: E731
    tr = [set(toks(s)) for s in train]
    lex = [max(len(set(toks(c)) & t) / len(set(toks(c)) | t) for t in tr) for c in ev]
    last = [float(any(toks(c)[-1] in t for t in tr)) for c in ev]
    return {"tokens_sujeto": [len(toks(c)) for c in ev], "frecuencia": [sum(lm.count() for lm in c.lemmas()) for c in ev],
            "lexico": [round(x, 4) for x in lex], "ultimo_token_compartido": last}  # fmt: skip


def run_area(model, tok, wn, area: str, seed: int, anchor: int = 0) -> dict:
    """`anchor` > 0 suma a cada lote BATCH definiciones verdaderas de `anchor` conceptos lejanos (conjunto de
    preservación, como en edición de conocimiento): frena el derrame general de estilo (Gekhman et al.)."""
    train, ev = pick(wn, area, seed)
    don = donors(wn, area, {s.name() for s in train + ev}, len(train) + anchor, seed)
    don, anchors = don[: len(train)], don[len(train) :]
    keep = [(f"A {name(s)} is", f" {s.definition()}.") for s in anchors]
    new = [(f"A {name(s)} is", f" {d.definition()}.") for s, d in zip(train, don)]
    true_ev = [(f"A {name(s)} is", f" {s.definition()}.") for s in ev]
    true_tr = [(f"A {name(s)} is", f" {s.definition()}.") for s in train]
    H_tr, E_tr = probes(model, tok, [name(s) for s in train])
    H_ev, E_ev = probes(model, tok, [name(s) for s in ev])
    before, before_tr, before_new = def_losses(model, tok, true_ev), def_losses(model, tok, true_tr), def_losses(model, tok, new)
    # copia en CPU solo lo que cambia (entrenables y buffers del router): en la L4 no entra una segunda copia
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    rng, t0, hist, step = random.Random(seed), time.time(), [], 0
    for step in range(1, MAX_STEPS + 1):
        parts = [[new[i] for i in rng.sample(range(len(new)), BATCH)]]
        parts += [[keep[i] for i in rng.sample(range(len(keep)), BATCH)]] if keep else []
        for part in parts:  # de a medio lote: 32 oraciones juntas no entran en la L4 (OOM en la calibración)
            ids, mask, labels = batch(tok, part)
            with amp(DEVICE):
                (model(input_ids=ids, attention_mask=mask, labels=labels).loss / len(parts)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % CHECK_EVERY == 0:
            cur = float(def_losses(model, tok, new).mean())
            hist.append({"paso": step, "nuevos": round(cur, 4), "segundos": round(time.time() - t0)})
            if cur < STOP_LOSS:
                break
    after, after_tr, learned = def_losses(model, tok, true_ev), def_losses(model, tok, true_tr), def_losses(model, tok, new)
    model.load_state_dict(state, strict=False)  # cada área parte del mismo Granite
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"area": area, "entrenados": [s.name() for s in train], "evaluados": [s.name() for s in ev],
            "donantes": [s.name() for s in don], "anclas": len(keep), "pasos": step,
            "raiz": [round(s.wup_similarity(wn.synset(area)) or 0.0, 4) for s in ev], "subida": (after - before).round(4).tolist(),
            "antes": before.round(4).tolist(), "subida_entrenados_verdad": float((after_tr - before_tr).mean()),
            "nuevos_antes": float(before_new.mean()), "nuevos_final": float(learned.mean()), "historia": hist,
            "predictores": predictors(train, ev, H_tr, H_ev, E_tr, E_ev), "covariables": covariates(tok, train, ev),
            "segundos": round(time.time() - t0)}  # fmt: skip


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def gate(res: dict) -> str | None:
    """Puertas preregistradas (ticket 15): devuelve el motivo para cortar, o None."""
    done = list(res["areas"].values())
    r = done[0]
    far = float(np.mean([x for x, w in zip(r["subida"], r["raiz"]) if w < FAR_BAND]))
    # la media de toda la evaluación incluye a los vecinos dañados, que son el efecto medido: la base es la franja lejana
    if not (r["nuevos_final"] < STOP_LOSS and r["subida_entrenados_verdad"] > max(0.5, 2 * far)):
        return "C0: sin conflicto específico en la primera área"
    if r["segundos"] > MAX_AREA_SECONDS:
        return f"tiempo: {r['segundos']} s en la primera área"
    if len(done) >= 2 and all(spearman(d["predictores"]["estado"], d["subida"]) <= 0 for d in done[:2]):
        return "C1: el estado no predice en las dos primeras áreas"
    return None


def main(path: str, areas: str = ",".join(AREAS), seeds: str = "0", anchor: str = "0", freeze_from: str = "",
         model=None, tok=None) -> None:  # fmt: skip
    """Recorre semilla por semilla todas las áreas; cada clave de `areas` es «área|s<semilla>»."""
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    res |= {"semillas": seeds, "config": {"anclas": int(anchor), "congeladas_desde": freeze_from, "far": FAR, "stop_loss": STOP_LOSS, "max_steps": MAX_STEPS, "batch": BATCH,
                                        "lr": LR, "train_from": TRAIN_FROM}}  # fmt: skip
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    if freeze_from:  # variante de la calibración: entrena solo las capas TRAIN_FROM..freeze_from-1
        for layer in model.model.layers[int(freeze_from) :]:
            layer.requires_grad_(False)
    for area, seed in [(a, int(s)) for s in seeds.split(",") for a in areas.split(",")]:
        key = f"{area}|s{seed}"
        if key in res["areas"]:
            continue
        res["areas"][key] = r = run_area(model, tok, wn, area, seed, int(anchor))
        sub = r["subida"]
        print(json.dumps({"area": key, "spearman": {k: round(spearman(v, sub), 3) for k, v in r["predictores"].items()},
                          "subida_media": round(float(np.mean(sub)), 3), "entrenados": round(r["subida_entrenados_verdad"], 3),
                          "nuevos": round(r["nuevos_final"], 3), "pasos": r["pasos"], "segundos": r["segundos"]}), flush=True)  # fmt: skip
        why = gate(res)
        res["corte"] = why
        json.dump(res, open(path, "w"), ensure_ascii=False)
        if why:
            print(json.dumps({"corte": why}), flush=True)
            return


if __name__ == "__main__":
    main(*sys.argv[1:6])
