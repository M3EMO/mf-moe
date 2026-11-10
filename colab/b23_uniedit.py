"""B23 (ticket 03 del mapa mf-moe-generalidad): ¿el daño del conflicto sigue al tipo de la entidad, a la relación o a ninguno?

Banco con hechos reales de UniEdit. Plantilla: «The {propiedad} of {cabeza} is» + « {valor}.». Diseño preregistrado en el ticket.
- seleccion: recorre UniEdit y guarda hasta NEED hechos por tipo que pasan F0 (el valor real gana a 4 donantes y no gana con una
  cabeza ajena). F0 salió de la etapa 0 en CPU. Una fila por cabeza, así las cabezas son disjuntas entre entrenamiento, evaluación y anclas.
- gradiente: cada conjunto entrena 64 hechos de un tipo con el valor de otro hecho de igual propiedad, solo con una mitad al azar de
  las propiedades del tipo. La otra mitad forma la celda «mismo tipo, otra relación», que el muestreo natural casi no produce.
  Las anclas salen de los otros tipos y no usan la mitad entrenada: así no protegen de forma directa a la relación medida.
Uso: python -m colab.b23_uniedit seleccion SALIDA.json
     python -m colab.b23_uniedit gradiente SALIDA.json HECHOS.json 0,1,2 4   (conjuntos y número de repeticiones)
     python -m colab.b23_uniedit consolidacion SALIDA.json HECHOS.json [BRAZOS] [CONTROLES]   (p. ej. prec24 1)
- consolidacion (etapa 4): los mismos conjuntos 0 y 1 con 8 expertos congelados por capa. cons24 congela los que más usa lo viejo
  (B21), conf24 los de producto interno más negativo entre el gradiente medio nuevo y el viejo, azar24 al azar, prec24 los de
  mayor Fisher diagonal de lo viejo (B21). Un control sin congelar por tipo comprueba que la etapa 2 se repite en esta sesión.
"""

import collections as C
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b21_contencion as b21
from colab.b16_conceptos import DEVICE, TRAIN_FROM
from mf_moe import granite
from mf_moe.model import moe
from mf_moe.phase2 import amp, set_trainable

REPO = "qizhou/UniEdit"
TYPES = {
    "persona": "Q5",
    "organizacion": "Q4830453 Q43229 Q327333 Q163740 Q7278 Q31855 Q3918 Q48204 Q7315155 Q157031 Q3914 Q955824 Q783794 Q6881511 Q1400264 Q1666019 Q2085381 Q7210356 Q847017 Q2367225 Q480242 Q2467461 Q176799 Q210167",
    "biomedicina": "Q2996394 Q7187 Q8054 Q5058355 Q21014462 Q189118 Q15304597 Q14860489 Q4915012 Q2656896 Q16521 Q23038290 Q112193867",
    "publicacion": "Q737498 Q5633421 Q732577 Q1907875 Q13442814 Q187685 Q591041 Q23927052 Q1002697 Q21481766 Q58632367 Q191067 Q1143604 Q773668 Q1402850 Q11032 Q41298",
    "obra_cultural": "Q7725634 Q11424 Q3331189 Q482994 Q21191270 Q134556 Q3305213 Q11060274 Q7889 Q24862 Q47461344 Q105543609 Q125191 Q860861 Q5398426 Q838948 Q13464614 Q93184 Q15416 Q506240 Q1464125 Q55850593 Q169930 Q106805967 Q4502142 Q20437094 Q571 Q87167 Q1667921 Q83790",
    "lugar_estructura": "Q27686 Q811979 Q41176 Q839954 Q35666 Q16917 Q33506 Q537127 Q486972 Q41253 Q16970 Q15911738 Q131596 Q1003207 Q3947 Q179049 Q718966 Q1415423 Q13226383",
    "derecho_gobierno": "Q7604686 Q100532807 Q56731284 Q7604693 Q7604692 Q105774620 Q2334719 Q294414 Q4164871 Q10541491",
    "software_datos": "Q341 Q7397 Q35127 Q1172284 Q7094076 Q8366 Q620615 Q506883 Q783866 Q9143",
}
TYPE_OF = {q: t for t, v in TYPES.items() for q in v.split()}
K, NEED, CHUNK, BUDGET, MIN_TYPES = 4, 444, 64, 1300, 6  # BUDGET en segundos: 22 min de selección y 8 de arranque y descarga caben en 1 unidad
N_TRAIN, N_EVAL, ANCHORS, MIN_CELL, TOP_COS = 64, 60, 512, 10, 10
ARMS4, SETS4 = ("cons24", "conf24", "azar24"), (0, 1)


def uniedit_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    rows = []
    for f in sorted(x for x in list_repo_files(REPO, repo_type="dataset") if x.startswith(("train/", "test/")) and x.endswith(".json")):
        rows += [r["edit"] for r in json.load(open(hf_hub_download(REPO, f, repo_type="dataset"), encoding="utf-8")).values()]
    return rows


def candidates(rows: list[dict]) -> dict[str, list[dict]]:
    """Por tipo, un hecho al azar por nombre de cabeza, con K donantes de igual propiedad y una cabeza ajena."""
    rows = [e for e in rows if e["tail_entity"]["datatype"] == "wikibase-item" and e["property"]["single_value"]]
    rows.sort(key=lambda e: (e["head_entity"]["id"], e["property"]["id"], e["tail_entity"]["value"]["id"]))
    random.Random("b23:filas").shuffle(rows)  # sin esto, cada cabeza conserva su propiedad de número más bajo
    freq = C.Counter(q for e in rows for q in e["head_entity"].get("P31") or [])
    values, swaps, seen, out = C.defaultdict(dict), C.defaultdict(list), set(), C.defaultdict(list)
    for e in rows:
        values[e["property"]["id"]][e["tail_entity"]["value"]["id"]] = e["tail_entity"]["value"]["label"]
        swaps[e["property"]["id"]].append((e["head_entity"]["label"], e["tail_entity"]["value"]["id"]))
    for e in rows:
        h, p, v = e["head_entity"], e["property"], e["tail_entity"]["value"]
        mapped = [q for q in h.get("P31") or [] if q in TYPE_OF]
        if not mapped or h["label"] in seen:
            continue
        others = sorted({lab for q, lab in values[p["id"]].items() if q != v["id"] and lab.lower() != v["label"].lower()})
        foreign = sorted({lab for lab, q in swaps[p["id"]] if q != v["id"] and lab != h["label"]})
        if len(others) < K or not foreign:
            continue
        seen.add(h["label"])
        rng = random.Random(f"b23:{h['id']}:{p['id']}")
        out[TYPE_OF[max(mapped, key=lambda q: (freq[q], q))]].append({"propiedad": p["label"], "cabeza": h["label"], "valor": v["label"], "donantes": rng.sample(others, K), "cabeza_ajena": rng.choice(foreign)})
    for t, v in out.items():
        random.Random(f"b23:orden:{t}").shuffle(v)
    return dict(out)


def fact(x: dict, value: str | None = None, head: str | None = None) -> tuple[str, str]:
    return f"The {x['propiedad']} of {head or x['cabeza']} is", f" {value or x['valor']}."


def candidate_losses(model, tok, facts: list[dict], foreign: bool = False) -> np.ndarray:
    """Pérdida del valor real y de los K donantes: [hechos, K + 1]."""
    items = [fact(x, val, x["cabeza_ajena"] if foreign else None) for x in facts for val in [x["valor"], *x["donantes"]]]
    return b17.def_losses(model, tok, items).reshape(-1, K + 1)


def wins(L: np.ndarray) -> np.ndarray:
    return L[:, 0] < L[:, 1:].min(1)


def screen(model, tok, pool: dict[str, list[dict]], path: str, budget: float = BUDGET) -> dict:
    """Pasa por F0 hasta NEED hechos por tipo. Guarda por ronda: una VM nueva retoma donde quedó."""
    res = json.load(open(path)) if os.path.exists(path) else {"hechos": {t: [] for t in pool}, "vistos": {t: 0 for t in pool}, "secuencias": 0, "segundos": 0}
    start = time.time() - res["segundos"]
    while time.time() - start < budget:
        todo = [t for t in pool if len(res["hechos"][t]) < NEED and res["vistos"][t] < len(pool[t])]
        if not todo:
            break
        for t in todo:
            part = pool[t][res["vistos"][t] : res["vistos"][t] + CHUNK]
            known = [x for x, k in zip(part, wins(candidate_losses(model, tok, part))) if k]
            floor = wins(candidate_losses(model, tok, known, foreign=True))
            res["hechos"][t] += [x for x, f in zip(known, floor) if not f]
            res["vistos"][t] += len(part)
            res["secuencias"] += (K + 1) * (len(part) + len(known))
        res["segundos"] = round(time.time() - start)
        tmp = path + ".tmp"
        json.dump(res, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, path)
        print(json.dumps({"segundos": res["segundos"], "secuencias": res["secuencias"], "por_segundo": round(res["secuencias"] / max(1, res["segundos"]), 1), "hechos": {t: len(v) for t, v in res["hechos"].items()}, "vistos": res["vistos"]}), flush=True)
    return res


def split(hechos: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Evaluación fija por tipo y el resto, de donde salen entrenados y anclas."""
    ev, pool = {}, {}
    for t, v in hechos.items():
        chosen = set(random.Random(f"b23:evaluacion:{t}").sample(range(len(v)), N_EVAL))
        ev[t], pool[t] = [v[i] for i in sorted(chosen)], [x for i, x in enumerate(v) if i not in chosen]
    return ev, pool


def half(pool: list[dict], ev: list[dict], t: str, s: int) -> set[str]:
    """Mitad de las propiedades que alcanza para entrenar y deja ambas celdas del tipo con evaluación suficiente."""
    rng, props = random.Random(f"b23:mitad:{t}:{s}"), sorted({x["propiedad"] for x in pool + ev})
    for _ in range(2000):
        rng.shuffle(props)
        h = set(props[: len(props) // 2])
        same = sum(x["propiedad"] in h for x in ev)
        if sum(x["propiedad"] in h for x in pool) >= N_TRAIN and min(same, len(ev) - same) >= MIN_CELL:
            return h
    raise SystemExit(f"{t}: ninguna mitad deja {N_TRAIN} hechos para entrenar y {MIN_CELL} por celda")


def anchors(pool: dict[str, list[dict]], t: str, s: int, h: set[str]) -> list[dict]:
    """Hechos verdaderos de los otros tipos, en partes iguales, fuera de la mitad entrenada."""
    others, out = sorted(k for k in pool if k != t), []
    for o in others:
        free = [x for x in pool[o] if x["propiedad"] not in h]
        out += random.Random(f"b23:anclas:{t}:{s}:{o}").sample(free, min(ANCHORS // len(others), len(free)))
    return out


@torch.no_grad()
def states(model, tok, facts: list[dict]) -> np.ndarray:
    """Estado residual en la capa de la sonda, último token de la cabeza dentro de la plantilla (la medida de CLaRE)."""
    model.eval()
    out = []
    for x in facts:
        ids = torch.tensor(tok(f"The {x['propiedad']} of {x['cabeza']}")["input_ids"])[None].to(DEVICE)
        with amp(DEVICE):
            out.append(model(input_ids=ids, output_hidden_states=True).hidden_states[b16.STATE_LAYER][0, -1].float().cpu().numpy())
    model.train()
    return np.stack(out)


def top_cosine(H_ev: np.ndarray, H_tr: np.ndarray) -> np.ndarray:
    norm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
    return np.sort(norm(H_ev) @ norm(H_tr).T, axis=1)[:, -TOP_COS:].mean(1)


def run_set(model, tok, t: str, s: int, order: int, ev: dict, pool: dict, all_ev: list[dict], base: dict, state: dict, probe=None) -> dict:
    h = half(pool[t], ev[t], t, s)
    train = random.Random(f"b23:entrenados:{t}:{s}").sample([x for x in pool[t] if x["propiedad"] in h], N_TRAIN)
    pick = random.Random(f"b23:conflicto:{t}:{s}")
    new, keep, true_tr = [fact(x, pick.choice(x["donantes"])) for x in train], [fact(x) for x in anchors(pool, t, s, h)], [fact(x) for x in train]
    before_tr, before_new, H_tr = b17.def_losses(model, tok, true_tr), b17.def_losses(model, tok, new), states(model, tok, train)
    if probe:  # las anclas vienen agrupadas por tipo: lo viejo del brazo se sortea entre todas, sin tocar los lotes del entrenamiento
        probe.pre(model, tok, new, random.Random(f"b23:viejo:{t}:{s}").sample(keep, len(keep)))
    params =[p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=b17.LR, weight_decay=0.0)
    rng, t0, hist, step = random.Random(f"b23:lotes:{t}:{s}:{order}"), time.time(), [], 0
    torch.manual_seed(1000 * s + order)
    for step in range(1, b17.MAX_STEPS + 1):
        for part in ([new[i] for i in rng.sample(range(len(new)), b17.BATCH)], [keep[i] for i in rng.sample(range(len(keep)), b17.BATCH)]):
            ids, mask, labels = b17.batch(tok, part)
            with amp(DEVICE):
                (model(input_ids=ids, attention_mask=mask, labels=labels).loss / 2).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % b17.CHECK_EVERY == 0:
            cur = float(b17.def_losses(model, tok, new).mean())
            hist.append({"paso": step, "nuevos": round(cur, 4), "segundos": round(time.time() - t0)})
            if cur < b17.STOP_LOSS:
                break
    after = candidate_losses(model, tok, all_ev)
    after_tr, learned = b17.def_losses(model, tok, true_tr), b17.def_losses(model, tok, new)
    if probe:
        probe.post(model, state)
    model.load_state_dict(state, strict=False)  # cada conjunto parte del mismo Granite
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"tipo": t, "conjunto": s, "orden": order, "mitad": sorted(h), "entrenados": [x["cabeza"] for x in train], "pasos": step,
            "subida": (after[:, 0] - base["antes"][:, 0]).round(4).tolist(), "gana_despues": wins(after).astype(int).tolist(),
            "coseno": top_cosine(base["H"], H_tr).round(4).tolist(), "subida_entrenados_verdad": float((after_tr - before_tr).mean()),
            "nuevos_antes": float(before_new.mean()), "nuevos_final": float(learned.mean()), "historia": hist, "anclas": len(keep),
            "segundos": round(time.time() - t0)} | (probe.final(model, tok) if probe else {})  # fmt: skip


def gate(r: dict, evaluated: list[dict]) -> str | None:
    """Puerta C0 de B18 con otro tipo y otra relación como franja lejana."""
    far = float(np.mean([u for u, x in zip(r["subida"], evaluated) if x["tipo"] != r["tipo"] and x["propiedad"] not in r["mitad"]]))
    if not (r["nuevos_final"] < b17.STOP_LOSS and r["subida_entrenados_verdad"] > max(0.5, 2 * far)):
        return "C0: sin conflicto específico en la primera corrida"
    if r["segundos"] > b17.MAX_AREA_SECONDS:
        return f"tiempo: {r['segundos']} s en la primera corrida"
    return None


def load(model, tok, trainable: bool):
    if model is not None:
        return model, tok
    from transformers import AutoTokenizer

    model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
    if trainable:
        set_trainable(model, TRAIN_FROM)
    return model, tok


def seleccion(path: str, model=None, tok=None, rows: list[dict] | None = None) -> None:
    pool = candidates(uniedit_rows() if rows is None else rows)
    print(json.dumps({"candidatos": {t: len(v) for t, v in pool.items()}}), flush=True)
    model, tok = load(model, tok, False)
    res = screen(model, tok, pool, path)
    print(json.dumps({"FIN": {t: len(v) for t, v in res["hechos"].items()}, "entran": sum(len(v) >= NEED for v in res["hechos"].values())}), flush=True)


def conflict(model, tok, new: list, old: list, layers) -> np.ndarray:
    """Producto interno, por experto, entre el gradiente medio de lo nuevo y el de lo viejo: [capas, expertos].
    Negativo: a primer orden, el paso que baja la pérdida nueva sube la vieja. Excluye al router, como `b21.sensitivity`."""

    def mean_grad(items: list) -> list[list[torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        for i in range(0, len(items), b17.BATCH):
            part = items[i : i + b17.BATCH]
            ids, mask, labels = b17.batch(tok, part)
            with amp(DEVICE):
                (model(input_ids=ids, attention_mask=mask, labels=labels).loss * len(part) / len(items)).backward()
        out = [[(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten(1).clone() for p in (e.gate_up_proj, e.down_proj)]
               for e, _ in (moe(model, i) for i in layers)]  # fmt: skip
        model.zero_grad(set_to_none=True)
        return out

    model.eval()
    g_old = mean_grad(old)
    g_new = mean_grad(new)
    model.train()
    return np.stack([sum((a * b).sum(1) for a, b in zip(o, n)).cpu().numpy() for o, n in zip(g_old, g_new)])


class Protect(b21.Mask):
    """Brazos de B21 y conf24, que congela la misma cantidad de expertos que cons24."""

    def choose(self, model, tok, new, old, layers, U_new) -> list[list[int]]:
        if self.arm != "conf24":
            return super().choose(model, tok, new, old, layers, U_new)
        n = U_new.shape[1]
        frozen = n - round(n * b21.ARMS["cons24"][1])
        self.dots = conflict(model, tok, new, old, layers)
        return [sorted(int(x) for x in np.argsort(d, kind="stable")[:frozen]) for d in self.dots]

    def final(self, model, tok) -> dict:
        extra = {"negativos": float((self.dots < 0).mean())} if self.arm == "conf24" else {}
        return super().final(model, tok) | extra


def plan4(types: list[str], arms=ARMS4, controls: int = 99) -> list[tuple]:
    """Controles primero; después, por conjunto y tipo, los brazos juntos: un corte deja conjuntos pareados."""
    return [(t, 0, "control") for t in types[:controls]] + [(t, s, arm) for s in SETS4 for t in types for arm in arms]


def prepare(path: str, facts_path: str, model, tok) -> tuple:
    hechos = {t: v for t, v in json.load(open(facts_path, encoding="utf-8"))["hechos"].items() if len(v) >= NEED}
    if len(hechos) < MIN_TYPES:
        raise SystemExit(f"solo {len(hechos)} tipos con {NEED} hechos: B23 necesita {MIN_TYPES}")
    ev, pool = split(hechos)
    all_ev = [x | {"tipo": t} for t in sorted(ev) for x in ev[t]]
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    res |= {"config": {"n_train": N_TRAIN, "n_eval": N_EVAL, "anclas": ANCHORS, "train_from": TRAIN_FROM, "tipos": sorted(hechos)}, "evaluados": [{k: x[k] for k in ("tipo", "propiedad", "cabeza")} for x in all_ev]}
    model, tok = load(model, tok, True)
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    base = {"antes": candidate_losses(model, tok, all_ev), "H": states(model, tok, all_ev)}
    res["antes"], res["gana_antes"] = base["antes"][:, 0].round(4).tolist(), wins(base["antes"]).astype(int).tolist()
    return hechos, ev, pool, all_ev, res, model, tok, state, base


def consolidacion(path: str, facts_path: str, arms: str = ",".join(ARMS4), controls: str = "99", model=None, tok=None) -> None:
    hechos, ev, pool, all_ev, res, model, tok, state, base = prepare(path, facts_path, model, tok)
    arms = tuple(arms.split(","))
    assert all(a in b21.ARMS or a == "conf24" for a in arms), arms
    res["config"] |= {"brazos": list(arms), "conjuntos": list(SETS4), "n_uso": b21.N_USE}
    for t, s, arm in plan4(sorted(hechos), arms, int(controls)):
        key = f"{t}|s{s}|{arm}"
        if key in res["corridas"] or b21.given_up(res["corridas"], arm):
            continue
        probe = None if arm == "control" else Protect(arm, seed=sum(map(ord, t)) + s)  # hash() cambia entre procesos
        res["corridas"][key] = r = run_set(model, tok, t, s, 0, ev, pool, all_ev, base, state, probe)
        print(json.dumps({"corrida": key, "nuevos": round(r["nuevos_final"], 3), "subida_media": round(float(np.mean(r["subida"])), 3), "pasos": r["pasos"],
                          "fuga": r.get("fuga"), "segundos": r["segundos"]}), flush=True)  # fmt: skip
        why = gate(r, all_ev) if len(res["corridas"]) == 1 else None
        res["corte"] = why
        json.dump(res, open(path, "w"), ensure_ascii=False)
        if why:
            print(json.dumps({"corte": why}), flush=True)
            return


def gradiente(path: str, facts_path: str, sets: str = "0,1,2", repeats: str = "4", model=None, tok=None) -> None:
    hechos, ev, pool, all_ev, res, model, tok, state, base = prepare(path, facts_path, model, tok)
    ids = [int(s) for s in sets.split(",")]
    plan = [(t, s, 0) for s in ids for t in sorted(hechos)] + [(t, ids[0], 1) for t in sorted(hechos)[: int(repeats)]]
    for t, s, order in plan:
        key = f"{t}|s{s}" + (f"|r{order}" if order else "")
        if key in res["corridas"]:
            continue
        res["corridas"][key] = r = run_set(model, tok, t, s, order, ev, pool, all_ev, base, state)
        print(json.dumps({"corrida": key, "entrenados": round(r["subida_entrenados_verdad"], 3), "nuevos": round(r["nuevos_final"], 3), "subida_media": round(float(np.mean(r["subida"])), 3), "pasos": r["pasos"], "segundos": r["segundos"]}), flush=True)
        why = gate(r, all_ev) if len(res["corridas"]) == 1 else None
        res["corte"] = why
        json.dump(res, open(path, "w"), ensure_ascii=False)
        if why:
            print(json.dumps({"corte": why}), flush=True)
            return


if __name__ == "__main__":
    {"seleccion": seleccion, "gradiente": gradiente, "consolidacion": consolidacion}[sys.argv[1]](*sys.argv[2:])
