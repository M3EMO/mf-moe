import json

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b21_contencion as b21
from colab import b23_uniedit as b
from mf_moe import granite
from mf_moe.model import moe


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def row(head: str, prop: str, value: str, cls: str = "Q5", datatype: str = "wikibase-item", single: bool = True) -> dict:
    return {"head_entity": {"id": head, "label": head, "P31": [cls]}, "property": {"id": prop, "label": prop, "single_value": single},
            "tail_entity": {"datatype": datatype, "value": {"id": value, "label": value}}}  # fmt: skip


def facts(n: int, props: list[str]) -> list[dict]:
    return [{"propiedad": props[i % len(props)], "cabeza": f"c{i}", "valor": f"v{i}", "donantes": ["a", "b", "c", "d"], "cabeza_ajena": "z"} for i in range(n)]


def small(monkeypatch):
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    for k, v in {"NEED": 16, "N_TRAIN": 2, "N_EVAL": 4, "ANCHORS": 8, "MIN_CELL": 1, "MIN_TYPES": 3, "TOP_COS": 1}.items():
        monkeypatch.setattr(b, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2}.items():
        monkeypatch.setattr(b17, k, v)


def test_candidates_keep_one_valid_fact_per_head():
    rows = [row("h", "p1", f"x{i}") for i in range(1)] + [row(f"g{i}", "p1", f"y{i}") for i in range(6)]
    rows += [row("h", "p2", f"w{i}") for i in range(1)] + [row(f"k{i}", "p2", f"u{i}") for i in range(6)]
    rows += [row("sin_clase", "p1", "y0", cls="Q999"), row("texto", "p1", "t", datatype="string"), row("multi", "p1", "m", single=False)]
    out = b.candidates(rows)
    assert list(out) == ["persona"]
    heads = [x["cabeza"] for x in out["persona"]]
    assert len(heads) == len(set(heads)) and "h" in heads and not {"sin_clase", "texto", "multi"} & set(heads)
    for x in out["persona"]:
        assert len(x["donantes"]) == b.K and x["valor"] not in x["donantes"] and x["cabeza_ajena"] != x["cabeza"]


def test_half_leaves_both_cells_and_enough_to_train(monkeypatch):
    monkeypatch.setattr(b, "N_TRAIN", 3)
    monkeypatch.setattr(b, "MIN_CELL", 2)
    pool, ev = facts(12, list("abcd")), facts(8, list("abcd"))
    h = b.half(pool, ev, "persona", 0)
    assert h == b.half(pool, ev, "persona", 0) and len(h) == 2
    assert sum(x["propiedad"] in h for x in pool) >= 3 and 2 <= sum(x["propiedad"] in h for x in ev) <= 6


def test_anchors_avoid_the_trained_half_and_the_trained_type(monkeypatch):
    monkeypatch.setattr(b, "ANCHORS", 8)
    pool = {t: [x | {"cabeza": f"{t}{x['cabeza']}"} for x in facts(20, list("abcd"))] for t in ("t1", "t2", "t3")}
    out = b.anchors(pool, "t1", 0, {"a", "b"})
    assert len(out) == 8 and all(x["propiedad"] in "cd" and not x["cabeza"].startswith("t1") for x in out)


def test_screen_keeps_known_not_floor_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "NEED", 1)
    monkeypatch.setattr(b, "CHUNK", 2)
    # el valor real gana con la cabeza propia en c0 y c1, y solo c1 también gana con la ajena: c0 pasa, c1 es piso, c2 no se conoce
    loss = {"The p of c0 is": 0, "The p of c1 is": 0, "The p of c2 is": 9, "The p of z is": 5, "The p of y is": 0}
    calls = []

    def fake(model, tok, items):
        calls.append(len(items))
        return np.array([(loss[p] if " v" in r else 1.0) for p, r in items])

    monkeypatch.setattr(b17, "def_losses", fake)
    pool = {"persona": [x | {"propiedad": "p"} for x in facts(2, ["p"])] + [{"propiedad": "p", "cabeza": "c2", "valor": "v2", "donantes": list("abcd"), "cabeza_ajena": "z"}]}
    pool["persona"][1]["cabeza_ajena"] = "y"
    out = str(tmp_path / "s.json")
    res = b.screen(None, None, pool, out)
    assert [x["cabeza"] for x in res["hechos"]["persona"]] == ["c0"] and res["vistos"]["persona"] == 2
    n = len(calls)
    assert b.screen(None, None, pool, out)["vistos"]["persona"] == 2 and len(calls) == n  # ya llegó a NEED: no vuelve a medir


def test_gradient_keeps_eval_fixed_disjoint_and_restores_model(tmp_path, monkeypatch):
    small(monkeypatch)
    monkeypatch.setattr(b, "gate", lambda r, e: None)
    hechos = {t: [x | {"cabeza": f"{t}{x['cabeza']}"} for x in facts(16, list("abcd"))] for t in ("t1", "t2", "t3")}
    src, out = tmp_path / "h.json", str(tmp_path / "g.json")
    src.write_text(json.dumps({"hechos": hechos}))
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    b.gradiente(out, str(src), "0,1", "1", model=model, tok=chars)
    res = json.load(open(out))
    R = res["corridas"]
    assert len(R) == 3 * 2 + 1 and len(res["evaluados"]) == 12
    s0, s1, r1 = R["t1|s0"], R["t1|s1"], R["t1|s0|r1"]
    ev = {x["cabeza"] for x in res["evaluados"]}
    assert s0["entrenados"] == r1["entrenados"] and s0["mitad"] == r1["mitad"] and not set(s0["entrenados"]) & ev
    assert len(s0["subida"]) == len(s0["coseno"]) == len(s0["gana_despues"]) == 12
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_conflict_is_a_per_expert_inner_product():
    model = granite.tiny()
    items = [("A cat is", " a small animal."), ("A dog is", " a loyal animal.")]
    D = b.conflict(model, chars, items, items, range(2))
    assert D.shape == (2, 6) and (D >= 0).all() and D.sum() > 0  # consigo mismo, cada producto es una norma al cuadrado
    assert all(p.grad is None for p in model.parameters())


def test_consolidation_pairs_arms_freezes_like_cons24_and_restores(tmp_path, monkeypatch):
    small(monkeypatch)
    monkeypatch.setattr(b, "gate", lambda r, e: None)
    monkeypatch.setattr(b21, "TRAIN_FROM", 0)
    monkeypatch.setattr(b21, "N_USE", 4)
    monkeypatch.setattr(b21, "GIVE_UP", 99)  # con 2 pasos el modelo chico no aprende y los brazos se abandonarían
    monkeypatch.setattr(b17, "LR", 1e-2)
    hechos = {t: [x | {"cabeza": f"{t}{x['cabeza']}"} for x in facts(16, list("abcd"))] for t in ("t1", "t2", "t3")}
    src, out = tmp_path / "h.json", str(tmp_path / "c.json")
    src.write_text(json.dumps({"hechos": hechos}))
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    b.consolidacion(out, str(src), model=model, tok=chars)
    R = json.load(open(out))["corridas"]
    assert len(R) == 3 + 3 * 2 * 3 and "brazo" not in R["t1|s0|control"]  # el control no congela
    conf, cons = R["t2|s1|conf24"], R["t2|s1|cons24"]
    assert conf["brazo"] == "conf24" and conf["fuga"] == 0.0 and 0 <= conf["negativos"] <= 1
    assert [len(x) for x in conf["congelados"]] == [len(x) for x in cons["congelados"]] == [2, 2]  # 6 expertos: entrenan 4
    assert conf["entrenados"] == cons["entrenados"] == R["t2|s1|azar24"]["entrenados"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.requires_grad for p in model.parameters()) and not moe(model, 0)[0].gate_up_proj._backward_hooks


def test_gate_uses_other_type_other_relation_as_far_band():
    ev = [{"tipo": "a", "propiedad": "p"}, {"tipo": "b", "propiedad": "q"}]
    r = {"tipo": "a", "mitad": ["p"], "subida": [9.0, 0.1], "nuevos_final": 0.1, "subida_entrenados_verdad": 1.0, "segundos": 10}
    assert b.gate(r, ev) is None
    assert b.gate(r | {"subida": [9.0, 0.6]}, ev).startswith("C0")
    assert b.gate(r | {"nuevos_final": 2.0}, ev).startswith("C0")
