import json

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b20_matriz as b20
from colab import b21_analisis as a
from colab import b21_contencion as b
from mf_moe import granite
from mf_moe.model import moe


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_rules_pick_the_right_experts():
    new = np.array([[0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
    old = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.3, 0.5]])
    rng = np.random.default_rng(0)
    assert b.frozen_rows("control", new, old, rng) == [[]]
    assert b.frozen_rows("cont8", new, old, rng) == [[2, 3, 4, 5, 6, 7]]  # entrenan los 2 que más usa lo nuevo
    assert b.frozen_rows("cons24", new, old, rng) == [[6, 7]]  # se congela lo que más usa lo viejo
    topic = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1]])
    flat = np.full_like(old, 0.125)
    free = set(range(8)) - set(b.frozen_rows("esp8", new, flat, rng, topic)[0])
    assert 6 in free and len(free) == 2  # entra el experto que el tema usa más que el resto
    assert len(b.frozen_rows("azar24", new, old, rng)[0]) == 2 and len(b.frozen_rows("azar8", new, old, rng)[0]) == 6
    swapped = b.frozen_rows("cons8", old, new, rng)  # con usos invertidos se ve que `viejo` no mira lo nuevo
    assert swapped == [[0, 1, 2, 3, 4, 5]]


def run_arm(monkeypatch, tmp_path, arm: str):
    """Corre un brazo en un Granite chico y devuelve el modelo, los pesos iniciales, los pesos entrenados y la salida."""
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    monkeypatch.setattr(b, "TRAIN_FROM", 0)
    monkeypatch.setattr(b18, "gate", lambda r: None)
    monkeypatch.setattr(b20, "TOPICS", {"reptile.n.01": "animal", "fungus.n.01": "planta"})
    monkeypatch.setattr(b20, "N_EVAL", 3)
    monkeypatch.setattr(b, "N_USE", 8)
    monkeypatch.setattr(b, "plan", lambda: [("reptile.n.01", 0, arm)])
    for k, v in {"N_TRAIN": 4, "ANCHORS": 8}.items():
        monkeypatch.setattr(b18, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2, "LR": 1e-2}.items():
        monkeypatch.setattr(b17, k, v)
    model = granite.tiny()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    moved: dict = {}
    original = b.Mask.post

    def spy(self, m, state):
        moved.update({k: v.detach().clone() for k, v in m.named_parameters()})
        original(self, m, state)

    monkeypatch.setattr(b.Mask, "post", spy)
    out = str(tmp_path / "b21.json")
    b.main(out, model=model, tok=chars)
    return model, before, moved, json.load(open(out))["areas"][f"reptile.n.01|s0|{arm}"]


def test_frozen_rows_do_not_move_and_the_rest_do(tmp_path, monkeypatch):
    model, before, moved, r = run_arm(monkeypatch, tmp_path, "cont8")
    assert r["fuga"] == 0.0 and r["brazo"] == "cont8"
    names = {id(p): n for n, p in model.named_parameters()}
    for layer, rows in enumerate(r["congelados"]):
        free = sorted(set(range(6)) - set(rows))
        experts, router = moe(model, layer)
        for p in (experts.gate_up_proj, experts.down_proj, router.weight):
            k = names[id(p)]
            assert torch.equal(before[k][rows], moved[k][rows])
            assert not torch.equal(before[k][free], moved[k][free])
    assert all(torch.equal(before[k], v) for k, v in model.named_parameters())  # el modelo vuelve al estado inicial
    assert all(p.requires_grad for p in model.parameters())
    assert not moe(model, 0)[0].gate_up_proj._backward_hooks  # el congelado no sobrevive a la corrida


def test_trunk_arm_only_moves_experts(tmp_path, monkeypatch):
    model, before, moved, r = run_arm(monkeypatch, tmp_path, "cont8_tronco")
    assert r["fuga"] == 0.0
    still = [k for k in before if ".experts." not in k]
    assert still and all(torch.equal(before[k], moved[k]) for k in still)
    assert any(not torch.equal(before[k], moved[k]) for k in before if ".experts." in k)
    assert all(p.requires_grad for p in model.parameters())  # lo congelado se devuelve


def test_topic_arm_uses_the_topic_and_records_its_reach(tmp_path, monkeypatch):
    model, before, moved, r = run_arm(monkeypatch, tmp_path, "esp8")
    assert r["fuga"] == 0.0 and all(len(rows) == 4 for rows in r["congelados"]) and 0 <= r["uso_entrenable"] <= 1.0001


def test_sensitivity_arm_freezes_what_breaks_old_knowledge(tmp_path, monkeypatch):
    model, before, moved, r = run_arm(monkeypatch, tmp_path, "prec24")
    assert r["fuga"] == 0.0 and all(len(rows) == 2 for rows in r["congelados"])  # 6 expertos: entrenan 4
    items = [("A cat is", " a small animal."), ("A dog is", " a loyal animal.")]
    S = b.sensitivity(model, chars, items, range(2))
    assert S.shape == (2, 6) and (S >= 0).all() and S.sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_plan_keeps_topics_paired():
    p = b.plan()
    assert sum(arm == "control" for *_, arm in p) == 5
    arms = [x for x in b.ARMS if x != "control"]
    assert p[5 : 5 + len(arms)] == [("fish.n.01", 0, x) for x in arms] and "esp8" in arms and "cont24" not in arms
    assert len(p) == 5 + len(arms) * 10 * len(b.SETS)


def test_giving_up_only_after_failing_every_time():
    fail, good = {"nuevos_final": 2.0}, {"nuevos_final": 0.3}
    assert not b.given_up({"a|s0|cons8": fail, "b|s0|cons8": fail}, "cons8")
    assert b.given_up({"a|s0|cons8": fail, "b|s0|cons8": fail, "c|s0|cons8": fail}, "cons8")
    assert not b.given_up({"a|s0|cons8": fail, "b|s0|cons8": good, "c|s0|cons8": fail}, "cons8")
    assert not b.given_up({"a|s0|cons8": fail, "b|s0|cons8": fail, "c|s0|cons8": fail}, "azar8")


def test_holm_and_verdict():
    assert a.holm({"x": 0.01, "y": 0.04}) == {"x": 0.02, "y": 0.04}
    rng = np.random.default_rng(0)
    noisy = lambda m: m + rng.normal(0, 0.01, 10)  # noqa: E731
    cell = lambda v: {k: v for k in ("total", "tema", "super", "otro", "fuera")}  # noqa: E731
    high, zero = noisy(0.10), rng.normal(0, 0.03, 10)
    T = {"esp8": cell(noisy(0.20)), "azar8": cell(high), "cont8": cell(noisy(0.10)), "cons24": cell(zero), "azar24": cell(rng.normal(0, 0.03, 10)), "cont8_tronco": cell(noisy(0.40))}
    ok = dict.fromkeys(T, 1.0)
    v = a.verdict(T, ok)
    assert v["C_pos"] and v["H_esp"] and v["D_esp"] and not v["D_pop"] and not v["H_cons24"] and not v["D_cons"] and v["D_tronco"]
    T["cont8"] = cell(noisy(-0.10))  # una diferencia en contra también cuenta como diferencia
    assert a.verdict(T, ok)["D_pop"]
    ok["esp8"] = 0.5  # no aprendió: no gana por dañar menos
    v = a.verdict(T, ok)
    assert not v["H_esp"] and not v["D_esp"]


def test_table_pairs_each_arm_with_its_control():
    tops = {"t1": "p", "t2": "p", "t3": "q"}
    ev = {"t1": ["a"], "t2": ["b"], "t3": ["c"]}
    runs = {"t1|s0|cont8": {"subida": [0.5, 0.5, 0.5], "evaluados": ["a", "b", "c"]}, "t1|s0|control": {"subida": [1.0, 1.0, 1.0], "evaluados": ["a", "b", "c"]}}
    T = a.table({"areas": runs, "temas": tops, "evaluacion": ev}, {"areas": {"t1|s0": {"subida": [1.0, 0.75, 0.75]}}})
    assert np.allclose(T["cont8"]["tema"], 0.5) and np.allclose(T["cont8"]["super"], 0.25) and np.allclose(T["cont8"]["otro"], 0.25)
    assert "control" not in T
