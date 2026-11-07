import json

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b20_matriz as b20
from colab import b22_analisis as a
from colab import b22_profundidad as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_only_the_last_layers_move_and_the_model_comes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    monkeypatch.setattr(b18, "gate", lambda r: None)
    monkeypatch.setattr(b20, "TOPICS", {"reptile.n.01": "animal", "fungus.n.01": "planta"})
    monkeypatch.setattr(b20, "N_EVAL", 3)
    monkeypatch.setattr(b, "plan", lambda: [("reptile.n.01", 0, "p1")])
    for k, v in {"N_TRAIN": 4, "ANCHORS": 8}.items():
        monkeypatch.setattr(b18, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2, "LR": 1e-2}.items():
        monkeypatch.setattr(b17, k, v)
    model = granite.tiny()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    moved: dict = {}
    original = b.Depth.post

    def spy(self, m, state):
        moved.update({k: v.detach().clone() for k, v in m.named_parameters()})
        original(self, m, state)

    monkeypatch.setattr(b.Depth, "post", spy)
    out = str(tmp_path / "b22.json")
    b.main(out, model=model, tok=chars)
    last = len(model.model.layers) - 1
    prefix = f"model.layers.{last}."
    assert any(not torch.equal(before[k], moved[k]) for k in before if k.startswith(prefix))
    assert all(torch.equal(before[k], moved[k]) for k in before if ".layers." in k and not k.startswith(prefix))
    assert all(torch.equal(before[k], v) for k, v in model.named_parameters())  # el modelo vuelve al estado inicial
    assert all(p.requires_grad for p in model.parameters())  # lo congelado se devuelve
    assert json.load(open(out))["areas"]["reptile.n.01|s0|p1"]["capas"] == 1


def test_plan_repeats_b20_once_and_pairs_the_depths():
    p = b.plan(["x", "y"])
    assert p == [("x", 0, "p18"), ("x", 0, "p4"), ("x", 0, "p1"), ("y", 0, "p4"), ("y", 0, "p1")]


def world(depth_rho: float, learned: float = 0.3, n: int = 5):
    """Temas con 60 conceptos cada uno: `depth_rho` alto copia a B20; bajo, lo reemplaza por ruido."""
    rng = np.random.default_rng(0)
    tops = [f"t{i}" for i in range(n)]
    ev = {t: [f"{t}x{j}" for j in range(60)] for t in tops}
    names = [x for t in tops for x in ev[t]]
    R20, R22 = {}, {}
    for t in tops:
        base = rng.normal(0, 0.2, len(names)) + np.array([0.5 * (x.startswith(t)) for x in names])
        R20[f"{t}|s0"] = {"subida": base.tolist()}
        y = base if depth_rho > 0.5 else rng.normal(0, 0.2, len(names))
        R22[f"{t}|s0|p1"] = {"subida": y.tolist(), "raiz": [0.0] * len(names), "evaluados": names, "nuevos_final": learned, "subida_entrenados_verdad": 3.0}
    return {"areas": R22, "evaluacion": ev}, {"areas": R20}


def test_a_depth_is_valid_only_when_it_keeps_the_bank():
    assert a.verdict(*world(1.0))["p1"]["valida"]
    assert not a.verdict(*world(0.0))["p1"]["valida"]  # sin parecido con B20 ni gradiente por tema
    assert not a.verdict(*world(1.0, learned=2.0))["p1"]["valida"]  # no aprende: no sirve aunque el daño se parezca
    assert not a.verdict(*world(1.0, n=2))["p1"]["valida"]  # con dos temas no hay conclusión
