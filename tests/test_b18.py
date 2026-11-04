import json

import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def small(monkeypatch):
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    for k, v in {"N_SETS": 2, "N_TRAIN": 4, "N_FAR": 3, "ANCHORS": 8, "MAX_IN": 5}.items():
        monkeypatch.setattr(b, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2}.items():
        monkeypatch.setattr(b17, k, v)


def test_evaluation_is_fixed_and_repeat_keeps_the_trained_set(tmp_path, monkeypatch):
    small(monkeypatch)
    monkeypatch.setattr(b, "gate", lambda r: None)
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "b18.json")
    b.main(out, "reptile.n.01", model=model, tok=chars)
    R = json.load(open(out))["areas"]
    s0, s1, r1 = R["reptile.n.01|s0"], R["reptile.n.01|s1"], R["reptile.n.01|s0|r1"]
    assert s0["evaluados"] == s1["evaluados"] == r1["evaluados"]
    assert s0["entrenados"] != s1["entrenados"] and s0["entrenados"] == r1["entrenados"]
    assert not set(s0["entrenados"]) & set(s0["evaluados"])
    assert not set(s0["donantes"]) & set(s0["evaluados"] + s0["entrenados"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_gate_cuts_after_first_run(tmp_path, monkeypatch):
    small(monkeypatch)
    out = str(tmp_path / "b18.json")
    b.main(out, "reptile.n.01", model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    assert res["corte"].startswith("C0") and list(res["areas"]) == ["reptile.n.01|s0"]
