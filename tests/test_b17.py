import json

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_padding_does_not_change_per_sentence_loss():
    model = granite.tiny()
    short, long = ("A cat is", " small."), ("A dog is", " a domestic animal that barks a lot.")
    alone = b.def_losses(model, chars, [short])[0]
    padded = b.def_losses(model, chars, [short, long])[0]
    assert np.isclose(alone, padded, atol=2e-2)  # bf16 en autocast: tolerancia de redondeo


def test_smoke_cuts_at_gate_and_restores_model(tmp_path, monkeypatch):
    for k, v in {"N_TRAIN": 4, "N_BIN": 2, "STATE_LAYER": 1}.items():
        monkeypatch.setattr(b16, k, v)
    for k, v in {"MAX_STEPS": 3, "CHECK_EVERY": 1, "BATCH": 2}.items():
        monkeypatch.setattr(b, k, v)
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "b17.json")
    b.main(out, "dog.n.01,fish.n.01", model=model, tok=chars)
    res = json.load(open(out))
    r = res["areas"]["dog.n.01"]
    assert res["corte"].startswith("C0") and "fish.n.01" not in res["areas"]  # el Granite chico no aprende en 3 pasos
    assert not set(r["donantes"]) & set(r["evaluados"] + r["entrenados"])
    n = len(r["evaluados"])
    assert all(len(v) == n for v in [*r["predictores"].values(), *r["covariables"].values()])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
