import json

import torch

from colab import b16_conceptos as b
from mf_moe import granite


def chars(text: str) -> dict:
    """Tokenizador de a carácter: el prefijo tokenizado es prefijo del texto completo, como en el real."""
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_smoke_restores_model_and_records_predictors(tmp_path, monkeypatch):
    for k, v in {"N_TRAIN": 4, "N_BIN": 2, "STATE_LAYER": 1, "STEPS": 2, "BATCH": 2}.items():
        monkeypatch.setattr(b, k, v)
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "b16.json")
    b.main(out, "dog.n.01", model=model, tok=chars)
    r = json.load(open(out))["areas"]["dog.n.01"]
    n = len(r["evaluados"])
    assert 2 <= n <= 8 and not set(r["evaluados"]) & set(r["entrenados"])
    assert len(r["subida"]) == n and all(len(v) == n for v in r["predictores"].values())
    assert all(0 <= x <= 1 for x in r["predictores"]["router"])
    assert any(abs(x) > 0 for x in r["subida"])  # entrenar movió algo
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # cada área parte del mismo modelo
