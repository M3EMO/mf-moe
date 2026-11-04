import json

import numpy as np
import torch
from scipy.stats import spearmanr

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b19_gradientes as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_first_order_predicts_real_rise_and_blocks_add_up(tmp_path, monkeypatch):
    """Con pasos chicos, ⟨∇L_c, Δθ⟩ tiene que seguir a la subida real: si el signo o los pesos estuvieran mal, no."""
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    monkeypatch.setattr(b, "N_SETS", 1)
    monkeypatch.setattr(b18, "gate", lambda r: None)
    for k, v in {"N_TRAIN": 4, "N_FAR": 2, "ANCHORS": 8, "MAX_IN": 30}.items():
        monkeypatch.setattr(b18, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2, "LR": 1e-3}.items():
        monkeypatch.setattr(b17, k, v)
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "b19.json")
    b.main(out, "reptile.n.01", model=model, tok=chars)
    r = json.load(open(out))["areas"]["reptile.n.01|s0"]
    idx = [r["evaluados"].index(n) for n in r["evaluados_gradiente"]]
    rise = np.array(r["subida"])[idx]
    n = len(idx)
    assert n >= 10 and all(len(r[k]) == n for k in ("primer_orden", "ntk", "uso_def"))
    assert spearmanr(r["primer_orden"], rise)[0] > 0.5
    parts = np.array(list(r["primer_orden_bloques"].values())).sum(0)
    assert np.allclose(parts, r["primer_orden"])
    assert np.abs(r["primer_orden_bloques"]["expertos"]).sum() > 0 and np.abs(r["primer_orden_bloques"]["router"]).sum() > 0
    assert all(0 <= x <= 1.0001 for x in r["uso_def"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
