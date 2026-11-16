import json
import os

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b27_puerta as b27
from colab import b30_fuentes as b30
from colab import b35_modelo as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_scaled_keeps_granite_1b_and_moves_layers_to_the_same_relative_depth():
    assert b.scaled(b27.LAYERS, 24) == b27.LAYERS and b.scaled(b27.LAYERS, 16) == (5, 8, 11, 13) and b.scaled((6,), 32) == (8,)


def test_run_builds_the_bank_then_runs_gate_and_threshold_and_resumes_from_the_joined_file(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b27, "LAYERS", (1, 2))
    monkeypatch.setattr(b30, "DOSES", (1, 2))
    monkeypatch.setattr(b.b29.b25, "K_FILTER", 3)
    monkeypatch.setattr(b.b29.b23, "wins", lambda L: np.ones(len(L), bool))  # el modelo diminuto no conoce nada
    hechos = {"hechos": {"t": [{"propiedad": "p", "cabeza": f"H{i}", "valor": f"V{i}"} for i in range(30)]}}
    cands = [{"tipo": "t", "propiedad": "p", "cabeza": "H0", "valor": "V0"}, {"tipo": "t", "propiedad": "p", "cabeza": "Z", "valor": "V1"}]
    hp, cp, out = (str(tmp_path / f) for f in ("h.json", "c.json", "r.json"))
    json.dump(hechos, open(hp, "w"))
    json.dump(cands, open(cp, "w"))
    monkeypatch.setattr(b, "HECHOS", hp)
    monkeypatch.setattr(b, "CANDS", cp)
    model = granite.tiny()
    b.run(out, model, chars)
    res = json.load(open(out))
    assert set(res) == set(b.STAGES) and len(res["banco"]) == 28
    assert sorted(res["puerta"]["corridas"]) == sorted(f"s{s}|{a}|d10" for s in range(2) for a in ("oraculo", "azar", "sonda"))
    assert sorted(res["umbral"]["corridas"]) == sorted(f"s{s}|{a}|{g}|d{d}" for s in range(2) for a, g in b.ARMS_U for d in (1, 2))
    for k in b.STAGES:  # una VM nueva solo tiene `out`
        os.remove(out.replace(".json", f"_{k}.json"))
    b.run(out, model, chars)
    assert json.load(open(out)) == res
