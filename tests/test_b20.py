import json

import numpy as np
import torch

from colab import b16_conceptos as b16
from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b20_analisis as a
from colab import b20_matriz as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def fast(monkeypatch):
    for k, v in {"N_SEEDS": 15, "INNER": 2, "RANKS": (2, 5), "NUGGETS": (1.0,), "NEIGHBORS": (3,)}.items():
        monkeypatch.setattr(a, k, v)


def world(name: str, seed: int = 0):
    return a.synthetic(name, np.random.default_rng(seed), n_topics=6, per_topic=30, sets=3)


def test_map_with_the_true_structure_wins(monkeypatch):
    """Si el mundo es por bloques, la covarianza correcta tiene que predecir; en el nulo, ninguna."""
    fast(monkeypatch)
    res = a.evaluate(*world("bloques"), draws=2)
    assert res["global"]["bloques"].mean() > 0.2
    null = a.evaluate(*world("nulo"), draws=2)
    assert all(abs(v.mean()) < 0.1 for v in null["global"].values())


def test_tree_cov_is_symmetric_and_psd():
    R = a.residuals(np.random.default_rng(0).normal(size=(40, 12)))
    K = a.tree_cov(R)
    assert np.allclose(K, K.T) and np.linalg.eigvalsh(K).min() > -1e-8


def test_signal_matches_the_share_of_repeatable_interaction():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(200, 4))
    R = {f"t|s{i}": {"subida": shared[:, i].tolist()} for i in range(4)}
    R["t|s0|r1"] = {"subida": (shared[:, 0]).tolist()}  # repetición idéntica: fiabilidad 1
    assert abs(a.signal(R) - 0.75) < 0.1  # con 4 conjuntos independientes, 3/4 de la varianza es interacción
    R["t|s0|r1"] = {"subida": rng.normal(size=200).tolist()}  # repetición sin relación: sin señal
    assert a.signal(R) < 0.08  # regresión: una base sin el conjunto daba fiabilidad 0,25 sin relación alguna


def test_verdict_needs_every_comparison():
    n = 10
    high, low = np.linspace(0.3, 0.4, n), np.linspace(-0.05, 0.05, n)
    res = {"global": {"arbol": high, "plano": low, "knn": low, "wordnet": low, "bloques": high}}
    v = a.verdict(res)
    assert v["H_estructura"] is False and not v["H_arbol"]
    res["global"]["plano"] = high
    res["global"]["bloques"] = low
    v = a.verdict(res)
    assert v["H_estructura"] and not v["H_arbol"]
    res["global"]["plano"] = low
    assert a.verdict(res)["H_arbol"]


def test_matrix_keeps_evaluation_fixed_and_repeats_the_set(tmp_path, monkeypatch):
    monkeypatch.setattr(b16, "STATE_LAYER", 1)
    monkeypatch.setattr(b18, "gate", lambda r: None)
    monkeypatch.setattr(b, "TOPICS", {"reptile.n.01": "animal", "fungus.n.01": "planta"})
    monkeypatch.setattr(b, "N_SETS", 3)
    monkeypatch.setattr(b, "STAGE1", 2)
    monkeypatch.setattr(b, "signal", lambda R: 0.0)
    monkeypatch.setattr(b, "N_EVAL", 3)
    for k, v in {"N_TRAIN": 4, "ANCHORS": 8}.items():
        monkeypatch.setattr(b18, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 2}.items():
        monkeypatch.setattr(b17, k, v)
    model = granite.tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "b20.json")
    b.main(out, model=model, tok=chars)
    R = json.load(open(out))["areas"]
    assert set(R) == {"reptile.n.01|s0", "reptile.n.01|s1", "reptile.n.01|s0|r1", "fungus.n.01|s0", "fungus.n.01|s1", "fungus.n.01|s0|r1"}
    assert len({tuple(r["evaluados"]) for r in R.values()}) == 1 and len(R["reptile.n.01|s0"]["evaluados"]) == 6
    assert R["reptile.n.01|s0"]["entrenados"] == R["reptile.n.01|s0|r1"]["entrenados"] != R["reptile.n.01|s1"]["entrenados"]
    assert not set(R["fungus.n.01|s0"]["entrenados"]) & set(R["fungus.n.01|s0"]["evaluados"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert json.load(open(out))["corte"].startswith("señal")  # sin señal no corre la segunda etapa
    monkeypatch.setattr(b, "signal", lambda R: 1.0)
    b.main(out, model=model, tok=chars)
    R = json.load(open(out))["areas"]
    assert "reptile.n.01|s2" in R and "fungus.n.01|s2" in R and json.load(open(out))["corte"] is None
    d = a.from_json(out)
    assert d["Z"].shape == (6, 6) and len(d["C2"]) == 2 and d["W"].shape == (6, 6)
