import json

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b27_puerta as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_weighted_loss_with_unit_weights_is_the_hf_loss():
    model = granite.tiny()
    ids, mask, labels = b17.batch(chars, [("The p of Ka is", " Lo."), ("Vi is the q of", " Dremor.")])
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.allclose(b24.weighted_loss(out.logits, labels, torch.ones(2)), out.loss, atol=1e-5)


def test_gate_runs_oracle_and_random_on_every_set_and_uniform_on_set_zero(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2, "DOSES": (1, 2)}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    bank, out = str(tmp_path / "banco.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    b.main(out, bank, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    w, r = np.array(res["pesos"]["oraculo"]), np.array(res["pesos"]["azar"])
    assert abs(w.mean() - 1) < 1e-3 and sorted(w) == sorted(r)
    assert (np.argsort(w) == np.argsort(res["base"]["perdida_ida"])).all()  # más duda, más peso
    expected = [f"s0|uniforme|d{d}" for d in (1, 2)] + [f"s{s}|{a}|d{d}" for s in range(2) for a in ("oraculo", "azar") for d in (1, 2)]
    assert sorted(res["corridas"]) == sorted(expected)
    assert all(len(x["perdida_nuevo"]) == len(x["sobrescritos"]) for x in res["corridas"].values())


def test_b27b_signal_ignores_the_values_and_runs_both_arms(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2, "DOSES": (1, 2)}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    model = granite.tiny()
    swapped = [x | {"valor": "Otro"} for x in facts]
    assert np.allclose(b.own_confidence(model, chars, facts), b.own_confidence(model, chars, swapped))  # solo la pregunta
    bank, out = str(tmp_path / "banco.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    b.main_b(out, bank, model=model, tok=chars)
    res = json.load(open(out))
    strong = np.array(res["pesos"]["oraculo_fuerte"])
    assert abs(strong.mean() - 1) < 1e-3 and abs(strong.min() - 0.02) < 1e-6
    assert sorted(res["corridas"]) == sorted(f"s{s}|{a}|d{d}" for s in range(2) for a in ("senal", "oraculo_fuerte") for d in (1, 2))


def test_ridge_recovers_a_linear_target_and_cross_fit_never_sees_its_own_group():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 200))
    y = X[:, :5].sum(1) + 0.1 * rng.normal(size=60)
    p = b.ridge(X[:40], y[:40], X[40:])
    assert np.corrcoef(p, y[40:])[0, 1] > 0.5
    groups = np.repeat([-1, 0, 1], 20)
    a = b.cross_fit(X, y, groups)
    y2 = y.copy()
    y2[groups == 0] = 100.0  # etiquetas del grupo 0 cambiadas: su predicción no puede moverse
    assert np.allclose(a[groups == 0], b.cross_fit(X, y2, groups)[groups == 0]) and np.isnan(a[groups == -1]).all()


def test_b27c_runs_the_probe_arm_with_cross_fitted_weights(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2, "DOSES": (1, 2)}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "LAYERS", (1, 2))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    cands = [{"propiedad": "p", "cabeza": f"X{i}", "valor": f"W{i}"} for i in range(8)] + facts[:4]
    bank, cand, out = (str(tmp_path / f) for f in ("banco.json", "cand.json", "r.json"))
    json.dump(facts, open(bank, "w"))
    json.dump(cands, open(cand, "w"))
    b.main_c(out, bank, cand, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    assert len(res["sonda"]["prediccion"]) == 12 and np.isfinite(res["sonda"]["prediccion"]).all()
    assert abs(np.mean(res["pesos"]["sonda"]) - 1) < 1e-3 and res["config"]["n_sonda"] == 20
    assert sorted(res["corridas"]) == sorted(f"s{s}|sonda|d{d}" for s in range(2) for d in (1, 2))
