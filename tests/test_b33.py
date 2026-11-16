import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b27_puerta as b27
from colab import b33_sonda2 as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_noisy_hits_the_target_spearman():
    sig = np.random.default_rng(0).standard_normal(206)
    _, rho = b.noisy(sig, 0.616)
    assert abs(rho - 0.616) < 0.01


def test_b33_selects_on_candidates_and_runs_both_arms_with_the_b25_sets(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b24, "DOSES", (1, 2))
    monkeypatch.setattr(b27, "LAYERS", (2,))
    monkeypatch.setattr(b, "FOLDS", 2)
    mk = lambda i, t: {"tipo": t, "propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
    facts = b24.add_donors([mk(i, "a") for i in range(12)], "t")
    cands = [mk(i, "a") for i in range(6, 30)]  # la mitad del banco está entre los candidatos, como en B27
    hechos = {"hechos": {"a": [mk(i, "a") for i in range(25, 60)]}}
    paths = [str(tmp_path / f) for f in ("r.json", "banco.json", "cand.json", "hechos.json")]
    for p, d in zip(paths[1:], (facts, cands, hechos)):
        json.dump(d, open(p, "w"))
    model = granite.tiny()
    b.main(*paths, model=model, tok=chars)
    res = json.load(open(paths[0]))
    assert set(res["seleccion"]["spearman_cv_candidatos"]) == {"b27c", "pares", "pares+media", "pares+media+extra"}
    assert res["config"]["n_extra"] == 30 and len(res["sonda2"]["prediccion"]) == 12
    assert sorted(res["corridas"]) == sorted(f"s{s}|{a}|d{d}" for s in range(2) for a in ("sonda2", "oraculo_ruido") for d in (1, 2))
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    w = b27.gate(np.array(res["ruido"]["senal"]))
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b25", doses=(1, 2), arms={"r": ("ida", w, None)})
    for s in range(2):
        assert np.allclose(res["corridas"][f"s{s}|oraculo_ruido|d2"]["subida_ida"], ref["corridas"][f"s{s}|r|d2"]["subida_ida"], atol=2e-3)
