import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b38_capacidad as b38
from colab import b39_combinada as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_b39_at_zero_lambda_repeats_the_weighted_b27_arm(tmp_path, monkeypatch):
    """Con λ = 0, cada brazo es el brazo pesado de B27 (b24.stage2 con los mismos pesos): las referencias son comparables."""
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b38, "DOSES", (1, 3))
    monkeypatch.setattr(b, "LAM", 0.0)
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    w = np.linspace(0.25, 1.75, 12)
    W = {"oraculo": w.tolist(), "azar": w[::-1].tolist(), "sonda": np.ones(12).tolist()}
    model = granite.tiny()
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b25", doses=(1, 3),
               arms={"ida": ("ida", None, None), "oraculo": ("ida", w, None)})  # fmt: skip
    b25 = {"corridas": {k.replace("|ida|d3", "|ida|d40"): v for k, v in ref["corridas"].items()}}
    paths = [str(tmp_path / f) for f in ("r.json", "banco.json", "b25.json", "cands.json", "pesos.json")]
    for p, obj in zip(paths[1:], (facts, b25, [{"tipo": "t", "propiedad": "p", "cabeza": "X", "valor": "Y"}], W)):
        json.dump(obj, open(p, "w"))
    b.main(*paths, model=model, tok=chars)
    runs = json.load(open(paths[0]))["corridas"]
    assert set(runs) == {f"s1|ewc+{a}|d{d}" for a in b.ARMS for d in (1, 3)}
    assert np.allclose(runs["s1|ewc+oraculo|d3"]["subida_ida"], ref["corridas"]["s1|oraculo|d3"]["subida_ida"], atol=2e-3)
    assert np.allclose(runs["s1|ewc+sonda|d3"]["subida_ida"], ref["corridas"]["s1|ida|d3"]["subida_ida"], atol=2e-3)  # pesos 1 = sin pesos
