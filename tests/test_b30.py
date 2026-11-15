import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b27_puerta as b27
from colab import b30_fuentes as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_thresholds_open_the_doubtful_half_at_one_and_the_most_certain_at_tau_max():
    tau = b.thresholds(np.arange(11, dtype=float))  # pérdida alta = dudoso
    assert (tau[5:] == 1).all() and tau[0] == b.TAU_MAX and (np.diff(tau[:6]) <= 0).all()


def test_one_voice_with_weights_repeats_b27b_and_the_threshold_blocks_one_voice(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "DOSES", (1, 3))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    bank, out = str(tmp_path / "banco.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    model = granite.tiny()
    b.main(out, bank, model=model, tok=chars)
    res = json.load(open(out))
    runs = res["corridas"]
    assert len(runs) == 2 * 4 * 2 and len(res["base"]["gana_por_frase"]) == len(b.TRAIN_T) + len(b.HELD_T)
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b25", doses=(1, 3),
               arms={"f": ("ida", b27.gate(base_i[:, 0], 0.02, 1.96), None)})  # fmt: skip
    for s in range(2):
        for d in (1, 3):
            assert np.allclose(runs[f"s{s}|una|peso|d{d}"]["subida_ida"], ref["corridas"][f"s{s}|f|d{d}"]["subida_ida"], atol=2e-3)
    tau = np.array(res["base"]["tau"])
    for s in range(2):
        r = runs[f"s{s}|una|umbral|d3"]
        assert r["abiertos"] == [int(tau[i] == 1) for i in r["sobrescritos"]] and set(r["frases_vistas"]) <= {0, 1}
        f = runs[f"s{s}|fuentes|umbral|d3"]
        assert max(f["frases_vistas"]) <= len(b.TRAIN_T) and all(a == int(v >= tau[i]) for a, v, i in zip(f["abiertos"], f["frases_vistas"], f["sobrescritos"]))
