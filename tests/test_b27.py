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
