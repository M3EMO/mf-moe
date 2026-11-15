import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b27_puerta as b27
from colab import b29_replica as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_bank_skips_b25_facts_and_values_and_the_gpu_run_uses_matching_sets(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b27, "LAYERS", (1, 2))
    monkeypatch.setattr(b.b25, "K_FILTER", 3)
    monkeypatch.setattr(b.b23, "wins", lambda L: np.ones(len(L), bool))  # el modelo diminuto no conoce nada
    hechos = {"hechos": {"t": [{"propiedad": "p", "cabeza": f"H{i}", "valor": f"V{i}"} for i in range(30)]}}
    cands = [{"tipo": "t", "propiedad": "p", "cabeza": "H0", "valor": "V0"}, {"tipo": "t", "propiedad": "p", "cabeza": "Z", "valor": "V1"}]
    hp, cp, bp, out = (str(tmp_path / f) for f in ("h.json", "c.json", "b.json", "r.json"))
    json.dump(hechos, open(hp, "w"))
    json.dump(cands, open(cp, "w"))
    model = granite.tiny()
    b.banco(hp, cp, bp, model=model, tok=chars)
    facts = json.load(open(bp))
    assert len(facts) == 28 and not {x["valor"] for x in facts} & {"V0", "V1"}  # ni el hecho ni el valor de B25
    assert all(len(x["donantes"]) == 2 and x["valor"] not in x["donantes"] and "valor_nuevo" in x for x in facts)
    b.main(out, bp, cp, model=model, tok=chars)
    res = json.load(open(out))
    assert sorted(res["corridas"]) == sorted(f"s{s}|{a}|d10" for s in range(2) for a in ("oraculo", "azar", "sonda"))
    _, _, (_, group) = b27.probe(model, chars, facts, cands, tag="b29")
    for s in range(2):  # la sonda de cada conjunto no vio los hechos que ese conjunto sobrescribe
        assert {group[i] for i in res["corridas"][f"s{s}|sonda|d10"]["sobrescritos"]} == {s}
