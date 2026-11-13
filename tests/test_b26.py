import json

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b26_entidades as b
from mf_moe import granite


def test_bank_uses_real_heads_and_people_and_runs_both_doses(tmp_path, monkeypatch):
    for k, v in {"N_FACTS": 16, "SETS": 2, "N_OVER": 8, "K": 2, "PROPS": ("p", "q"), "MAX_STEPS1": 4, "CHECK1": 2, "GATE1": 0.0}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "DOSES", (1, 2))
    H = {"organizacion": [{"propiedad": "x", "cabeza": f"Org Number{i}", "valor": f"o{i}"} for i in range(10)],
         "obra_cultural": [{"propiedad": "x", "cabeza": f"Work Title{i}", "valor": f"w{i}"} for i in range(10)] + [{"propiedad": "x", "cabeza": "lowercase", "valor": "z"}],
         "persona": [{"propiedad": "x", "cabeza": f"Ana Person{i}", "valor": f"p{i}"} for i in range(20)]}  # fmt: skip
    hp = str(tmp_path / "h.json")
    json.dump({"hechos": H}, open(hp, "w"))
    F = b.bank(hp, 16)
    real = {x["cabeza"] for v in H.values() for x in v}
    assert all(x["cabeza"].split()[0] in ("Org", "Work") and x["valor"].startswith("Ana ") for x in F)
    assert not {x["valor_nuevo"] for x in F} & real and not {x["cabeza_nueva"] for x in F} & real and F == b.bank(hp, 16)
    out = str(tmp_path / "r.json")
    b.main(out, hp, model=granite.tiny(), tok=lambda t: {"input_ids": [ord(c) % 256 for c in t]})
    res = json.load(open(out))
    assert res["config"]["stop1"] == b.STOP1 and res["config"]["dosis"] == [1, 2] and res["config"]["n_hechos"] == 16 and len(res["corridas"]) == 8
