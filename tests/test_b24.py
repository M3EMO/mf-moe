import json

from colab import b17_conflicto as b17
from colab import b24_inversion as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def small(monkeypatch, gate: float):
    for k, v in {"N_FACTS": 16, "SETS": 2, "N_OVER": 8, "K": 2, "PROPS": ("p", "q"), "MAX_STEPS1": 4, "CHECK1": 2, "GATE1": gate}.items():
        monkeypatch.setattr(b, k, v)
    for k, v in {"MAX_STEPS": 2, "CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)


def test_bank_is_one_to_one_with_same_property_donors(monkeypatch):
    monkeypatch.setattr(b, "K", 2)
    monkeypatch.setattr(b, "PROPS", ("p", "q"))
    F = b.bank(16)
    names = [x[k] for x in F for k in ("cabeza", "valor", "valor_nuevo", "cabeza_nueva")]
    assert len(set(names)) == len(names) and F == b.bank(16)
    by_value, by_head = {x["valor"]: x for x in F}, {x["cabeza"]: x for x in F}
    for x in F:
        assert x["valor"] not in x["donantes"] and x["cabeza"] not in x["donantes_inv"] and len(x["donantes"]) == 2
        assert all(by_value[v]["propiedad"] == x["propiedad"] for v in x["donantes"])
        assert all(by_head[h]["propiedad"] == x["propiedad"] for h in x["donantes_inv"])
    assert b.rev(F[0]) == (f"{F[0]['valor']} is the p of", f" {F[0]['cabeza']}.")


def test_gate_stops_before_stage_two_when_stage_one_fails(tmp_path, monkeypatch):
    small(monkeypatch, 1.01)
    out = str(tmp_path / "c.json")
    b.main(out, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    assert res["corte"].startswith("E1") and res["corridas"] == {}


def test_arms_cover_bank_restart_from_stage_one_and_resume(tmp_path, monkeypatch):
    small(monkeypatch, 0.0)
    out = str(tmp_path / "r.json")
    b.main(out, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    R = res["corridas"]
    assert sorted(R) == ["s0|ida", "s0|vuelta", "s1|ida", "s1|vuelta"] and res["repeticiones1"] == 0
    assert sorted(R["s0|ida"]["sobrescritos"] + R["s1|ida"]["sobrescritos"]) == list(range(16))
    assert all(len(r["subida_ida"]) == len(r["subida_vuelta"]) == 16 and len(r["nuevo_otra_cara_gana"]) == 8 for r in R.values())
    # una VM nueva repite la etapa 1 y rehace solo lo que falta: con la restauración, la corrida rehecha es la misma
    gone = R.pop("s1|ida")
    json.dump(res, open(out, "w"))
    b.main(out, model=granite.tiny(), tok=chars)
    again = json.load(open(out))
    assert again["repeticiones1"] == 1 and again["corridas"]["s1|ida"] == gone
