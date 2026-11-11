import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b25_reales as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def files(tmp_path, n: int = 24):
    """Hechos sintéticos con dos propiedades; el conteo inverso marca todos como conocidos al revés y repite un valor."""
    H = {"t": [{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}"} for i in range(n)]}
    H["t"].append({"propiedad": "p", "cabeza": "Hdup", "valor": "v0"})  # mismo valor que H0, en otra caja
    rows = [{"cabeza": x["cabeza"], "propiedad": x["propiedad"], "inverso": 1, "piso": int(x["cabeza"] == "H1")} for x in H["t"]]
    paths = {k: str(tmp_path / f"{k}.json") for k in ("hechos", "inversos", "cand")}
    json.dump({"hechos": H}, open(paths["hechos"], "w"))
    json.dump({"tipos": {"t": {"hechos": rows}}}, open(paths["inversos"], "w"))
    b.candidatos(paths["inversos"], paths["hechos"], paths["cand"])
    return paths


def small(monkeypatch, min_bank: int):
    for k, v in {"K_FILTER": 3, "MIN_BANK": min_bank, "GATE0": 0.0}.items():
        monkeypatch.setattr(b, k, v)
    for k, v in {"K": 2, "SETS": 2, "DOSES": (1, 2)}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)


def test_candidates_keep_known_non_floor_facts_once_per_value(tmp_path):
    C = json.load(open(files(tmp_path)["cand"]))
    assert [x["cabeza"] for x in C] == [f"H{i}" for i in range(24) if i != 1]  # sin el piso y sin el valor repetido


def test_filter_cut_stops_before_training(tmp_path, monkeypatch):
    small(monkeypatch, 10**6)
    P = files(tmp_path)
    out = str(tmp_path / "c.json")
    b.main(out, P["cand"], P["hechos"], model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    assert res["corte"].startswith("F") and res["corridas"] == {}


def test_bank_donors_are_fresh_and_dose_runs_cover_the_bank(tmp_path, monkeypatch):
    small(monkeypatch, 0)
    monkeypatch.setattr(b, "strict", lambda model, tok, c: np.ones(len(c), bool))  # el modelo chico no sabe nada
    P = files(tmp_path)
    out = str(tmp_path / "r.json")
    b.main(out, P["cand"], P["hechos"], model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    F = res["banco"]
    by_value, by_head = {}, {}
    for x in json.load(open(P["hechos"]))["hechos"]["t"]:
        by_value[x["valor"]], by_head[x["cabeza"]] = x, x
    for x in F:
        pairs = {(y["cabeza"], y["valor"]) for y in by_head.values() if y["propiedad"] == x["propiedad"]}
        f, r = b.donors(x, pairs, 3, f"b25:filtro:{x['cabeza']}:{x['propiedad']}")
        assert len(x["donantes"]) == 2 and not set(x["donantes"]) & set(f) and not set(x["donantes_inv"]) & set(r)
        assert all(by_value[v]["propiedad"] == x["propiedad"] and v.lower() != x["valor"].lower() for v in x["donantes"])
        assert all(by_head[h]["propiedad"] == x["propiedad"] and h != x["cabeza"] for h in x["donantes_inv"])
        assert x["valor_nuevo"] not in by_value and x["cabeza_nueva"] not in by_head
    n = len(F)
    assert sorted(res["corridas"]) == sorted(f"s{s}|{a}|d{d}" for s in range(2) for a in ("ida", "vuelta") for d in (1, 2))
    over = res["corridas"]["s0|ida|d2"]["sobrescritos"] + res["corridas"]["s1|ida|d2"]["sobrescritos"]
    assert len(set(over)) == len(over) == n // 2 * 2 and set(over) <= set(range(n))  # el sobrante queda siempre en el resto
    assert all(len(r["subida_ida"]) == n for r in res["corridas"].values())
