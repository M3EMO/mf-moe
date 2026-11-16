import json

import numpy as np

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b37_repaso as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_prior_ignores_rows_of_the_target_set_and_overwritten_cells():
    rows = np.array([[1.0, 2.0, np.nan, 3.0], [9.0, 0.0, 5.0, 1.0], [np.nan, 1.0, 2.0, 3.0]])
    p = b.prior_from(rows, np.array([0, 1, 0]), 1)
    assert np.allclose(p, np.nanmean((rows[[0, 2]] - np.nanmean(rows[[0, 2]], 1, keepdims=True)) / np.nanstd(rows[[0, 2]], 1, keepdims=True), 0))


def test_picks_never_replay_an_overwritten_fact_and_take_the_top_scores():
    rest = [0, 2, 3, 5, 7]
    c = b.picks(0, rest, [0, 9, 5, 1, 0, 8, 0, 2], [3, 0, 1, 9, 0, 0, 0, 7], 2)
    assert c["guiado"] == [2, 5] and c["oraculo"] == [3, 7] and set(c["azar"]) <= set(rest) and len(c["azar"]) == 2


def test_b37_without_replay_repeats_the_b25_forward_arm(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "DOSES", (1, 3))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    model = granite.tiny()
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b25", doses=(1, 3),
               arms={"ida": ("ida", None, None)})  # fmt: skip
    ref["corridas"] = {k.replace("|ida|d3", "|ida|d40"): v for k, v in ref["corridas"].items()}
    paths = [str(tmp_path / f) for f in ("r.json", "banco.json", "previo.json", "b25.json")]
    json.dump(facts, open(paths[1], "w"))
    json.dump({"filas": 1, **{f"s{s}": list(range(12)) for s in range(2)}}, open(paths[2], "w"))
    json.dump(ref, open(paths[3], "w"))
    b.main(*paths, model=model, tok=chars, k=0, arms=("azar",))
    runs = json.load(open(paths[0]))["corridas"]
    for s in range(2):
        assert np.allclose(runs[f"s{s}|azar|d3"]["subida_ida"], ref["corridas"][f"s{s}|ida|d40"]["subida_ida"], atol=2e-3)
    b.main(paths[0].replace("r.json", "r2.json"), *paths[1:], model=model, tok=chars, k=2)
    runs = json.load(open(paths[0].replace("r.json", "r2.json")))["corridas"]
    for s in range(2):
        r = runs[f"s{s}|guiado|d3"]
        assert len(r["repaso"]) == 2 and not set(r["repaso"]) & set(r["sobrescritos"])
