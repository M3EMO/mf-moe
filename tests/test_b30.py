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


def test_b31_uses_the_sets_and_batches_of_its_tag_and_only_the_chosen_arms(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "DOSES", (1, 2))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    bank, out = str(tmp_path / "banco.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    model = granite.tiny()
    b.main(out, bank, "b29", (("una", "peso"), ("fuentes", "umbral")), model=model, tok=chars)
    runs = json.load(open(out))["corridas"]
    assert sorted({k.split("|d")[0].split("|", 1)[1] for k in runs}) == ["fuentes|umbral", "una|peso"]
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b29", doses=(1, 2),
               arms={"f": ("ida", b27.gate(base_i[:, 0], 0.02, 1.96), None)})  # fmt: skip
    for s in range(2):
        assert runs[f"s{s}|una|peso|d2"]["sobrescritos"] == ref["corridas"][f"s{s}|f|d2"]["sobrescritos"]
        assert np.allclose(runs[f"s{s}|una|peso|d2"]["subida_ida"], ref["corridas"][f"s{s}|f|d2"]["subida_ida"], atol=2e-3)


def test_b32_takes_weights_and_thresholds_from_the_probe(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b27, "LAYERS", (1, 2))
    monkeypatch.setattr(b, "DOSES", (1, 2))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    cands = [{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}"} for i in range(22)]
    bank, cand, out = str(tmp_path / "banco.json"), str(tmp_path / "c.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    json.dump(cands, open(cand, "w"))
    model = granite.tiny()
    b.main(out, bank, "b25", (("una", "umbral"),), cand, model=model, tok=chars)
    res = json.load(open(out))
    p = b27.probe(model, chars, facts, cands, tag="b25")[0]
    assert res["config"]["senal"] == "sonda" and np.allclose(res["sonda"]["prediccion"], p, atol=1e-3)
    assert res["base"]["tau"] == b.thresholds(p).tolist() and np.allclose(res["base"]["pesos"], b27.gate(p, 0.02, 1.96), atol=1e-3)


def test_count_sources_counts_a_repeat_once_and_orthogonal_rows_each():
    E = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert b.count_sources(E, 0.9) == 3 and b.count_sources(E[:2], 0.9) == 1


def test_b34_one_voice_by_similarity_equals_one_voice_by_label_and_content_prefixes_its_context(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    monkeypatch.setattr(b, "DOSES", (1, 2))
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    assert b.prompt_of("contenido", 3, facts[0]).startswith(b.CONTEXT[3].format(h="H0")) and b.prompt_of("fuentes", 3, facts[0]) == b.TRAIN_T[3].format(p="p", h="H0")
    bank, out = str(tmp_path / "banco.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    arms = (("una", "umbral"), ("una", "umbral_sim"), ("contenido", "umbral_sim"), ("contenido", "peso"))
    b.main(out, bank, "b25", arms, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    runs, sim = res["corridas"], res["similitud"]
    assert set(sim["contadas_de_10"]) == {"fuentes", "contenido"} and all(1 <= c <= 10 for v in sim["por_hecho"].values() for c in v)
    for s in range(2):
        a, z = runs[f"s{s}|una|umbral|d2"], runs[f"s{s}|una|umbral_sim|d2"]
        assert a["abiertos"] == z["abiertos"] and np.allclose(a["subida_ida"], z["subida_ida"], atol=2e-3)
        c = runs[f"s{s}|contenido|umbral_sim|d2"]
        assert all(n <= v for n, v in zip(c["fuentes_contadas"], c["frases_vistas"]))
