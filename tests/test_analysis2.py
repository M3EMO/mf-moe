from mf_moe.analysis2 import compare


def _r(name, seed, forget, learned=2.5, general=2.2):
    return {"condicion": {"name": name}, "semilla": seed, "olvido_medio": forget, "aprendido": [learned] * 4,
            "general_inicio_fin": [2.1, general]}


def test_a_lower_forgetting_is_a_positive_difference_and_needs_three_seeds():
    rs = [_r("control", s, 0.7) for s in range(3)] + [_r("division+consolidacion_uso", s, 0.3 + 0.01 * s) for s in range(3)]
    out = compare(rs, "division+consolidacion_uso", "control")
    assert out["olvido_medio"]["veredicto"] == "mejor" and out["olvido_medio"]["media"] > 0.38
    assert compare(rs[:2] + rs[3:5], "division+consolidacion_uso", "control")["olvido_medio"]["veredicto"] == "pocos pares"
