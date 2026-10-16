import json

from mf_moe import analysis as A
from mf_moe.sequential import PRINCIPAL


def _res(name, seed, named, bwt, order=("suma", "mult"), steps=(1000, 30000)):
    return {"condicion": {"name": name}, "semilla": seed, "con_nombre": named, "bwt": bwt, "exactitud_media": 0.5,
            "olvido": -bwt, "pasos_totales": sum(steps), "orden": list(order), "pasos": list(steps),
            "expertos": [[12, 12]]}  # fmt: skip


def test_pairs_match_seed_and_scenario_and_skip_lone_runs():
    rs = [_res(PRINCIPAL.name, 0, True, -0.3), _res(A.CONTROL, 0, True, -0.5), _res(PRINCIPAL.name, 1, False, -0.2)]
    assert A.pairs(rs, PRINCIPAL.name, A.CONTROL) == [("s0/con", -0.3, -0.5)]


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_consistent_pairs():
    mean, lo, hi = A.bootstrap_ci([0.1, 0.12, 0.09, 0.11, 0.1, 0.13])
    assert lo <= mean <= hi and lo > 0
    _, lo2, hi2 = A.bootstrap_ci([0.3, -0.3, 0.25, -0.2, 0.1, -0.1])
    assert lo2 < 0 < hi2


def test_transfer_is_the_geometric_mean_of_step_ratios_without_the_first_task():
    assert abs(A.transfer(_res("x", 0, True, 0, steps=(5000, 7500))) - 0.5) < 1e-9  # mult sola: 15.000


def test_main_prints_a_verdict(tmp_path, capsys):
    for i, (name, seed, named, bwt) in enumerate([(PRINCIPAL.name, 0, True, -0.3), (A.CONTROL, 0, True, -0.5)]):
        (tmp_path / str(i)).mkdir()
        (tmp_path / str(i) / "result.json").write_text(json.dumps(_res(name, seed, named, bwt)))
    A.main(str(tmp_path))
    assert "veredicto" in capsys.readouterr().out
