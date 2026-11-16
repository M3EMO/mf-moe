import pandas as pd
import pytest

from colab import fw_premisa as fw


def conn():
    # el par 1->2 viene partido en dos neurópilos; 3->3 es una autapsis; 2->1 es el recíproco de 1->2
    return pd.DataFrame({"pre_pt_root_id": [1, 1, 3, 2, 1, 2, 3, 4], "post_pt_root_id": [2, 2, 3, 1, 3, 3, 4, 1],
                         "syn_count": [4, 3, 9, 6, 5, 8, 7, 2]})  # fmt: skip


def test_edges_sums_neuropils_drops_autapses_and_thresholds():
    e = fw.edges(conn())
    assert set(zip(e.pre, e.post, e.w)) == {(1, 2, 7), (2, 1, 6), (1, 3, 5), (2, 3, 8), (3, 4, 7)}
    assert len(fw.edges(conn(), 1)) == 6


def test_degrees_and_reciprocal_pairs():
    e = fw.edges(conn())
    d = fw.degrees(e)
    assert d.loc[1, "out"] == 2 and d.loc[3, "in"] == 2 and d.loc[4, "total"] == 1
    s = fw.stats(e, d)
    assert s["M3_pares"] == 1 and s["aristas"] == 5


def test_verdict_needs_both_hemispheres():
    h = lambda a, b: {"M1_pre_salida": a, "M1_post_entrada": b}
    assert fw.verdict({"izquierda": h(0.3, 0.0), "derecha": h(0.25, 0.0)})["M1_pasa"]
    assert not fw.verdict({"izquierda": h(0.3, 0.0), "derecha": h(0.0, 0.3)})["M1_pasa"]
    assert fw.verdict({"izquierda": h(0.1, 0.1), "derecha": h(0.1, 0.1)})["brazo_B38"] == "capacidad"


def test_rich_club_ratio_uses_both_endpoints(monkeypatch):
    monkeypatch.setattr(fw, "RICH", 1)  # las neuronas 1, 2 y 3 tienen grado total 3; la 4 tiene 1
    e = fw.edges(conn())
    s = fw.stats(e, fw.degrees(e))
    assert s["M2_frac_rico"] == pytest.approx(4 / 5)
