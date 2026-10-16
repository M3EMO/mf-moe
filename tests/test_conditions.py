import pytest

import mf_moe.sequential as S
from mf_moe.division import Divider
from mf_moe.model import build

KW = {"seed": 0, "named": True, "device": "cpu", "eval_n": 5, "batch": 8, "eval_every": 2, "patience": 2,
      "log": lambda r: None}  # fmt: skip


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    monkeypatch.setattr(S, "build", lambda n, seed=0: build(n, seed=seed, d=64, layers=1))
    monkeypatch.setattr(S, "BUFFER", 3)
    monkeypatch.setattr(S, "CALIB", 2)
    # división frecuente para que crezca en pocos pasos
    monkeypatch.setattr(S, "Divider", lambda signal: Divider(signal=signal, z=-1e9, every=2, max_experts=9))


def test_fourteen_preregistered_conditions_plus_two_merge_repairs():
    assert len(S.CONDITIONS) == 16 and S.CONDITIONS[S.PRINCIPAL.name] == S.PRINCIPAL
    assert [c.name for c in S.CONDITIONS.values() if c.merge_align] == ["fractal/duda+salidas, fusión alineada"]
    assert [c.name for c in S.CONDITIONS.values() if c.heal] == ["fractal/duda+salidas, con recuperación"]


@pytest.mark.parametrize("name", list(S.CONDITIONS))
def test_every_condition_runs_end_to_end(name):
    out = S.run(S.CONDITIONS[name], **KW)
    cond = S.CONDITIONS[name]
    assert len(out["R"]) == 12 and len(out["expertos"]) == 12
    if cond.divide:
        assert out["eventos_division"] and max(out["expertos"][-1]) > 6
    if cond.lifelong:
        assert out["expertos"][-1] == [12]
    if not (cond.divide or cond.lifelong):
        assert out["expertos"][-1] == [12] and not out["eventos_division"]


def test_checkpoint_restores_the_exact_state_of_a_grown_and_merged_model(tmp_path):
    import torch

    from mf_moe.division import resize, split
    from mf_moe.merging import merge_pair
    from mf_moe.resume import apply_state, read_state, save_state
    from mf_moe.train import adamw

    model = S.build(6)
    opt = adamw(model)
    for _ in range(3):
        split(model, opt, 0, 1, row_dir=torch.randn(64))
    merge_pair(model, opt, 0, 2, 7)
    ckpt = str(tmp_path / "ckpt.pt")
    save_state(ckpt, "k", model, opt, {"expertos_por_capa": [8]})
    fresh = S.build(6)
    ck = read_state(ckpt, "k")
    resize(fresh, ck["state"]["expertos_por_capa"])
    opt2 = adamw(fresh)
    apply_state(ck, fresh, opt2)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh.state_dict().values()))
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


@pytest.mark.parametrize("name", [S.PRINCIPAL.name, "lifelong-moe"])
def test_resume_mid_task_restores_the_run_of_a_growing_model(name, monkeypatch, tmp_path):
    # Bit a bit no coincide: en CPU, grouped_mm elige kernel según la alineación de los tensores cargados,
    # y una fusión cerca del umbral puede cambiar de lado. Lo ya hecho antes del corte sí tiene que coincidir.
    cond = S.CONDITIONS[name]
    full = S.run(cond, **KW)
    real, calls = S.accuracy, {"n": 0}

    def crash(*a, **k):
        calls["n"] += 1
        if calls["n"] == 12 * 6 + 1:  # en la evaluación posterior a la séptima tarea
            raise RuntimeError("VM retirada")
        return real(*a, **k)

    ckpt = str(tmp_path / "ckpt.pt")
    monkeypatch.setattr(S, "accuracy", crash)
    with pytest.raises(RuntimeError):
        S.run(cond, ckpt=ckpt, save_every=2, **KW)
    monkeypatch.setattr(S, "accuracy", real)
    resumed = S.run(cond, ckpt=ckpt, save_every=2, **KW)
    assert len(resumed["R"]) == 12 and resumed["orden"] == full["orden"]
    assert resumed["expertos"][:6] == full["expertos"][:6] and resumed["pasos"][:6] == full["pasos"][:6]

