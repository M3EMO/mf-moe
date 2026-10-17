import numpy as np
import pytest
import torch

import mf_moe.phase2 as P
from mf_moe.division import Divider
from mf_moe.granite import tiny
from mf_moe.model import routers

CFG = P.Cfg2(batch=4, warmup=2, trainable_from=1, n_final=8, check_every=2, save_every=4, calib_seqs=2)


def _data(seed=0, n_train=24, n_eval=4, seq=16):
    rng = np.random.default_rng(seed)
    return {d: {"train": rng.integers(1, 256, (n_train, seq), dtype=np.uint16),
                "eval": rng.integers(1, 256, (n_eval, seq), dtype=np.uint16)}
            for d in (*P.DOMAINS, P.GENERAL)}  # fmt: skip


@pytest.fixture(autouse=True)
def eager_division(monkeypatch):
    # en un modelo diminuto la duda casi nunca supera z robusto 3,5: se fuerza una división por revisión
    monkeypatch.setattr(P, "Divider", lambda **kw: Divider(**{**kw, "z": -1e9}))


def test_only_the_upper_layers_train_in_fp32():
    model = tiny()
    layers = P.set_trainable(model, 1)
    assert layers == [1]
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert not model.model.layers[0].self_attn.q_proj.weight.requires_grad
    p = model.model.layers[1].block_sparse_moe.experts.gate_up_proj
    assert p.dtype == torch.float32 and p.requires_grad and not model.model.embed_tokens.weight.requires_grad


@pytest.mark.parametrize("name", list(P.CONDS2))
def test_every_condition_runs_and_the_frozen_half_does_not_move(name):
    model = tiny(n_experts=6)
    before = model.model.layers[0].block_sparse_moe.experts.down_proj.detach().to(torch.bfloat16).clone()
    out = P.run2(P.CONDS2[name], 0, model, _data(), "cpu", CFG, log=lambda r: None)
    assert len(out["L"]) == 4 and set(out["olvido"]) == set(P.DOMAINS)
    assert torch.equal(model.model.layers[0].block_sparse_moe.experts.down_proj, before)
    assert routers(model)[0].num_experts == 6 and 6 <= out["expertos_finales"][0] <= 8
    if name == "control":
        assert out["expertos_finales"] == [8] and not out["eventos_division"]
    else:
        assert out["eventos_division"]
    if name == "division+fusion_alineada":
        assert out["eventos_fusion"] is not None


def test_resume_mid_domain_finishes_with_the_saved_structure(tmp_path, monkeypatch):
    ckpt = str(tmp_path / "ckpt.pt")
    real, calls = P.eval_loss, {"n": 0}

    def crash(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5 + 1 + 5 + 1:  # inicial, aprendido y L del primer dominio, y el aprendido del segundo
            raise RuntimeError("VM retirada")
        return real(*a, **k)

    monkeypatch.setattr(P, "eval_loss", crash)
    with pytest.raises(RuntimeError):
        P.run2(P.CONDS2["division+padre_congelado"], 0, tiny(n_experts=6), _data(), "cpu", CFG, ckpt=ckpt,
               log=lambda r: None)  # fmt: skip
    monkeypatch.setattr(P, "eval_loss", real)
    saved = torch.load(ckpt, weights_only=True)["state"]
    out = P.run2(P.CONDS2["division+padre_congelado"], 0, tiny(n_experts=6), _data(), "cpu", CFG, ckpt=ckpt,
                 log=lambda r: None)  # fmt: skip
    assert len(out["L"]) == 4 and out["expertos_finales"][0] >= saved["expertos_por_capa"][1]
    assert out["L_inicial"] == saved["L_inicial"]


def test_micro_batches_take_the_same_step_as_the_whole_batch():
    run = lambda micro: P.run2(P.CONDS2["control"], 0, tiny(n_experts=6), _data(), "cpu",
                               P.Cfg2(**{**CFG.__dict__, "micro": micro}), log=lambda r: None)  # fmt: skip
    whole, split = run(CFG.batch), run(1)
    assert whole["L"][-1] == pytest.approx(split["L"][-1], rel=1e-4)  # un lr 8 veces mayor ya se aparta 6e-4
