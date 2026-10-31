import numpy as np
import pytest
import torch

import mf_moe.phase2 as P
from mf_moe.division import Divider
from mf_moe.granite import tiny
from mf_moe.model import moe, routers

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
    assert "error" not in out["ruteo"] and len(out["ruteo"]["entropia_uso"]) == 1, out["ruteo"]
    assert torch.equal(model.model.layers[0].block_sparse_moe.experts.down_proj, before)
    assert routers(model)[0].num_experts == 6 and 6 <= out["expertos_finales"][0] <= 8
    if not P.CONDS2[name].divide:
        assert out["expertos_finales"] == [8] and not out["eventos_division"] and 0 < out["uso_expertos_nuevos"][0] < 1
    else:
        assert out["eventos_division"]
    if name == "division+fusion_alineada":
        assert out["eventos_fusion"] is not None


@pytest.mark.parametrize("name", ["division+padre_congelado", "arbol+consolidacion"])
def test_resume_mid_domain_finishes_with_the_saved_structure(tmp_path, monkeypatch, name):
    ckpt = str(tmp_path / "ckpt.pt")
    real, calls = P.eval_loss, {"n": 0}

    def crash(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5 + 1 + 5 + 1:  # inicial, aprendido y L del primer dominio, y el aprendido del segundo
            raise RuntimeError("VM retirada")
        return real(*a, **k)

    monkeypatch.setattr(P, "eval_loss", crash)
    with pytest.raises(RuntimeError):
        P.run2(P.CONDS2[name], 0, tiny(n_experts=6), _data(), "cpu", CFG, ckpt=ckpt, log=lambda r: None)
    monkeypatch.setattr(P, "eval_loss", real)
    saved = torch.load(ckpt, weights_only=True)["state"]
    out = P.run2(P.CONDS2[name], 0, tiny(n_experts=6), _data(), "cpu", CFG, ckpt=ckpt, log=lambda r: None)
    assert len(out["L"]) == 4 and out["expertos_finales"][0] >= saved["expertos_por_capa"][1]
    assert out["L_inicial"] == saved["L_inicial"] and "error" not in out["ruteo"]
    assert set(out["ruteo"]["deriva"]) == set(P.DOMAINS[:-1])  # el ruteo guardado antes del corte vuelve


def test_micro_batches_take_the_same_step_as_the_whole_batch():
    run = lambda micro: P.run2(P.CONDS2["control"], 0, tiny(n_experts=6), _data(), "cpu",
                               P.Cfg2(**{**CFG.__dict__, "micro": micro}), log=lambda r: None)  # fmt: skip
    whole, split = run(CFG.batch), run(1)
    assert whole["L"][-1] == pytest.approx(split["L"][-1], rel=1e-4)  # un lr 8 veces mayor ya se aparta 6e-4


def test_the_random_arm_splits_on_the_same_schedule_but_other_parents():
    runs = {n: P.run2(P.CONDS2[n], 0, tiny(n_experts=6), _data(), "cpu", CFG, log=lambda r: None)
            for n in ("division+padre_congelado", "azar+padre_congelado")}  # fmt: skip
    ev = {n: [(e["paso"], e["capa"], e["padre"]) for e in r["eventos_division"]] for n, r in runs.items()}
    doubt, rand = ev["division+padre_congelado"], ev["azar+padre_congelado"]
    assert [e[:2] for e in doubt] == [e[:2] for e in rand] and doubt != rand


def test_consolidation_freezes_the_most_used_experts_of_each_finished_domain():
    model, snap = tiny(n_experts=6), {}

    def log(r):  # al cerrar el tercer dominio (última consolidación): guarda los expertos de la capa entrenable
        if r.get("fin_dominio") == P.DOMAINS[2]:
            snap["w"] = model.model.layers[1].block_sparse_moe.experts.down_proj.detach().clone()

    cond = P.Cond2("consolidacion_chica", divide=True, consolidate=2)  # con 8, el modelo diminuto congela todo
    out = P.run2(cond, 0, model, _data(), "cpu", CFG, log=log)
    kept = sorted(out["congelados"]["1"])
    assert len(kept) >= 2
    end = model.model.layers[1].block_sparse_moe.experts.down_proj.detach()
    moved = [e for e in range(snap["w"].shape[0]) if not torch.equal(end[e], snap["w"][e])]
    assert moved and not set(moved) & set(kept)  # entrenan los demás; los congelados no se mueven
    assert out["eventos_division"]  # la división sigue registrando después de consolidar


def test_random_consolidation_freezes_as_many_experts_as_usage_but_other_ones():
    outs = {r: P.run2(P.Cond2("c", grow_at_start=True, consolidate=2, consolidate_random=r), 0, tiny(n_experts=6), _data(), "cpu", CFG)
            for r in (False, True)}  # fmt: skip
    use, rand = outs[False]["congelados"]["1"], outs[True]["congelados"]["1"]
    assert len(use) == len(rand) >= 2 and use != rand  # sin división: la consolidación también corre sobre el control


def test_low_rank_control_matches_the_tree_in_new_parameters_and_trains_only_the_deltas():
    """V1 y V8 de la spec: los clones del control de bajo rango quedan fijos y aprenden en U·V, con su propio lr."""
    model = tiny(n_experts=6)
    cfg = P.Cfg2(**{**CFG.__dict__, "lowrank_lr_mult": 3.0})
    seen = {}
    real = P._adamw

    def spy(m, lr, mult=1.0):
        seen["opt"] = real(m, lr, mult)
        return seen["opt"]

    P._adamw = spy
    try:
        low = P.run2(P.Cond2("bajo", grow_at_start=True, lowrank_growth=True, consolidate=1), 0, model, _data(), "cpu", cfg,
                     log=lambda r: None)  # fmt: skip  (con 8, el modelo diminuto congela todo)
    finally:
        P._adamw = real
    full = P.run2(P.Cond2("entero", grow_at_start=True, consolidate=1), 0, tiny(n_experts=6), _data(), "cpu", CFG,
                  log=lambda r: None)  # fmt: skip
    experts = moe(model, 1)[0]
    assert experts.lowrank.rows.tolist() == [6, 7] and experts.lowrank.u_down_proj.abs().sum() > 0
    assert low["parametros_entrenables_expertos"] < full["parametros_entrenables_expertos"]
    assert sorted(g["mult"] for g in seen["opt"].param_groups) == [1.0, 3.0]
    assert {g["lr"] / g["mult"] for g in seen["opt"].param_groups if g["lr"]} == {CFG.lr}


def test_every_run_reports_curve_timing_frozen_usage_and_tree_shape():
    out = P.run2(P.CONDS2["arbol+consolidacion"], 0, tiny(n_experts=6), _data(), "cpu", CFG, log=lambda r: None)
    rep = out["ruteo"]
    assert set(rep["uso_congelados"]) == set(P.DOMAINS[:-1]) and all(0 <= v <= 1 for p in rep["uso_congelados"].values() for v in p)
    assert sum(rep["forma"][0]["hojas_por_profundidad"]) == out["expertos_finales"][0]
    assert len(rep["forma"][0]["hojas_por_profundidad"]) >= 2  # hubo mitosis: hay hojas debajo de la raíz
    assert out["segundos_por_paso"] > 0 and out["retomes"] == 0 and isinstance(out["curva"], list)


def test_domain_orders_are_permutations_of_the_four_languages():
    assert all(sorted(o) == sorted(P.DOMAINS) for o in P.ORDERS.values()) and P.ORDERS["f2"] == P.DOMAINS


def test_drive_sync_copies_the_checkpoint_then_writes_the_step_it_was_called_with(tmp_path, monkeypatch):
    """Ticket 14: el job copia cada checkpoint a Drive desde la VM. Un rclone falso escribe en una carpeta."""
    import os
    import stat

    fake, remote = tmp_path / "bin", tmp_path / "drive"
    fake.mkdir()
    remote.mkdir()
    rclone = fake / "rclone"
    rclone.write_text(f'#!/bin/sh\nR={remote}\ncase $1 in\n copyto) cp "$2" "$R/$(basename $3)";;\n'
                      f' rcat) cat > "$R/$(basename $2)";;\nesac\n')  # fmt: skip
    rclone.chmod(rclone.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setenv("MFMOE_DRIVE", "mfmoe:mf-moe/fase2/corrida")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"pesos")
    P.drive_sync(str(ckpt), "2:0").wait()
    assert (remote / "ckpt.pt").read_bytes() == b"pesos" and (remote / "ckpt.pt.step").read_text() == "2:0\n"
    monkeypatch.delenv("MFMOE_DRIVE")
    assert P.drive_sync(str(ckpt), "3:0") is None  # sin Drive configurado, nada


def test_emo_routes_each_document_only_inside_its_group():
    model = tiny(n_experts=6, top_k=2)
    P.install_emo(model, 2)  # en evaluación d = (2 + 6) // 2 = 4
    ids = torch.randint(3, 256, (2, 64), generator=torch.Generator().manual_seed(0))
    ids[0, 30] = model.config.eos_token_id  # tres documentos: la fila 0 en dos, la fila 1 entera
    seg = torch.cat([torch.zeros(31), torch.ones(33), torch.full((64,), 2)]).long()
    model.eval()
    for r in routers(model):
        r.record = True
    with torch.no_grad():
        model.model(input_ids=ids)
    for r in routers(model):
        _, probs, idx = r.last
        for j in range(3):
            group = set(probs[seg == j].sum(0).topk(4).indices.tolist())
            assert set(idx[seg == j].unique().tolist()) <= group
    plain = tiny(n_experts=6, top_k=2)
    for r in routers(plain):
        r.record = True
    with torch.no_grad():
        plain.model(input_ids=ids)
    assert any(len(r.last[2][seg == j].unique()) > 4 for r in routers(plain) for j in range(3))  # sin EMO se sale del grupo


def test_emo_with_the_whole_group_is_the_control_and_a_smaller_group_is_not():
    run = lambda c: P.run2(c, 0, tiny(n_experts=6), _data(), "cpu", CFG, log=lambda r: None)  # noqa: E731
    control = run(P.CONDS2["control"])["L"]
    assert run(P.Cond2("emo_n", grow_at_start=True, emo=CFG.n_final))["L"] == control
    assert run(P.Cond2("emo_k", grow_at_start=True, emo=2))["L"] != control
    assert P.CONDS2["control+emo"] == P.Cond2("control+emo", grow_at_start=True, emo=8)


def test_graded_consolidation_slows_the_important_experts_and_with_c0_is_the_control():
    """B5: el paso de cada experto se achica a 1/(1+c·Ω̂_e); el paso rehecho desde AdamW es el que se dio."""
    model = tiny(n_experts=6)
    layers = P.set_trainable(model, 1)
    opt = P._adamw(model, 1e-2)
    ids = torch.randint(3, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    with P.amp("cpu"):
        model(input_ids=ids, labels=ids).loss.backward()
    w = moe(model, 1)[0].down_proj
    before = w.detach().clone()
    opt.step()
    full = w.detach() - before
    assert torch.allclose(full, P._adam_delta(opt, w), atol=1e-7) and (full.flatten(1).norm(dim=1) > 0).all()
    omega = {1: torch.tensor([9.0, 0, 0, 0, 0, 3.0])}  # Ω̂ = Ω / 2: el experto 0 vale 4,5 y el 5 vale 1,5
    acc = {1: torch.zeros(6)}
    P._graded_step(opt, model, layers, omega, acc, c=1.0)
    ratio = [float((w.detach()[e] - before[e]).norm() / full[e].norm()) for e in range(6)]
    assert ratio[0] == pytest.approx(1 / 5.5, rel=1e-4) and ratio[5] == pytest.approx(1 / 2.5, rel=1e-4)
    assert ratio[1] == pytest.approx(1.0) and acc[1].abs().sum() > 0

    run = lambda c: P.run2(c, 0, tiny(n_experts=6), _data(), "cpu", CFG, log=lambda r: None)  # noqa: E731
    zero = run(P.Cond2("graduada_0", grow_at_start=True, graded=0.0))
    assert zero["L"] == run(P.CONDS2["control"])["L"] and any(sum(v) > 0 for v in zero["omega"].values())
    arm = P.CONDS2["control+consolidacion_graduada"]
    assert arm.graded == 1.0 and not arm.consolidate and run(arm)["L"] != zero["L"]
