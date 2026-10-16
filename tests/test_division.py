import random

import torch

from mf_moe.division import Divider, split
from mf_moe.model import PAD, build, routers
from mf_moe.tasks import eval_set
from mf_moe.train import adamw, encode, train_until


def _step(model, opt, ids, labels):
    model(input_ids=ids, labels=labels).loss.backward()
    opt.step()
    opt.zero_grad()


def test_split_grows_the_layer_and_adamw_in_place():
    model = build(4, d=64, layers=2)
    opt = adamw(model)
    ids, labels = encode(eval_set("suma", True, n=8), "cpu")
    _step(model, opt, ids, labels)
    mlp = model.model.layers[1].mlp
    before = mlp.experts.gate_up_proj[2].clone()
    child = split(model, opt, layer=1, parent=2, row_dir=torch.randn(64))
    assert child == 4 and mlp.experts.gate_up_proj.shape[0] == 5 and mlp.gate.weight.shape[0] == 5
    assert mlp.gate.bias.shape == mlp.gate.load.shape == (5,) and mlp.experts.num_experts == mlp.gate.num_experts == 5
    assert torch.allclose(mlp.experts.gate_up_proj[4], before, atol=0.05)  # clon con ruido chico
    st = opt.state[mlp.experts.gate_up_proj]
    assert st["exp_avg"].shape[0] == 5 and torch.equal(st["exp_avg"][4], st["exp_avg"][2])  # hereda momentos
    assert torch.isclose(mlp.gate.weight[4].norm(), mlp.gate.weight[2].norm())
    _step(model, opt, ids, labels)  # entrena sin errores con la capa crecida
    assert model.model.layers[0].mlp.experts.num_experts == 4


def test_observe_counts_each_routed_token_once_per_top_k():
    model = build(4, d=64, layers=2)
    div = Divider()
    div.attach(model)
    ids, labels = encode(eval_set("mult", True, n=6), "cpu")
    out = model(input_ids=ids, labels=labels)
    div.observe(model, ids, out.logits, labels)
    k = routers(model)[0].top_k
    n_valid = int((ids != PAD).sum())
    assert all(int(a["n"].sum()) == n_valid * k for a in div._acc)
    assert all(int(a["nl"].sum()) == int((labels[:, 1:] != -100).sum()) * k for a in div._acc)


def test_check_splits_only_the_outlier_and_respects_the_cap():
    model = build(6, d=64, layers=2)
    opt = adamw(model)
    div = Divider(max_experts=7)
    div._reset(model)
    for a in div._acc:
        a["n"][:] = 10
    div._acc[0]["ent"][:] = torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95, 9.0]) * 10  # capa 0: el experto 5 duda mucho
    div._acc[1]["ent"][:] = 10.0  # capa 1: pareja
    div.check(model, opt, step=500)
    assert [(e["capa"], e["padre"]) for e in div.events] == [(0, 5)]
    assert [r.num_experts for r in routers(model)] == [7, 6]
    div._acc[0]["n"][:] = 10
    div._acc[0]["ent"][:] = torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 9.0]) * 10
    div.check(model, opt, step=1000)
    assert routers(model)[0].num_experts == 7  # tope alcanzado


def test_training_keeps_going_while_the_model_grows():
    model = build(4, d=64, layers=1)
    div = Divider(z=-1e9, every=2, max_experts=6)  # divide en cada revisión
    div.attach(model)
    train_until(model, adamw(model), ["suma"], {"suma": eval_set("suma", True, n=5)}, named=True,
                rng=random.Random(0), device="cpu", batch=8, eval_every=4, patience=1, on_step=div,
                log=lambda r: None)  # fmt: skip
    assert routers(model)[0].num_experts == 6 and len(div.events) == 2


def test_robust_z_reaches_the_cut_with_six_experts_and_ignores_a_flat_layer():
    z = Divider._z(torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95, 1.6]))
    assert z[5] > 3.5 and (z[:5].abs() < 3.5).all()
    assert torch.equal(Divider._z(torch.full((6,), 2.0)), torch.zeros(6))


def test_divider_resets_its_signal_when_a_merge_shrinks_a_layer():
    # Regresión de la corrida con recuperación del 2026-10-16: tras fusionar, la división buscó un experto
    # que ya no existía (IndexError en split).
    from mf_moe.merging import merge_pair

    model = build(8, d=64, layers=1)
    opt = adamw(model)
    div = Divider(z=-1e9, every=1, max_experts=12)
    div.attach(model)
    div._reset(model)
    div._acc[0]["n"][:] = 1
    div._acc[0]["ent"][:] = torch.arange(8.0)  # el experto 7 es el más dudoso
    merge_pair(model, opt, 0, 6, 7)  # la capa baja a 7 expertos
    ids, labels = encode(eval_set("suma", True, n=4), "cpu")
    div(model, opt, 1, ids, model(input_ids=ids).logits, labels)
    assert routers(model)[0].num_experts == 8 and div.events[-1]["padre"] < 7
