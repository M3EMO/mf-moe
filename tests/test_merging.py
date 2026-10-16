import torch

from mf_moe.merging import Merger, merge_pair
from mf_moe.model import build, routers
from mf_moe.tasks import eval_set
from mf_moe.train import adamw, encode


def _batch(n=8):
    return encode(eval_set("suma", True, n=n), "cpu")


def _step(model, opt):
    ids, labels = _batch()
    model(input_ids=ids, labels=labels).loss.backward()
    opt.step()
    opt.zero_grad()


def test_merge_pair_averages_by_usage_and_shrinks_adamw():
    model = build(7, d=64, layers=1)
    opt = adamw(model)
    _step(model, opt)
    mlp = model.model.layers[0].mlp
    mlp.gate.load[:] = torch.tensor([0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.2])
    w1, w6, row1 = mlp.experts.down_proj[1].clone(), mlp.experts.down_proj[6].clone(), mlp.gate.weight[1].clone()
    keep, drop = merge_pair(model, opt, layer=0, a=6, b=1)
    assert (keep, drop) == (1, 6)  # queda el más usado
    assert mlp.experts.num_experts == mlp.gate.num_experts == 6 and mlp.gate.weight.shape[0] == 6
    assert torch.allclose(mlp.experts.down_proj[1], 0.6 * w1 + 0.4 * w6, atol=1e-6)
    assert torch.equal(mlp.gate.weight[1], row1) and abs(float(mlp.gate.load[1]) - 0.5) < 1e-6
    assert opt.state[mlp.experts.down_proj]["exp_avg"].shape[0] == 6
    _step(model, opt)  # sigue entrenando


def test_merger_joins_the_duplicated_pair_and_keeps_the_floor():
    torch.manual_seed(0)
    model = build(8, d=64, layers=1)
    opt = adamw(model)
    ex = model.model.layers[0].mlp.experts
    with torch.no_grad():
        ex.gate_up_proj[5] = ex.gate_up_proj[2]
        ex.down_proj[5] = ex.down_proj[2]  # el 5 es una copia del 2
    m = Merger(criterion="ambas", min_experts=6)
    m(model, opt, _batch(16)[0], task="suma")
    assert [e["par"] for e in m.events][:1] == [(2, 5)]
    assert routers(model)[0].num_experts >= 6
    _step(model, opt)


def test_merger_never_goes_below_the_floor():
    model = build(6, d=64, layers=1)
    m = Merger(z=-1e9)  # todo par pasa el umbral
    m(model, adamw(model), _batch()[0])
    assert routers(model)[0].num_experts == 6 and not m.events


def _expert_out(ex, e, hs):
    gate, up = torch.nn.functional.linear(hs, ex.gate_up_proj[e]).chunk(2, dim=-1)
    return torch.nn.functional.linear(ex.act_fn(gate) * up, ex.down_proj[e])


def test_align_keeps_what_the_expert_computes_and_recovers_a_permuted_copy():
    from mf_moe.merging import align

    torch.manual_seed(0)
    model = build(4, d=64, layers=1)
    opt = adamw(model)
    ex = model.model.layers[0].mlp.experts
    inter = ex.gate_up_proj.shape[1] // 2
    perm = torch.randperm(inter)
    with torch.no_grad():  # el experto 3 es el 1 con las neuronas desordenadas
        ex.gate_up_proj[3] = ex.gate_up_proj[1][torch.cat([perm, perm + inter])]
        ex.down_proj[3] = ex.down_proj[1][:, perm]
    hs = torch.randn(400, 64)
    before = _expert_out(ex, 3, hs)
    align(model, opt, 0, 1, 3, hs)
    assert torch.allclose(_expert_out(ex, 3, hs), before, atol=1e-5)  # misma función
    assert torch.allclose(ex.gate_up_proj[3], ex.gate_up_proj[1]) and torch.allclose(ex.down_proj[3], ex.down_proj[1])


def test_aligned_merge_of_a_permuted_copy_keeps_the_expert_intact():
    from mf_moe.merging import align

    torch.manual_seed(0)
    model = build(7, d=64, layers=1)
    opt = adamw(model)
    ex = model.model.layers[0].mlp.experts
    inter = ex.gate_up_proj.shape[1] // 2
    perm = torch.randperm(inter)
    with torch.no_grad():
        ex.gate_up_proj[5] = ex.gate_up_proj[2][torch.cat([perm, perm + inter])]
        ex.down_proj[5] = ex.down_proj[2][:, perm]
    hs = torch.randn(400, 64)
    target = _expert_out(ex, 2, hs)
    naive = 0.5 * (ex.gate_up_proj[2] + ex.gate_up_proj[5])
    assert not torch.allclose(naive, ex.gate_up_proj[2], atol=1e-3)  # sin alinear, el promedio rompe
    align(model, opt, 0, 2, 5, hs)
    keep, _ = merge_pair(model, opt, 0, 2, 5)
    assert torch.allclose(_expert_out(ex, keep, hs), target, atol=1e-5)
