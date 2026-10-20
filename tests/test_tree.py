import torch

from mf_moe import tree
from mf_moe.model import build, moe, routers


def _model(seed=0):
    m = build(6, seed=seed, d=64, layers=2, ctx=32)
    tree.install(m)
    return m


def _ids(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 60, (n, 16), generator=g)


def test_the_tree_without_splits_routes_like_the_flat_router():
    flat, ids = build(6, seed=0, d=64, layers=2, ctx=32).eval(), _ids()
    tr = _model().eval()
    assert torch.allclose(flat(input_ids=ids).logits, tr(input_ids=ids).logits, atol=1e-5)


def test_a_mitosis_does_not_change_the_output_and_splits_the_parent_tokens_by_doubt():
    m, ids = _model(), _ids()
    m.eval()
    r = routers(m)[1]
    r.record = True
    before = m(input_ids=ids).logits
    hs, _, idx = r.last
    parent = int(torch.bincount(idx.reshape(-1)).argmax())
    d = torch.randn(64)
    child = tree.split(m, torch.optim.SGD(m.parameters(), lr=0.0), 1, parent, d, noise=0.0)
    after = m(input_ids=ids).logits
    assert child == 6 and r.num_experts == 7 and moe(m, 1)[0].num_experts == 7
    assert torch.allclose(before, after, atol=1e-5)
    went = r.last[2]
    was_parent = (idx == parent).any(-1)
    to_child, to_parent = (went == child).any(-1) & was_parent, (went == parent).any(-1) & was_parent
    assert 0.3 < float(to_child.sum()) / float(was_parent.sum()) < 0.7  # la compuerta parte por la mitad
    s = hs.float() @ (d / d.norm())
    assert s[to_child].mean() > s[to_parent].mean()  # los dudosos van a la nueva


def test_freeze_old_keeps_the_old_daughter_still_and_the_gate_learns():
    m, ids = _model(), _ids()
    r = routers(m)[1]
    r.record = True
    m(input_ids=ids)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)  # como `train.adamw`
    parent = int(torch.bincount(r.last[2].reshape(-1)).argmax())
    child = tree.split(m, opt, 1, parent, torch.randn(64), freeze_old=True)
    experts = moe(m, 1)[0]
    old, new = experts.down_proj[parent].clone(), experts.down_proj[child].clone()
    for _ in range(5):
        opt.zero_grad()
        m(input_ids=ids, labels=ids).loss.backward()
        assert r.gates.grad is not None and r.gates.grad.abs().sum() > 0
        opt.step()
    assert torch.equal(experts.down_proj[parent], old) and not torch.equal(experts.down_proj[child], new)


def test_resize_rebuilds_the_tree_to_load_a_checkpoint():
    m, ids = _model(), _ids()
    for r in routers(m):
        r.record = True
    m(input_ids=ids)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    for layer in (0, 1, 1):
        parent = int(torch.bincount(routers(m)[layer].last[2].reshape(-1)).argmax())
        tree.split(m, opt, layer, parent, torch.randn(64))
        m(input_ids=ids)
    fresh = _model(seed=1)
    tree.resize(fresh, [r.num_experts for r in routers(m)])
    fresh.load_state_dict(m.state_dict())
    m.eval(), fresh.eval()
    assert torch.allclose(m(input_ids=ids).logits, fresh(input_ids=ids).logits, atol=1e-5)


def test_specialization_is_a_fraction_per_layer():
    m = _model()
    s = tree.specialization(m, {"a": _ids(seed=1), "b": _ids(seed=2)})
    assert len(s) == 2 and all(0.0 <= x <= 1.0 for x in s)
