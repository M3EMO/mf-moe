import torch

from mf_moe.division import Divider, split
from mf_moe.granite import RecordingGraniteRouter, tiny
from mf_moe.merging import Merger, align, merge_pair
from mf_moe.model import moe, routers
from mf_moe.train import adamw


def _ids(n=4, length=12):
    return torch.randint(1, 256, (n, length))


def test_patched_router_routes_like_granite_and_records():
    model = tiny()
    r = routers(model)[0]
    assert isinstance(r, RecordingGraniteRouter) and torch.all(r.bias == 0)
    ids = _ids()
    ref = model(input_ids=ids).logits
    r.record = True
    out = model(input_ids=ids).logits
    assert torch.allclose(out, ref) and r.last[1].shape == (ids.numel(), r.num_experts)


def test_split_merge_and_align_work_on_granite_experts():
    model = tiny(n_experts=6)
    opt = adamw(model)
    ids = _ids()
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    split(model, opt, 1, 2, row_dir=torch.randn(64))
    experts, router = moe(model, 1)
    assert experts.num_experts == router.num_experts == 7 and router.load.shape == (7,)
    hs = torch.randn(200, 64)
    align(model, opt, 1, 0, 6, hs)
    merge_pair(model, opt, 1, 0, 6)
    assert moe(model, 1)[0].num_experts == 6
    model(input_ids=ids, labels=ids).loss.backward()  # sigue entrenando
    opt.step()


def test_divider_and_merger_run_on_granite():
    model = tiny(n_experts=6)
    opt = adamw(model)
    div = Divider(z=-1e9, every=1, max_experts=8)
    div.attach(model)
    ids = _ids()
    out = model(input_ids=ids, labels=ids)
    div(model, opt, 1, ids, out.logits, ids)
    assert [r.num_experts for r in routers(model)] == [7, 7]
    m = Merger(z=-1e9, min_experts=6, align=True)
    m(model, opt, ids)
    assert [r.num_experts for r in routers(model)] == [6, 6]


def test_a_silent_child_is_never_chosen_until_balancing_raises_its_bias():
    import mf_moe.division as D
    from mf_moe.model import routers

    model = tiny(n_experts=6)
    r = routers(model)[1]
    x = torch.randn(64, model.config.hidden_size)
    D.split(model, torch.optim.SGD(model.parameters(), lr=0.0), 1, 0, row_dir=None, silent=True)
    assert not (r(x)[0] == 6).any()
    r.u = 1e-2
    for _ in range(300):
        r(x)
    assert (r(x)[0] == 6).any()
