import torch

from mf_moe import fractal_tree as FT
from mf_moe.granite import tiny
from mf_moe.lifelong import freeze_rows  # noqa: F401 - FT lo usa
from mf_moe.model import moe, routers

IDS = torch.randint(3, 250, (4, 16), generator=torch.Generator().manual_seed(0))


def _tree(seed=0):
    model = tiny(n_experts=6, seed=seed)
    model.config._experts_implementation = "grouped_mm_bf16"
    with torch.no_grad():
        before = model(input_ids=IDS).logits
    FT.install(model)
    for r in routers(model):
        r.record = True
    return model, before


def _logits(model):
    with torch.no_grad():
        return model(input_ids=IDS).logits


def test_without_mitosis_the_tree_routes_like_granite():
    model, before = _tree()
    assert torch.allclose(_logits(model), before, atol=1e-6)


def test_mitosis_and_nested_mitosis_keep_the_output_and_the_child_gets_tokens():
    model, _ = _tree()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
    ref = _logits(model)
    parent = int(routers(model)[1].last[2].reshape(-1).bincount().argmax())
    child = FT.split(model, opt, 1, parent, None)
    assert torch.allclose(_logits(model), ref, atol=1e-5)
    used = routers(model)[1].last[2].reshape(-1).tolist()
    assert child in used and parent in used  # la compuerta parte los tokens de la madre
    FT.split(model, opt, 1, child, None)  # nieta: hija de una hoja de bajo rango
    assert torch.allclose(_logits(model), ref, atol=1e-5)
    assert routers(model)[1].num_experts == moe(model, 1)[0].num_experts == 8


def test_training_moves_only_the_low_rank_delta_and_the_gates():
    model, _ = _tree()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    _logits(model)
    parent = int(routers(model)[1].last[2].reshape(-1).bincount().argmax())
    child = FT.split(model, opt, 1, parent, None)
    FT.freeze(model, opt, 1, [parent])
    experts, router = moe(model, 1)
    w0 = {n: getattr(experts, n).detach().clone() for n in FT.NAMES}
    u0, g0 = experts.lowrank.u_down_proj.detach().clone(), router.gate_b.detach().clone()
    for _ in range(5):
        model(input_ids=IDS, labels=IDS).loss.backward()
        opt.step()
        opt.zero_grad()
    experts, router = moe(model, 1)
    for n in FT.NAMES:
        w = getattr(experts, n).detach()
        assert torch.equal(w[parent], w0[n][parent]) and torch.equal(w[child], w0[n][child])
        assert not torch.equal(w, w0[n])  # las hojas no congeladas siguen aprendiendo
    assert not torch.equal(experts.lowrank.u_down_proj, u0) and not torch.equal(router.gate_b, g0)


def test_resize_rebuilds_the_tree_for_its_checkpoint():
    model, _ = _tree()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    _logits(model)
    c = FT.split(model, opt, 1, 0, None)
    FT.split(model, opt, 1, c, None)
    with torch.no_grad():
        moe(model, 1)[0].lowrank.u_down_proj.normal_()  # deltas no nulos: el retome tiene que traerlos
    ref, state = _logits(model), model.state_dict()
    fresh, _ = _tree()
    FT.resize(fresh, [r.num_experts for r in routers(model)])
    assert not fresh.load_state_dict(state, strict=False).unexpected_keys
    assert torch.allclose(_logits(fresh), ref, atol=1e-5)


def test_gate_balancing_pushes_a_lopsided_gate_toward_both_daughters():
    model, _ = _tree()
    _logits(model)
    FT.split(model, torch.optim.SGD(model.parameters(), lr=0.0), 1, 0, None)
    router = routers(model)[1]
    router.u, router.decay = 0.05, 0.5
    with torch.no_grad():
        router.gate_b[0] = torch.tensor([10.0, -10.0])  # la compuerta manda todo a la hija vieja
    model.train()
    with torch.no_grad():
        for _ in range(30):
            model(input_ids=IDS)
    assert router.gate_bias[0, 1] > router.gate_bias[0, 0]  # el sesgo empuja hacia la hija sin tokens


def test_one_level_tree_never_splits_a_daughter_and_the_free_tree_does():
    """H10 de la spec: con max_depth 1 todas las mitosis son sobre expertos de Granite; sin límite, hay nietas."""
    import mf_moe.phase2 as P
    from tests.test_phase2 import CFG, _data

    cfg = P.Cfg2(**{**CFG.__dict__, "check_every": 1, "n_final": 10})
    deep = {}
    for name in ("arbol_un_nivel+consolidacion", "arbol+consolidacion"):
        model = tiny(n_experts=6)
        P.run2(P.Cond2(name, divide=True, tree=True, max_depth=P.CONDS2[name].max_depth), 0, model, _data(), "cpu", cfg,
               log=lambda r: None)  # fmt: skip
        deep[name] = int(FT.leaf_depths(routers(model)[1]).max())
    assert deep["arbol_un_nivel+consolidacion"] == 1 and deep["arbol+consolidacion"] >= 2


def test_a_gate_whose_bias_overrides_a_near_zero_probability_keeps_finite_gradients():
    """Piloto 8 (2026-10-22): el sesgo de balanceo elige una hija que la compuerta casi descarta (p ~ 0); con el paso
    directo p / p.detach() el gradiente es 1/p y la corrida entera dio NaN en la GPU."""
    model, _ = _tree()
    _logits(model)
    FT.split(model, torch.optim.SGD(model.parameters(), lr=0.0), 1, 0, None)
    router = routers(model)[1]
    with torch.no_grad():
        router.gate_b[0] = torch.tensor([0.0, -1000.0])  # la hija nueva tiene probabilidad 0 en fp32
        router.gate_bias[0] = torch.tensor([0.0, 5.0])  # pero el sesgo la elige
    model.train()
    loss = model(input_ids=IDS, labels=IDS).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
