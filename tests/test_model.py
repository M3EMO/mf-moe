import torch

from mf_moe.model import BiasBalancedRouter, build, config, routers


def test_sizes_match_the_architecture_ticket():
    small, full = build(6), build(12)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert 52e6 < n(small) < 53.5e6 and 90e6 < n(full) < 91.5e6


def test_router_is_initialized_and_bias_only_chooses():
    m = build(4, d=64, layers=2)
    r = routers(m)[0]
    assert isinstance(r, BiasBalancedRouter) and r.weight.abs().sum() > 0
    r.eval()
    x = torch.randn(5, 64)
    logits, _, _ = r(x)
    r.bias[:] = torch.tensor([10.0, 10.0, 10.0, -10.0])  # fuerza la elección de 0, 1 y 2
    _, w1, i1 = r(x)
    assert set(i1.flatten().tolist()) <= {0, 1, 2}
    probs = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(w1.float(), probs.gather(-1, i1), atol=1e-6)  # pesos sin sesgo


def test_bias_moves_toward_balance_when_training():
    cfg = config(4, d=64, layers=1)
    r = BiasBalancedRouter(cfg, u=0.01, decay=0.5)
    r.weight.data.zero_()
    r.weight.data[0] = 1.0  # el experto 0 se lleva todo
    x = torch.ones(32, 64)
    r.train()
    for _ in range(5):
        r(x)
    assert r.bias[0] < 0 and r.bias[3] > 0


def test_bf16_experts_match_fp32_and_train_fp32_weights():
    m = build(4, d=64, layers=1)
    assert m.config._experts_implementation == "grouped_mm_bf16"
    ids = torch.randint(1, 256, (2, 12))
    ref = m(input_ids=ids).logits
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    experts = m.model.layers[0].mlp.experts
    assert torch.allclose(out.logits.float(), ref, atol=0.1)
    assert experts.gate_up_proj.dtype == torch.float32 and experts.gate_up_proj.grad.abs().sum() > 0
