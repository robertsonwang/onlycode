import math

import torch

from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.sampling import (
    generate,
    sample_next_token,
    softmax_with_temperature,
    top_k_filter,
    top_p_filter,
)


def make_model(seed: int = 0) -> MiniTransformer:
    torch.manual_seed(seed)
    return MiniTransformer(ModelConfig(n_layers=2, d_model=32, n_heads=4, d_vocab=64, n_ctx=16))


def test_softmax_matches_torch():
    logits = torch.randn(3, 10)
    for temperature in (0.5, 1.0, 2.0):
        expected = torch.softmax(logits / temperature, dim=-1)
        assert torch.allclose(softmax_with_temperature(logits, temperature), expected, atol=1e-6)


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    assert sample_next_token(logits, temperature=0).tolist() == [1, 0]


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = top_k_filter(logits, k=2)
    kept = torch.isfinite(filtered[0]).nonzero().flatten().tolist()
    assert kept == [1, 4]


def test_top_p_keeps_nucleus():
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    filtered = top_p_filter(probs.log(), top_p=0.75)
    kept = torch.isfinite(filtered[0]).nonzero().flatten().tolist()
    assert kept == [0, 1]


def test_filtered_tokens_are_never_sampled():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    gen = torch.Generator().manual_seed(0)
    samples = {sample_next_token(logits, top_k=2, generator=gen).item() for _ in range(200)}
    assert samples <= {1, 4}


def test_generate_appends_tokens_and_keeps_prompt():
    model = make_model()
    prompt = torch.randint(0, model.cfg.d_vocab, (1, 4))
    out = generate(model, prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], prompt)


def test_generate_is_reproducible_with_generator():
    model = make_model()
    prompt = torch.randint(0, model.cfg.d_vocab, (1, 4))
    a = generate(model, prompt, 6, top_p=0.9, generator=torch.Generator().manual_seed(1))
    b = generate(model, prompt, 6, top_p=0.9, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert not math.isnan(a.float().sum().item())
