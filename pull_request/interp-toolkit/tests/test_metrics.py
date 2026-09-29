import torch

from interp_toolkit.metrics import (
    kl_divergence,
    logit_diff,
    patching_effect,
    perplexity,
    sequence_log_probs,
)


def test_logit_diff_basic():
    logits = torch.zeros(1, 3, 5)
    logits[0, -1, 2] = 3.0
    logits[0, -1, 4] = 1.0
    diff = logit_diff(logits, correct_token=2, incorrect_token=4)
    assert torch.allclose(diff, torch.tensor([2.0]))


def test_kl_divergence_zero_for_identical_logits():
    logits = torch.randn(2, 4, 6)
    kl = kl_divergence(logits, logits)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-6)


def test_patching_effect_bounds():
    baseline = torch.tensor([2.0])
    corrupted = torch.tensor([0.0])
    assert torch.allclose(patching_effect(baseline, baseline, corrupted), torch.tensor([1.0]))
    assert torch.allclose(patching_effect(baseline, corrupted, corrupted), torch.tensor([0.0]))


def test_sequence_log_probs_shape_and_alignment():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 2] = 10.0  # position 0 confidently predicts token 2
    tokens = torch.tensor([[0, 2, 1]])
    lp = sequence_log_probs(logits, tokens)
    assert lp.shape == (1, 2)
    assert lp[0, 0] > -1e-3


def test_perplexity_of_uniform_logits_is_vocab_size():
    logits = torch.zeros(2, 5, 7)
    tokens = torch.randint(0, 7, (2, 5))
    assert torch.allclose(perplexity(logits, tokens), torch.tensor(7.0))


def test_perplexity_ignores_padding():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 10)
    tokens = torch.randint(0, 10, (2, 6))
    mask = torch.ones(2, 6)
    mask[1, 4:] = 0

    ppl = perplexity(logits, tokens, attention_mask=mask)

    assert torch.isfinite(ppl)
    assert ppl > 1.0
