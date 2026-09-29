import torch

from interp_toolkit.metrics import kl_divergence, logit_diff, patching_effect


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
