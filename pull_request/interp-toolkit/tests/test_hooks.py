import torch

from interp_toolkit.hooks import run_with_cache, run_with_patch
from interp_toolkit.model import MiniTransformer, ModelConfig


def make_model(seed: int = 0) -> MiniTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(n_layers=2, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    return MiniTransformer(cfg)


def test_forward_pass_shape():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, model.cfg.d_vocab)


def test_run_with_cache_matches_plain_forward():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 8))

    plain_logits = model(tokens)
    cached_logits, cache = run_with_cache(model, tokens)

    assert torch.allclose(plain_logits, cached_logits)
    assert "blocks.0.hook_resid_post" in cache
    assert "blocks.1.hook_resid_post" in cache
    assert cache["blocks.0.hook_resid_post"].shape == (1, 8, model.cfg.d_model)


def test_run_with_patch_recovers_cached_activation():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 8))

    baseline_logits, cache = run_with_cache(model, tokens)

    # Patching every hook point back in with its own cached value should
    # reproduce the original forward pass exactly.
    patched_logits = run_with_patch(model, tokens, cache)

    assert torch.allclose(baseline_logits, patched_logits, atol=1e-5)


def test_run_with_patch_changes_output_when_activation_differs():
    model = make_model()
    tokens_a = torch.randint(0, model.cfg.d_vocab, (1, 8))
    tokens_b = torch.randint(0, model.cfg.d_vocab, (1, 8))

    logits_a = model(tokens_a)
    _, cache_b = run_with_cache(model, tokens_b)

    patch_name = "blocks.0.hook_resid_post"
    patched_logits = run_with_patch(model, tokens_a, {patch_name: cache_b[patch_name]})

    assert not torch.allclose(logits_a, patched_logits)


def test_unknown_hook_name_raises():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 8))
    try:
        run_with_patch(model, tokens, {"not.a.real.hook": torch.zeros(1)})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown hook name")
