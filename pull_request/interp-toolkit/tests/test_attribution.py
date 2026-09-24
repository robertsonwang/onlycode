import torch

from interp_toolkit.attribution import layer_ablation_sweep, per_layer_logit_attribution
from interp_toolkit.hooks import run_with_cache
from interp_toolkit.model import MiniTransformer, ModelConfig


def make_model(seed: int = 0) -> MiniTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    return MiniTransformer(cfg)


def test_per_layer_logit_attribution_returns_a_value_per_layer():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))
    _, cache = run_with_cache(model, tokens)

    attribution = per_layer_logit_attribution(model, cache, correct_token=5, incorrect_token=9)

    assert attribution.shape[-1] == 1  # batch dimension


def test_layer_ablation_sweep_runs_for_every_layer():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))

    sweep = layer_ablation_sweep(model, tokens, correct_token=5, incorrect_token=9)

    assert sweep.numel() == model.cfg.n_layers
