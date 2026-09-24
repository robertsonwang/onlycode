import torch

from interp_toolkit.cache_io import load_cache, save_cache
from interp_toolkit.hooks import run_with_cache
from interp_toolkit.model import MiniTransformer, ModelConfig


def test_save_and_load_cache_round_trips(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    tokens = torch.randint(0, cfg.d_vocab, (1, 6))
    _, cache = run_with_cache(model, tokens)

    path = tmp_path / "cache.pt"
    save_cache(cache, path)
    reloaded = load_cache(path)

    assert set(reloaded.keys()) == set(cache.keys())
    for key in cache:
        assert torch.allclose(reloaded[key], cache[key])
