import torch

from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.steering import (
    compute_steering_vector,
    generate_with_steering,
    load_steering_vector,
    save_steering_vector,
    steering_sweep,
)

HOOK_NAME = "blocks.1.hook_resid_post"


def make_model(seed: int = 0) -> MiniTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    return MiniTransformer(cfg)


def test_compute_steering_vector_has_model_dim_shape():
    model = make_model()
    positive = torch.randint(0, model.cfg.d_vocab, (4, 6))
    negative = torch.randint(0, model.cfg.d_vocab, (4, 6))

    direction = compute_steering_vector(model, positive, negative, HOOK_NAME)

    assert direction.shape == (model.cfg.d_model,)


def test_generate_with_steering_preserves_logits_shape():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))
    direction = torch.randn(model.cfg.d_model)

    logits = generate_with_steering(model, tokens, HOOK_NAME, direction, coefficient=1.0)

    assert logits.shape == (1, 6, model.cfg.d_vocab)


def test_steering_sweep_returns_one_result_per_coefficient():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))
    direction = torch.randn(model.cfg.d_model)

    results = steering_sweep(model, tokens, HOOK_NAME, direction, coefficients=[0.0, 1.0, 2.0])

    assert set(results.keys()) == {0.0, 1.0, 2.0}


def test_save_and_load_steering_vector_round_trips(tmp_path):
    direction = torch.randn(32)

    save_steering_vector(direction, label="my-concept", cache_dir=tmp_path)
    reloaded = load_steering_vector(label="my-concept", cache_dir=tmp_path)

    assert torch.allclose(reloaded, direction)
