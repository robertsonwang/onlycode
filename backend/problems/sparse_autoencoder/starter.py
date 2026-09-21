from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic activation data used by tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_sparse_activations(
    n: int = 96, d_input: int = 8, n_sources: int = 12, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(n_sources, d_input))
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=1, keepdims=True), 1e-8)
    codes = np.zeros((n, n_sources))
    for row in range(n):
        active = rng.choice(n_sources, size=2, replace=False)
        codes[row, active] = rng.uniform(0.5, 2.0, size=2)
    return codes @ dictionary + rng.normal(scale=0.01, size=(n, d_input))


# ---------------------------------------------------------------------------
# Stage 1: Initialize and run the SAE
# ---------------------------------------------------------------------------
def init_sae(d_input: int, d_hidden: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Return a deterministic parameter dict for a ReLU sparse autoencoder."""
    # TODO: implement
    raise NotImplementedError


def sae_forward(
    X: np.ndarray, params: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Return `(reconstruction, cache)`.

    cache must contain exactly the keys `X`, `pre`, and `features`.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Reconstruction plus sparsity objective
# ---------------------------------------------------------------------------
def sae_loss(
    X: np.ndarray,
    reconstruction: np.ndarray,
    features: np.ndarray,
    l1_coeff: float,
) -> dict[str, float]:
    """Return a dict with float values: mse, l1, and total."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Analytic gradients
# ---------------------------------------------------------------------------
def sae_gradients(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    l1_coeff: float,
) -> dict[str, np.ndarray]:
    """Return gradients dW_enc, db_enc, dW_dec, and db_dec."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Remove the encoder/decoder scaling loophole
# ---------------------------------------------------------------------------
def normalize_decoder(
    params: dict[str, np.ndarray], eps: float = 1e-8
) -> dict[str, np.ndarray]:
    """Return copied params with every W_dec row normalized to unit norm."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Full-batch training
# ---------------------------------------------------------------------------
def train_sae(
    X: np.ndarray,
    d_hidden: int = 16,
    l1_coeff: float = 0.02,
    lr: float = 0.05,
    n_steps: int = 600,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """
    Train with full-batch gradient descent.

    Return `(params, history)`, with one loss dict per optimization step.
    Apply decoder normalization after every parameter update.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Evaluate sparsity explicitly
# ---------------------------------------------------------------------------
def feature_statistics(
    features: np.ndarray, threshold: float = 1e-8
) -> dict[str, float]:
    """Return fraction_active, mean_l0, and mean_activation as floats."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    test_name = input_data["test"]

    if test_name == "init_shapes_and_reproducibility":
        p1 = init_sae(5, 9, seed=1)
        p2 = init_sae(5, 9, seed=1)
        assert set(p1) == {"W_enc", "b_enc", "W_dec", "b_dec"}
        assert p1["W_enc"].shape == (5, 9)
        assert p1["b_enc"].shape == (9,)
        assert p1["W_dec"].shape == (9, 5)
        assert p1["b_dec"].shape == (5,)
        assert all(np.allclose(p1[key], p2[key]) for key in p1)
        assert not np.allclose(p1["W_enc"], 0.0)
        return True

    if test_name == "forward_shapes_and_nonnegative_features":
        X = make_sparse_activations(n=7, d_input=5, seed=2)
        params = init_sae(5, 11, seed=2)
        reconstruction, cache = sae_forward(X, params)
        assert reconstruction.shape == X.shape
        assert set(cache) == {"X", "pre", "features"}
        assert cache["features"].shape == (7, 11)
        assert np.all(cache["features"] >= 0.0)
        return True

    if test_name == "loss_components_known":
        X = np.array([[1.0, 3.0]])
        reconstruction = np.array([[2.0, 1.0]])
        features = np.array([[0.0, 2.0, 4.0]])
        losses = sae_loss(X, reconstruction, features, l1_coeff=0.3)
        assert set(losses) == {"mse", "l1", "total"}
        assert np.isclose(losses["mse"], 2.5)
        assert np.isclose(losses["l1"], 2.0)
        assert np.isclose(losses["total"], 3.1)
        return True

    if test_name == "gradient_shapes":
        X = make_sparse_activations(n=6, d_input=4, seed=3)
        params = init_sae(4, 7, seed=3)
        _, cache = sae_forward(X, params)
        grads = sae_gradients(X, params, cache, l1_coeff=0.04)
        assert set(grads) == {"dW_enc", "db_enc", "dW_dec", "db_dec"}
        assert grads["dW_enc"].shape == params["W_enc"].shape
        assert grads["db_enc"].shape == params["b_enc"].shape
        assert grads["dW_dec"].shape == params["W_dec"].shape
        assert grads["db_dec"].shape == params["b_dec"].shape
        return True

    if test_name == "gradient_check":
        rng = np.random.default_rng(4)
        X = rng.normal(size=(5, 3))
        params = init_sae(3, 4, seed=4)
        # Move biases away from ReLU's nondifferentiable zero for this check.
        params["b_enc"] += np.array([0.3, -0.4, 0.5, -0.6])
        reconstruction, cache = sae_forward(X, params)
        grads = sae_gradients(X, params, cache, l1_coeff=0.03)
        eps = 1e-5

        for pname, gname in (("W_enc", "dW_enc"), ("b_enc", "db_enc"), ("W_dec", "dW_dec"), ("b_dec", "db_dec")):
            index = tuple(0 for _ in params[pname].shape)
            plus = {key: value.copy() for key, value in params.items()}
            minus = {key: value.copy() for key, value in params.items()}
            plus[pname][index] += eps
            minus[pname][index] -= eps
            plus_recon, plus_cache = sae_forward(X, plus)
            minus_recon, minus_cache = sae_forward(X, minus)
            plus_loss = sae_loss(X, plus_recon, plus_cache["features"], 0.03)["total"]
            minus_loss = sae_loss(X, minus_recon, minus_cache["features"], 0.03)["total"]
            numeric = (plus_loss - minus_loss) / (2 * eps)
            assert np.isclose(grads[gname][index], numeric, atol=2e-4, rtol=2e-3), (
                f"{gname}{index}: analytic={grads[gname][index]}, numeric={numeric}"
            )
        return True

    if test_name == "normalize_decoder_rows":
        params = init_sae(4, 6, seed=5)
        original = params["W_dec"].copy()
        normalized = normalize_decoder(params)
        assert np.allclose(np.linalg.norm(normalized["W_dec"], axis=1), 1.0)
        assert np.array_equal(params["W_dec"], original)
        assert not np.shares_memory(normalized["W_dec"], params["W_dec"])
        return True

    if test_name == "training_reduces_total_loss":
        X = make_sparse_activations(n=64, d_input=6, seed=6)
        params, history = train_sae(X, d_hidden=12, l1_coeff=0.01, lr=0.04, n_steps=500, seed=6)
        assert len(history) == 500
        assert history[-1]["total"] < history[0]["total"] * 0.7
        final_reconstruction, _ = sae_forward(X, params)
        assert np.mean((final_reconstruction - X) ** 2) < 0.12
        return True

    if test_name == "feature_statistics_known":
        features = np.array([[0.0, 2.0, 0.0, 4.0], [1.0, 0.0, 0.0, 3.0]])
        stats = feature_statistics(features)
        assert set(stats) == {"fraction_active", "mean_l0", "mean_activation"}
        assert np.isclose(stats["fraction_active"], 0.5)
        assert np.isclose(stats["mean_l0"], 2.0)
        assert np.isclose(stats["mean_activation"], 1.25)
        return True

    if test_name == "sparsity_penalty_reduces_activity":
        X = make_sparse_activations(n=72, d_input=6, seed=7)
        dense_params, _ = train_sae(X, d_hidden=12, l1_coeff=0.0, lr=0.035, n_steps=400, seed=7)
        sparse_params, _ = train_sae(X, d_hidden=12, l1_coeff=0.15, lr=0.035, n_steps=400, seed=7)
        _, dense_cache = sae_forward(X, dense_params)
        _, sparse_cache = sae_forward(X, sparse_params)
        dense_activity = feature_statistics(dense_cache["features"])["mean_activation"]
        sparse_activity = feature_statistics(sparse_cache["features"])["mean_activation"]
        assert sparse_activity < dense_activity
        return True

    raise ValueError(f"Unknown test: {test_name}")
