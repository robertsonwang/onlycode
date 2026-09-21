from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Tiny causal residual model used by tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
class TinyCausalModel:
    """A deterministic sequence model with residual mixing across positions."""

    def __init__(self, vocab_size=24, d_model=8, n_layers=4, seed=0):
        rng = np.random.default_rng(seed)
        self.n_layers = n_layers
        self.embedding = rng.normal(scale=0.35, size=(vocab_size, d_model))
        self.local_weights = rng.normal(scale=0.22, size=(n_layers, d_model, d_model))
        self.context_weights = rng.normal(scale=0.18, size=(n_layers, d_model, d_model))
        self.unembed = rng.normal(scale=0.3, size=(d_model, vocab_size))

    def forward(self, tokens, patch_layer=None, patch_mask=None, patch_values=None):
        tokens = np.asarray(tokens, dtype=int)
        x = self.embedding[tokens]
        cache = {}

        for layer in range(self.n_layers):
            # Prefix means make earlier positions able to affect later logits.
            prefix_sum = np.cumsum(x, axis=1)
            divisors = np.arange(1, x.shape[1] + 1, dtype=float)[None, :, None]
            prefix_mean = prefix_sum / divisors
            x = x + np.tanh(
                x @ self.local_weights[layer]
                + prefix_mean @ self.context_weights[layer]
            )

            if patch_layer == layer:
                if patch_mask is None or patch_values is None:
                    raise ValueError("patch_mask and patch_values are required")
                x = patch_positions(x, patch_values, patch_mask)

            cache[layer] = x.copy()

        logits = x @ self.unembed
        return logits, cache


# ---------------------------------------------------------------------------
# Stage 1: Metric
# ---------------------------------------------------------------------------
def logit_difference(
    logits: np.ndarray, positive_token_id: int, negative_token_id: int
) -> np.ndarray:
    """Return positive-minus-negative logits at the final sequence position."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Selective replacement
# ---------------------------------------------------------------------------
def patch_positions(
    corrupted_activation: np.ndarray,
    clean_activation: np.ndarray,
    position_mask: np.ndarray,
) -> np.ndarray:
    """
    Copy corrupted_activation and replace masked positions with clean values.

    Args:
        corrupted_activation: (batch, sequence, d_model)
        clean_activation: same shape
        position_mask: (batch, sequence), boolean or 0/1
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: One-layer causal intervention
# ---------------------------------------------------------------------------
def activation_patch_effect(
    model: TinyCausalModel,
    clean_tokens: np.ndarray,
    corrupted_tokens: np.ndarray,
    layer: int,
    position_mask: np.ndarray,
    positive_token_id: int,
    negative_token_id: int,
) -> dict[str, np.ndarray]:
    """
    Patch clean activations into the corrupted run.

    Returns a dict with exactly these keys, each mapped to shape `(batch,)`:
        clean_logit_diff, corrupted_logit_diff, patched_logit_diff, recovery
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Scan every layer
# ---------------------------------------------------------------------------
def scan_layers(
    model: TinyCausalModel,
    clean_tokens: np.ndarray,
    corrupted_tokens: np.ndarray,
    position_mask: np.ndarray,
    positive_token_id: int,
    negative_token_id: int,
) -> np.ndarray:
    """Return recovery for each layer, shaped (n_layers, batch)."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Normalize to the clean-corrupted behavior gap
# ---------------------------------------------------------------------------
def normalized_recovery(
    clean_score: np.ndarray,
    corrupted_score: np.ndarray,
    patched_score: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute (patched - corrupted) / (clean - corrupted), safely."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    test_name = input_data["test"]

    if test_name == "logit_difference_final_position":
        logits = np.zeros((2, 3, 5))
        logits[0, -1, 3] = 4.0
        logits[0, -1, 1] = 1.5
        logits[1, -1, 3] = -2.0
        logits[1, -1, 1] = 0.5
        assert np.allclose(logit_difference(logits, 3, 1), np.array([2.5, -2.5]))
        return True

    if test_name == "patch_positions_selective_and_pure":
        corrupted = np.zeros((2, 3, 2))
        clean = np.arange(12, dtype=float).reshape(2, 3, 2)
        corrupted_before = corrupted.copy()
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        patched = patch_positions(corrupted, clean, mask)
        assert np.array_equal(corrupted, corrupted_before)
        assert np.array_equal(patched[0, 0], clean[0, 0])
        assert np.array_equal(patched[1, 1], clean[1, 1])
        assert np.array_equal(patched[0, 1], np.zeros(2))
        return True

    if test_name == "patch_all_matches_clean_activation":
        rng = np.random.default_rng(1)
        corrupted = rng.normal(size=(2, 4, 3))
        clean = rng.normal(size=(2, 4, 3))
        mask = np.ones((2, 4), dtype=bool)
        assert np.allclose(patch_positions(corrupted, clean, mask), clean)
        return True

    if test_name == "single_layer_effect_matches_manual_run":
        model = TinyCausalModel(seed=2)
        clean = np.array([[2, 4, 6, 8], [1, 3, 5, 7]])
        corrupt = np.array([[9, 4, 6, 8], [10, 3, 5, 7]])
        mask = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
        result = activation_patch_effect(model, clean, corrupt, 1, mask, 3, 5)

        clean_logits, clean_cache = model.forward(clean)
        corrupt_logits, _ = model.forward(corrupt)
        patched_logits, _ = model.forward(corrupt, 1, mask, clean_cache[1])
        expected_patched = logit_difference(patched_logits, 3, 5)
        expected_corrupt = logit_difference(corrupt_logits, 3, 5)

        assert set(result) == {
            "clean_logit_diff", "corrupted_logit_diff", "patched_logit_diff", "recovery"
        }
        assert np.allclose(result["clean_logit_diff"], logit_difference(clean_logits, 3, 5))
        assert np.allclose(result["patched_logit_diff"], expected_patched)
        assert np.allclose(result["recovery"], expected_patched - expected_corrupt)
        return True

    if test_name == "no_positions_has_zero_recovery":
        model = TinyCausalModel(seed=3)
        clean = np.array([[1, 2, 3]])
        corrupt = np.array([[4, 2, 3]])
        mask = np.zeros_like(clean, dtype=bool)
        result = activation_patch_effect(model, clean, corrupt, 2, mask, 4, 6)
        assert np.allclose(result["recovery"], 0.0)
        return True

    if test_name == "scan_layers_shape_and_values":
        model = TinyCausalModel(n_layers=3, seed=4)
        clean = np.array([[1, 2, 3], [5, 6, 7]])
        corrupt = np.array([[8, 2, 3], [9, 6, 7]])
        mask = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        scan = scan_layers(model, clean, corrupt, mask, 2, 7)
        assert scan.shape == (3, 2)
        for layer in range(3):
            one = activation_patch_effect(model, clean, corrupt, layer, mask, 2, 7)
            assert np.allclose(scan[layer], one["recovery"])
        return True

    if test_name == "normalized_recovery_known":
        clean = np.array([5.0, 2.0])
        corrupt = np.array([1.0, 6.0])
        patched = np.array([3.0, 5.0])
        assert np.allclose(normalized_recovery(clean, corrupt, patched), [0.5, 0.25])
        return True

    if test_name == "normalized_recovery_is_finite":
        clean = np.array([1.0, 1.0])
        corrupt = np.array([1.0, 1.0])
        patched = np.array([2.0, 0.0])
        result = normalized_recovery(clean, corrupt, patched)
        assert np.all(np.isfinite(result))
        return True

    raise ValueError(f"Unknown test: {test_name}")
