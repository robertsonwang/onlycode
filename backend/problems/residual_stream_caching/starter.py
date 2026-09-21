from collections.abc import Sequence
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Tiny hookable residual model used by the tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
class _HookHandle:
    def __init__(self, hooks: dict[int, list], layer: int, callback: Callable):
        self._hooks = hooks
        self._layer = layer
        self._callback = callback
        self._removed = False

    def remove(self):
        if not self._removed:
            self._hooks[self._layer].remove(self._callback)
            self._removed = True


class TinyResidualModel:
    """A deterministic CPU-only stand-in for a hookable transformer."""

    def __init__(self, vocab_size=32, d_model=6, n_layers=4, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = rng.normal(scale=0.3, size=(vocab_size, d_model))
        self.weights = rng.normal(scale=0.25, size=(n_layers, d_model, d_model))
        self.context_weights = rng.normal(scale=0.15, size=(n_layers, d_model, d_model))
        self.unembed = rng.normal(scale=0.2, size=(d_model, vocab_size))
        self._hooks = {layer: [] for layer in range(n_layers)}
        self.forward_calls = 0
        self.fail_on_forward = False

    @property
    def active_hook_count(self):
        return sum(len(callbacks) for callbacks in self._hooks.values())

    def register_forward_hook(self, layer: int, callback: Callable):
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer must be in [0, {self.n_layers})")
        self._hooks[layer].append(callback)
        return _HookHandle(self._hooks, layer, callback)

    def forward(self, tokens, attention_mask):
        self.forward_calls += 1
        if self.fail_on_forward:
            raise RuntimeError("simulated forward failure")

        mask = np.asarray(attention_mask, dtype=bool)
        x = self.embedding[np.asarray(tokens, dtype=int)]
        x = x * mask[..., None]

        for layer in range(self.n_layers):
            counts = np.maximum(mask.sum(axis=1, keepdims=True), 1)
            context = x.sum(axis=1) / counts
            update = np.tanh(
                x @ self.weights[layer]
                + (context @ self.context_weights[layer])[:, None, :]
            )
            x = (x + update) * mask[..., None]
            for callback in list(self._hooks[layer]):
                replacement = callback(layer, x)
                if replacement is not None:
                    x = np.asarray(replacement)

        return x @ self.unembed


# ---------------------------------------------------------------------------
# Stage 1: Pad variable-length token sequences
# ---------------------------------------------------------------------------
def pad_sequences(
    sequences: Sequence[Sequence[int]], pad_id: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tokens, attention_mask) with right padding."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Mask-aware pooling
# ---------------------------------------------------------------------------
def masked_mean(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Average hidden states over non-padding positions.

    Args:
        hidden: array shaped (batch, sequence, d_model)
        attention_mask: boolean/0-1 array shaped (batch, sequence)

    Returns:
        Array shaped (batch, d_model).
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Extract one residual stream safely
# ---------------------------------------------------------------------------
def extract_residual_stream(
    model: TinyResidualModel,
    tokens: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
) -> np.ndarray:
    """
    Capture the post-block residual stream at one layer.

    The hook must be removed even if model.forward raises. Store a copy rather
    than a view so later mutation cannot change the returned activation.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Cache selected layers in bounded batches
# ---------------------------------------------------------------------------
def cache_residual_stream(
    model: TinyResidualModel,
    sequences: Sequence[Sequence[int]],
    layers: Sequence[int],
    batch_size: int = 8,
    pad_id: int = 0,
) -> dict[int, np.ndarray]:
    """
    Cache selected residual streams using one forward pass per input batch.

    Returns:
        dict[int, np.ndarray], one `(n, sequence, d_model)` array per layer.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    test_name = input_data["test"]

    if test_name == "pad_sequences_values":
        tokens, mask = pad_sequences([[4, 5, 6], [7], [8, 9]], pad_id=0)
        assert tokens.dtype.kind in ("i", "u")
        assert np.array_equal(tokens, np.array([[4, 5, 6], [7, 0, 0], [8, 9, 0]]))
        assert np.array_equal(mask, np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool))
        return True

    if test_name == "pad_sequences_rejects_empty":
        for invalid in ([], [[1], []]):
            try:
                pad_sequences(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("empty batches and sequences must be rejected")
        return True

    if test_name == "masked_mean_ignores_padding":
        hidden = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
                [[2.0, 6.0], [50.0, 50.0], [60.0, 60.0]],
            ]
        )
        mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        expected = np.array([[2.0, 3.0], [2.0, 6.0]])
        assert np.allclose(masked_mean(hidden, mask), expected)
        return True

    if test_name == "extract_matches_direct_cache":
        model = TinyResidualModel(seed=3)
        tokens, mask = pad_sequences([[1, 2, 3], [4, 5]])
        direct = {}
        def save_activation(layer, value):
            direct[layer] = value.copy()

        handle = model.register_forward_hook(2, save_activation)
        model.forward(tokens, mask)
        handle.remove()
        actual = extract_residual_stream(model, tokens, mask, layer=2)
        assert np.allclose(actual, direct[2])
        return True

    if test_name == "extract_does_not_leak_hooks":
        model = TinyResidualModel(seed=4)
        tokens, mask = pad_sequences([[1, 2], [3]])
        first = extract_residual_stream(model, tokens, mask, layer=1)
        second = extract_residual_stream(model, tokens, mask, layer=1)
        assert np.allclose(first, second)
        assert model.active_hook_count == 0
        return True

    if test_name == "extract_cleans_up_after_error":
        model = TinyResidualModel(seed=5)
        model.fail_on_forward = True
        tokens, mask = pad_sequences([[1, 2]])
        try:
            extract_residual_stream(model, tokens, mask, layer=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the model error should propagate")
        assert model.active_hook_count == 0
        return True

    if test_name == "batched_cache_shapes_and_order":
        model = TinyResidualModel(d_model=5, seed=6)
        sequences = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11, 12]]
        cached = cache_residual_stream(model, sequences, layers=[0, 3], batch_size=2)
        assert set(cached) == {0, 3}
        assert cached[0].shape == (5, 4, 5)
        assert cached[3].shape == (5, 4, 5)
        return True

    if test_name == "batched_cache_matches_full_batch":
        sequences = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11, 12]]
        batched_model = TinyResidualModel(seed=7)
        full_model = TinyResidualModel(seed=7)
        batched = cache_residual_stream(batched_model, sequences, [1, 2], batch_size=2)
        full = cache_residual_stream(full_model, sequences, [1, 2], batch_size=20)
        assert np.allclose(batched[1], full[1])
        assert np.allclose(batched[2], full[2])
        assert batched_model.active_hook_count == 0
        return True

    if test_name == "batched_cache_one_forward_per_batch":
        model = TinyResidualModel(seed=8)
        sequences = [[i + 1, i + 2] for i in range(7)]
        cache_residual_stream(model, sequences, layers=[0, 1, 2], batch_size=3)
        assert model.forward_calls == 3
        assert model.active_hook_count == 0
        return True

    raise ValueError(f"Unknown test: {test_name}")
