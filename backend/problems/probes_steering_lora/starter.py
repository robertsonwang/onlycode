from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic representation data used by tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_probe_data(
    n: int = 240, d_model: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    direction = rng.normal(size=d_model)
    direction /= np.linalg.norm(direction)
    X = rng.normal(scale=0.55, size=(n, d_model))
    X += (2 * y - 1)[:, None] * 1.4 * direction
    return X, y


# ---------------------------------------------------------------------------
# Stage 1: Fit preprocessing on train only
# ---------------------------------------------------------------------------
def fit_standardizer(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Return `{"mean": ..., "std": ...}` using only X_train."""
    # TODO: implement
    raise NotImplementedError


def apply_standardizer(
    X: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    """Standardize X with previously fitted statistics."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Linear probe
# ---------------------------------------------------------------------------
def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_steps: int = 500,
    l2: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Train binary logistic regression; return weight and bias."""
    # TODO: implement
    raise NotImplementedError


def probe_predict(
    X: np.ndarray,
    probe: dict[str, np.ndarray | float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Return integer labels in {0, 1}."""
    # TODO: implement
    raise NotImplementedError


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of equal labels."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Direction discovery and causal interventions
# ---------------------------------------------------------------------------
def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the unit-normalized class-1 mean minus class-0 mean."""
    # TODO: implement
    raise NotImplementedError


def steer_activations(
    X: np.ndarray, direction: np.ndarray, strength: float
) -> np.ndarray:
    """Return X + strength * direction without mutating X."""
    # TODO: implement
    raise NotImplementedError


def project_out_direction(
    X: np.ndarray, direction: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Remove each row's component parallel to direction."""
    # TODO: implement
    raise NotImplementedError


def probe_accuracy_drop_after_ablation(
    X: np.ndarray,
    y: np.ndarray,
    probe: dict[str, np.ndarray | float],
    direction: np.ndarray,
) -> float:
    """Return accuracy(X) minus accuracy(project_out_direction(X))."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Low-rank adapter effect
# ---------------------------------------------------------------------------
def lora_delta(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Compute `(alpha / rank) * X @ A @ B`.

    A has shape (d_in, rank), B has shape (rank, d_out).
    """
    # TODO: implement
    raise NotImplementedError


def representation_shift_metrics(
    base: np.ndarray, adapted: np.ndarray, eps: float = 1e-12
) -> dict[str, float]:
    """Return float metrics: mean_l2, relative_l2, and mean_cosine."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Scan LoRA shifts across layers
# ---------------------------------------------------------------------------
def layerwise_lora_shifts(
    activations: list[np.ndarray],
    adapters: list[dict[str, np.ndarray | float]],
) -> list[dict[str, float]]:
    """
    Return one shift-metrics dict per `(activation, adapter)` pair.

    Each adapter has keys A, B, and alpha. The base output for a layer is the
    activation itself; the adapted output is activation + lora_delta(...).
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    test_name = input_data["test"]

    if test_name == "standardizer_uses_train_stats":
        train = np.array([[0.0, 2.0], [2.0, 6.0]])
        test = np.array([[10.0, 14.0]])
        stats = fit_standardizer(train)
        transformed = apply_standardizer(test, stats)
        assert np.allclose(stats["mean"], [1.0, 4.0])
        assert np.allclose(stats["std"], [1.0, 2.0])
        assert np.allclose(transformed, [[9.0, 5.0]])
        return True

    if test_name == "standardizer_handles_constant_columns":
        X = np.array([[1.0, 3.0], [1.0, 5.0], [1.0, 7.0]])
        stats = fit_standardizer(X)
        assert stats["std"][0] == 1.0
        assert np.all(np.isfinite(apply_standardizer(X, stats)))
        return True

    if test_name == "probe_learns_separable_data":
        X, y = make_probe_data(seed=2)
        train_X, test_X = X[:160], X[160:]
        train_y, test_y = y[:160], y[160:]
        stats = fit_standardizer(train_X)
        probe = train_linear_probe(apply_standardizer(train_X, stats), train_y, n_steps=600)
        predictions = probe_predict(apply_standardizer(test_X, stats), probe)
        assert accuracy(test_y, predictions) > 0.93
        return True

    if test_name == "probe_shapes_and_predictions":
        X, y = make_probe_data(n=40, d_model=6, seed=3)
        probe = train_linear_probe(X, y, n_steps=50)
        predictions = probe_predict(X, probe)
        assert probe["weight"].shape == (6,)
        assert np.isscalar(probe["bias"])
        assert predictions.shape == (40,)
        assert predictions.dtype.kind in ("i", "u")
        assert set(np.unique(predictions)).issubset({0, 1})
        return True

    if test_name == "mean_difference_direction_known":
        X = np.array([[0.0, 0.0], [0.0, 2.0], [3.0, 1.0], [3.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        expected = np.array([1.0, 0.0])
        assert np.allclose(mean_difference_direction(X, y), expected)
        return True

    if test_name == "steering_is_vectorized_and_pure":
        X = np.zeros((4, 3))
        before = X.copy()
        direction = np.array([1.0, -2.0, 0.5])
        result = steer_activations(X, direction, strength=2.0)
        assert np.array_equal(X, before)
        assert np.allclose(result, np.tile([2.0, -4.0, 1.0], (4, 1)))
        return True

    if test_name == "projection_removes_direction":
        rng = np.random.default_rng(4)
        X = rng.normal(size=(12, 5))
        direction = rng.normal(size=5)
        projected = project_out_direction(X, direction)
        assert np.allclose(projected @ direction, 0.0, atol=1e-9)
        return True

    if test_name == "ablation_reduces_probe_accuracy":
        X, y = make_probe_data(n=300, d_model=8, seed=5)
        direction = mean_difference_direction(X[:200], y[:200])
        probe = train_linear_probe(X[:200], y[:200], n_steps=700)
        drop = probe_accuracy_drop_after_ablation(X[200:], y[200:], probe, direction)
        assert drop > 0.25
        return True

    if test_name == "lora_delta_matches_formula":
        X = np.array([[1.0, 2.0], [-1.0, 3.0]])
        A = np.array([[1.0], [2.0]])
        B = np.array([[3.0, -1.0, 2.0]])
        expected = 0.5 * (X @ A @ B)
        assert np.allclose(lora_delta(X, A, B, alpha=0.5), expected)
        return True

    if test_name == "representation_shift_metrics_known":
        base = np.array([[3.0, 4.0], [1.0, 0.0]])
        adapted = np.array([[6.0, 8.0], [0.0, 1.0]])
        metrics = representation_shift_metrics(base, adapted)
        expected_mean_l2 = (5.0 + np.sqrt(2.0)) / 2.0
        expected_relative = np.linalg.norm(adapted - base) / np.linalg.norm(base)
        assert set(metrics) == {"mean_l2", "relative_l2", "mean_cosine"}
        assert np.isclose(metrics["mean_l2"], expected_mean_l2)
        assert np.isclose(metrics["relative_l2"], expected_relative)
        assert np.isclose(metrics["mean_cosine"], 0.5)
        return True

    if test_name == "layerwise_lora_shift_scan":
        rng = np.random.default_rng(6)
        activations = [rng.normal(size=(5, 4)), rng.normal(size=(5, 4))]
        adapters = [
            {"A": rng.normal(size=(4, 2)), "B": rng.normal(size=(2, 4)), "alpha": 2.0},
            {"A": np.zeros((4, 1)), "B": rng.normal(size=(1, 4)), "alpha": 1.0},
        ]
        metrics = layerwise_lora_shifts(activations, adapters)
        assert len(metrics) == 2
        assert metrics[0]["mean_l2"] > 0.0
        assert np.isclose(metrics[1]["mean_l2"], 0.0)
        assert np.isclose(metrics[1]["mean_cosine"], 1.0)
        return True

    raise ValueError(f"Unknown test: {test_name}")
