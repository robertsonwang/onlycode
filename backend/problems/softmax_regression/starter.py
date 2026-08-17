import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_simple_three_class_data():
    X = np.array([
        [-3.0, -3.0],
        [-2.0, -3.0],
        [-3.0, -2.0],
        [3.0, -3.0],
        [2.0, -3.0],
        [3.0, -2.0],
        [0.0, 3.0],
        [1.0, 3.0],
        [0.0, 2.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y


# ---------------------------------------------------------------------------
# Stage 1: Softmax
# ---------------------------------------------------------------------------
def softmax(logits):
    """
    Numerically stable softmax, applied row-wise.

    Args:
        logits: np.ndarray, shape (n_samples, n_classes) or (n_classes,)

    Returns:
        np.ndarray, same shape, rows sum to 1

    Hint: subtract the row max before exponentiating to avoid overflow.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: One-hot encoding
# ---------------------------------------------------------------------------
def one_hot_encode(y, num_classes):
    """
    Convert integer class labels into one-hot vectors.

    Args:
        y: np.ndarray, shape (n_samples,), integer labels
        num_classes: int

    Returns:
        np.ndarray, shape (n_samples, num_classes)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: predict_proba
# ---------------------------------------------------------------------------
def predict_proba(X, W):
    """
    Compute class probabilities: softmax(X @ W).

    Args:
        X: shape (n_samples, n_features)
        W: shape (n_features, n_classes)

    Returns:
        shape (n_samples, n_classes)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: prediction
# ---------------------------------------------------------------------------
def predict(X, W):
    """
    Predict the most likely class for each example.

    Returns:
        np.ndarray, shape (n_samples,), dtype int
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: cross-entropy loss
# ---------------------------------------------------------------------------
def cross_entropy_loss(y, probabilities):
    """
    Compute the mean cross-entropy loss.

    Args:
        y: integer class labels, shape (n_samples,)
        probabilities: predicted probabilities, shape (n_samples, n_classes)

    Returns:
        float

    Hint: clip probabilities away from zero before taking log.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5b: gradient computation
# ---------------------------------------------------------------------------
def compute_gradient(X, y, W):
    """
    Compute the gradient of the mean cross-entropy loss w.r.t. W.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,), integer class labels
        W: np.ndarray, shape (n_features, n_classes)

    Returns:
        np.ndarray, shape (n_features, n_classes)

    Hint:
        1. Compute probabilities = softmax(X @ W).
        2. One-hot encode y.
        3. The gradient is:

               (1 / n_samples) * X.T @ (probabilities - y_one_hot)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: training
# ---------------------------------------------------------------------------
def fit(X, y, num_classes, learning_rate=0.1, epochs=1000):
    """
    Train softmax regression using gradient descent.

    Args:
        X: shape (n_samples, n_features)
        y: integer class labels, shape (n_samples,)
        num_classes: int
        learning_rate: float
        epochs: int

    Returns:
        W: learned weight matrix, shape (n_features, n_classes)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1: Softmax ===
    if test_name == "softmax_known_values":
        logits = np.array([[1.0, 2.0, 3.0]])
        probs = softmax(logits)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        assert np.allclose(probs.flatten(), expected, atol=1e-6)
        return True

    if test_name == "softmax_sums_to_one":
        logits = np.array([[0.5, -1.0, 2.0, 4.0]])
        probs = softmax(logits)
        assert np.isclose(probs.sum(), 1.0)
        return True

    if test_name == "softmax_probabilities_are_valid":
        logits = np.array([[-2.0, 0.0, 1.0, 5.0]])
        probs = softmax(logits)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        return True

    if test_name == "softmax_largest_logit_gets_largest_probability":
        logits = np.array([[1.0, 4.0, 2.0]])
        probs = softmax(logits)
        assert np.argmax(probs) == 1
        return True

    if test_name == "softmax_shift_invariance":
        logits = np.array([[1.0, 2.0, 3.0]])
        probs1 = softmax(logits)
        probs2 = softmax(logits + 100.0)
        assert np.allclose(probs1, probs2)
        return True

    if test_name == "softmax_numerically_stable":
        logits = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(logits)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        assert np.all(np.isfinite(probs))
        assert np.isclose(probs.sum(), 1.0)
        assert np.allclose(probs.flatten(), expected, atol=1e-6)
        return True

    if test_name == "softmax_single_value":
        logits = np.array([[42.0]])
        probs = softmax(logits)
        assert np.isclose(probs.flatten()[0], 1.0)
        return True

    if test_name == "softmax_equal_logits":
        logits = np.array([[5.0, 5.0, 5.0, 5.0]])
        probs = softmax(logits)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.allclose(probs.flatten(), expected)
        return True

    # === Stage 2: One-hot encoding ===
    if test_name == "one_hot_encode_basic":
        y = np.array([0, 2, 1])
        result = one_hot_encode(y, num_classes=3)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.array_equal(result, expected)
        return True

    if test_name == "one_hot_encode_shape":
        y = np.array([0, 1, 2, 1, 0])
        result = one_hot_encode(y, num_classes=3)
        assert result.shape == (5, 3)
        return True

    if test_name == "one_hot_encode_each_row_sums_to_one":
        y = np.array([2, 0, 1, 2])
        result = one_hot_encode(y, num_classes=3)
        assert np.all(result.sum(axis=1) == 1)
        return True

    if test_name == "one_hot_encode_contains_only_zero_and_one":
        y = np.array([0, 2, 1, 0])
        result = one_hot_encode(y, num_classes=3)
        assert set(np.unique(result)).issubset({0, 1})
        return True

    if test_name == "one_hot_encode_larger_number_of_classes":
        y = np.array([0, 4, 2])
        result = one_hot_encode(y, num_classes=5)
        expected = np.array([[1,0,0,0,0], [0,0,0,0,1], [0,0,1,0,0]])
        assert np.array_equal(result, expected)
        return True

    # === Stage 3: predict_proba ===
    if test_name == "predict_proba_shape":
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
        probs = predict_proba(X, W)
        assert probs.shape == (3, 3)
        return True

    if test_name == "predict_proba_rows_sum_to_one":
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
        probs = predict_proba(X, W)
        assert np.allclose(probs.sum(axis=1), 1.0)
        return True

    if test_name == "predict_proba_known_case":
        X = np.array([[1.0, 1.0]])
        W = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
        probs = predict_proba(X, W)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        assert probs.shape == (1, 3)
        assert np.allclose(probs[0], expected, atol=1e-6)
        return True

    if test_name == "predict_proba_uses_matrix_multiplication":
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        W = np.array([[3.0, 0.0], [0.0, 2.0]])
        probs = predict_proba(X, W)
        expected_first = softmax(np.array([[3.0, 0.0]]))
        expected_second = softmax(np.array([[0.0, 2.0]]))
        assert np.allclose(probs[0], expected_first)
        assert np.allclose(probs[1], expected_second)
        return True

    # === Stage 4: Prediction ===
    if test_name == "predict_returns_class_with_largest_probability":
        X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        W = np.array([[3.0, 0.0, 1.0], [0.0, 2.0, 4.0]])
        preds = predict(X, W)
        assert np.array_equal(preds, [0, 2, 2])
        return True

    if test_name == "predict_shape":
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        W = np.zeros((2, 3))
        preds = predict(X, W)
        assert preds.shape == (4,)
        return True

    if test_name == "predict_returns_valid_class_indices":
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        W = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
        preds = predict(X, W)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        return True

    # === Stage 5: Cross-entropy loss ===
    if test_name == "cross_entropy_perfect_predictions":
        y = np.array([0, 1, 2])
        probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        loss = cross_entropy_loss(y, probabilities)
        assert np.isclose(loss, 0.0, atol=1e-4)
        return True

    if test_name == "cross_entropy_known_value":
        y = np.array([0])
        probabilities = np.array([[0.5, 0.3, 0.2]])
        loss = cross_entropy_loss(y, probabilities)
        assert np.isclose(loss, -np.log(0.5))
        return True

    if test_name == "cross_entropy_multiple_examples":
        y = np.array([0, 1])
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
        expected = (-np.log(0.8) - np.log(0.75)) / 2
        loss = cross_entropy_loss(y, probabilities)
        assert np.isclose(loss, expected)
        return True

    if test_name == "cross_entropy_penalizes_confident_wrong_predictions":
        y = np.array([0])
        good = np.array([[0.9, 0.1]])
        bad = np.array([[0.1, 0.9]])
        good_loss = cross_entropy_loss(y, good)
        bad_loss = cross_entropy_loss(y, bad)
        assert bad_loss > good_loss
        return True

    if test_name == "cross_entropy_is_finite_for_small_probabilities":
        y = np.array([0])
        probabilities = np.array([[1e-12, 1.0 - 1e-12]])
        loss = cross_entropy_loss(y, probabilities)
        assert np.isfinite(loss)
        return True

    # === Stage 5b: Gradient computation ===
    if test_name == "compute_gradient_shape":
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 2])
        W = np.zeros((2, 3))
        grad = compute_gradient(X, y, W)
        assert grad.shape == (2, 3)
        return True

    if test_name == "compute_gradient_numeric_check":
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 3))
        y = rng.integers(0, 4, size=10)
        W = rng.normal(size=(3, 4)) * 0.1
        grad = compute_gradient(X, y, W)
        eps = 1e-5
        numeric_grad = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.copy()
                W_minus = W.copy()
                W_plus[i, j] += eps
                W_minus[i, j] -= eps
                probs_plus = predict_proba(X, W_plus)
                probs_minus = predict_proba(X, W_minus)
                loss_plus = cross_entropy_loss(y, probs_plus)
                loss_minus = cross_entropy_loss(y, probs_minus)
                numeric_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
        assert np.allclose(grad, numeric_grad, atol=1e-4)
        return True

    if test_name == "compute_gradient_at_uniform_predictions":
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0, 1])
        W = np.zeros((2, 2))
        grad = compute_gradient(X, y, W)
        assert not np.allclose(grad, 0.0)
        assert np.all(np.isfinite(grad))
        return True

    # === Stage 6: Training ===
    if test_name == "fit_returns_weight_matrix":
        X, y = make_simple_three_class_data()
        W = fit(X, y, num_classes=3, learning_rate=0.1, epochs=100)
        assert W.shape == (2, 3)
        return True

    if test_name == "fit_returns_finite_weights":
        X, y = make_simple_three_class_data()
        W = fit(X, y, num_classes=3, learning_rate=0.1, epochs=100)
        assert np.all(np.isfinite(W))
        return True

    if test_name == "fit_learns_simple_three_class_problem":
        X, y = make_simple_three_class_data()
        W = fit(X, y, num_classes=3, learning_rate=0.1, epochs=1000)
        preds = predict(X, W)
        acc = np.mean(preds == y)
        assert acc >= 0.95, f"training accuracy was only {acc:.3f}"
        return True

    if test_name == "fit_improves_loss":
        X, y = make_simple_three_class_data()
        W_before = np.zeros((X.shape[1], 3))
        initial_probs = predict_proba(X, W_before)
        initial_loss = cross_entropy_loss(y, initial_probs)
        W_after = fit(X, y, num_classes=3, learning_rate=0.1, epochs=500)
        final_probs = predict_proba(X, W_after)
        final_loss = cross_entropy_loss(y, final_probs)
        assert final_loss < initial_loss
        return True

    if test_name == "fit_produces_better_than_chance_accuracy":
        X, y = make_simple_three_class_data()
        W = fit(X, y, num_classes=3, learning_rate=0.1, epochs=500)
        preds = predict(X, W)
        acc = np.mean(preds == y)
        assert acc > 0.8
        return True

    raise ValueError(f"Unknown test: {test_name}")
