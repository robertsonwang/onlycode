import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_separable_data(n_samples=300, n_features=2, seed=0, class_sep=2.0):
    rng = np.random.default_rng(seed)
    true_w = rng.uniform(-2, 2, size=n_features)
    true_b = rng.uniform(-1, 1)
    X = rng.normal(size=(n_samples, n_features)) * class_sep
    logits = X @ true_w + true_b
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probs).astype(float)
    return X, y, true_w, true_b


# ---------------------------------------------------------------------------
# Stage 1: Sigmoid
# ---------------------------------------------------------------------------
def sigmoid(z):
    """
    Logistic sigmoid, applied elementwise.

    Args:
        z: np.ndarray, any shape

    Returns:
        np.ndarray, same shape as z, values in (0, 1)

    Note: naive exp(-z) can overflow for very negative z. Consider a
    numerically stable formulation.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Binary cross-entropy loss
# ---------------------------------------------------------------------------
def bce_loss(y_pred, y_true):
    """
    Mean binary cross-entropy.

    Args:
        y_pred: np.ndarray, shape (n_samples,), predicted probabilities
                in (0, 1)
        y_true: np.ndarray, shape (n_samples,), labels in {0, 1}

    Returns:
        float

    Note: y_pred at exactly 0 or 1 will produce log(0). Consider clipping
    y_pred to a safe range, e.g. [eps, 1 - eps].
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Gradients
# ---------------------------------------------------------------------------
def compute_gradients(X, y, weights, bias):
    """
    Gradients of mean BCE loss w.r.t. weights and bias, where
    y_pred = sigmoid(X @ weights + bias).

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,), labels in {0, 1}
        weights: np.ndarray, shape (n_features,)
        bias: float

    Returns:
        (grad_weights, grad_bias)

    Hint: the gradient of mean BCE loss w.r.t. the pre-sigmoid logits has a
    surprisingly clean closed form. Derive it rather than chaining
    d(loss)/d(sigmoid) * d(sigmoid)/d(logit) numerically.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Training loop (batch gradient descent)
# ---------------------------------------------------------------------------
def fit_logistic_regression(X, y, lr=0.1, n_iters=1000):
    """
    Fit weights and bias via batch gradient descent.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,), labels in {0, 1}
        lr: learning rate
        n_iters: number of gradient descent steps

    Returns:
        (weights, bias)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: L2 regularization
# ---------------------------------------------------------------------------
def compute_gradients_l2(X, y, weights, bias, alpha):
    """
    Same as compute_gradients, but adds L2 penalty alpha * ||weights||^2
    to the loss (bias is NOT regularized).

    Args:
        alpha: float, regularization strength (alpha=0 reduces to
               compute_gradients)

    Returns:
        (grad_weights, grad_bias)
    """
    # TODO: implement
    raise NotImplementedError


def fit_ridge_logistic_regression(X, y, lr=0.1, n_iters=1000, alpha=0.1):
    """
    Fit weights and bias via batch gradient descent with L2 regularization.
    Should reuse compute_gradients_l2.

    Returns:
        (weights, bias)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
def predict_proba(X, weights, bias):
    """
    Predicted probabilities for the positive class.

    Returns:
        np.ndarray, shape (n_samples,)
    """
    # TODO: implement
    raise NotImplementedError


def predict(X, weights, bias, threshold=0.5):
    """
    Predicted hard labels (0 or 1).

    Returns:
        np.ndarray, shape (n_samples,), dtype int
    """
    # TODO: implement
    raise NotImplementedError


def accuracy(y_pred, y_true):
    """Fraction of predictions that match the true label."""
    # TODO: implement
    raise NotImplementedError


def precision_recall_f1(y_pred, y_true):
    """
    Precision, recall, and F1 for the positive class (label == 1).

    Returns:
        (precision, recall, f1): all floats. If a denominator would be
        zero (e.g. no predicted positives), define the corresponding
        metric as 0.0.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1: Sigmoid ===
    if test_name == "sigmoid_basic_values":
        assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
        assert sigmoid(np.array([100.0]))[0] > 0.999
        assert sigmoid(np.array([-100.0]))[0] < 0.001
        return True

    if test_name == "sigmoid_numerically_stable":
        z = np.array([-1000.0, 1000.0])
        with np.errstate(over="raise", invalid="raise"):
            out = sigmoid(z)
        assert np.all(np.isfinite(out))
        assert np.isclose(out[0], 0.0, atol=1e-6)
        assert np.isclose(out[1], 1.0, atol=1e-6)
        return True

    if test_name == "sigmoid_symmetry":
        z = np.array([0.3, -1.2, 2.5])
        assert np.allclose(sigmoid(z) + sigmoid(-z), 1.0)
        return True

    # === Stage 2: BCE loss ===
    if test_name == "bce_perfect_predictions":
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0 - 1e-9, 1e-9, 1.0 - 1e-9, 1e-9])
        assert bce_loss(y_pred, y_true) < 1e-6
        return True

    if test_name == "bce_known_value":
        y_true = np.array([1.0])
        y_pred = np.array([0.5])
        assert np.isclose(bce_loss(y_pred, y_true), np.log(2))
        return True

    if test_name == "bce_handles_extreme_probs_without_nan":
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])
        loss = bce_loss(y_pred, y_true)
        assert np.isfinite(loss)
        assert loss > 5
        return True

    # === Stage 3: Gradients ===
    if test_name == "gradients_numeric_check":
        X, y, _, _ = make_separable_data(n_samples=40, n_features=2, seed=42)
        weights = np.array([0.3, -0.7])
        bias = 0.2
        grad_w, grad_b = compute_gradients(X, y, weights, bias)
        eps = 1e-5

        def loss_at(w, b):
            z = X @ w + b
            y_pred = sigmoid(z)
            return bce_loss(y_pred, y)

        numeric_grad_w = np.zeros_like(weights)
        for i in range(len(weights)):
            w_plus, w_minus = weights.copy(), weights.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            numeric_grad_w[i] = (loss_at(w_plus, bias) - loss_at(w_minus, bias)) / (2 * eps)
        numeric_grad_b = (loss_at(weights, bias + eps) - loss_at(weights, bias - eps)) / (2 * eps)
        assert np.allclose(grad_w, numeric_grad_w, atol=1e-3)
        assert np.isclose(grad_b, numeric_grad_b, atol=1e-3)
        return True

    if test_name == "gradients_zero_when_perfectly_confident_and_correct":
        X = np.array([[10.0], [-10.0]])
        y = np.array([1.0, 0.0])
        weights = np.array([1.0])
        bias = 0.0
        grad_w, grad_b = compute_gradients(X, y, weights, bias)
        assert np.allclose(grad_w, 0.0, atol=1e-3)
        assert np.isclose(grad_b, 0.0, atol=1e-3)
        return True

    # === Stage 4: Training loop ===
    if test_name == "fit_separates_linearly_separable_data":
        rng = np.random.default_rng(2)
        n = 200
        X_pos = rng.normal(loc=[3, 3], scale=0.5, size=(n // 2, 2))
        X_neg = rng.normal(loc=[-3, -3], scale=0.5, size=(n // 2, 2))
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights, bias = fit_logistic_regression(X, y, lr=0.1, n_iters=2000)
        logits = X @ weights + bias
        preds = (sigmoid(logits) >= 0.5).astype(float)
        acc = np.mean(preds == y)
        assert acc > 0.98
        return True

    if test_name == "loss_decreases_with_training":
        X, y, _, _ = make_separable_data(n_samples=200, seed=3)
        init_pred = sigmoid(np.zeros(len(y)))
        init_loss = bce_loss(init_pred, y)
        weights, bias = fit_logistic_regression(X, y, lr=0.1, n_iters=500)
        final_pred = sigmoid(X @ weights + bias)
        final_loss = bce_loss(final_pred, y)
        assert final_loss < init_loss
        return True

    # === Stage 5: L2 regularization ===
    if test_name == "l2_gradient_reduces_to_plain_at_alpha_zero":
        X, y, _, _ = make_separable_data(n_samples=30, seed=4)
        weights = np.array([0.5, -0.2])
        bias = 0.1
        gw_plain, gb_plain = compute_gradients(X, y, weights, bias)
        gw_l2, gb_l2 = compute_gradients_l2(X, y, weights, bias, alpha=0.0)
        assert np.allclose(gw_plain, gw_l2)
        assert np.isclose(gb_plain, gb_l2)
        return True

    if test_name == "ridge_shrinks_weights":
        rng = np.random.default_rng(5)
        n = 200
        X_pos = rng.normal(loc=[3, 3, 3], scale=1.0, size=(n // 2, 3))
        X_neg = rng.normal(loc=[-3, -3, -3], scale=1.0, size=(n // 2, 3))
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        w_plain, _ = fit_logistic_regression(X, y, lr=0.05, n_iters=2000)
        w_ridge, _ = fit_ridge_logistic_regression(X, y, lr=0.05, n_iters=2000, alpha=5.0)
        assert np.linalg.norm(w_ridge) < np.linalg.norm(w_plain)
        return True

    # === Stage 6: Evaluation ===
    if test_name == "predict_proba_and_predict_consistency":
        X = np.array([[1.0], [-1.0], [0.0]])
        weights = np.array([2.0])
        bias = 0.0
        probs = predict_proba(X, weights, bias)
        preds = predict(X, weights, bias)
        assert np.allclose(probs, sigmoid(X @ weights + bias))
        assert np.array_equal(preds, (probs >= 0.5).astype(int))
        return True

    if test_name == "accuracy_basic":
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        assert np.isclose(accuracy(y_pred, y_true), 0.8)
        return True

    if test_name == "precision_recall_f1_known_values":
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        p, r, f1 = precision_recall_f1(y_pred, y_true)
        assert np.isclose(p, 2 / 3, atol=1e-6)
        assert np.isclose(r, 2 / 3, atol=1e-6)
        assert np.isclose(f1, 2 / 3, atol=1e-6)
        return True

    if test_name == "precision_recall_f1_no_predicted_positives":
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 0, 0])
        p, r, f1 = precision_recall_f1(y_pred, y_true)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
        return True

    if test_name == "full_pipeline_high_f1_on_separable_data":
        rng = np.random.default_rng(6)
        n = 200
        X_pos = rng.normal(loc=[3, 3], scale=0.5, size=(n // 2, 2))
        X_neg = rng.normal(loc=[-3, -3], scale=0.5, size=(n // 2, 2))
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights, bias = fit_logistic_regression(X, y, lr=0.1, n_iters=2000)
        preds = predict(X, weights, bias)
        _, _, f1 = precision_recall_f1(preds, y)
        assert f1 > 0.95
        return True

    raise ValueError(f"Unknown test: {test_name}")
