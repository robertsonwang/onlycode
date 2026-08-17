import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_synthetic_data(n_samples=200, n_features=3, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.uniform(-3, 3, size=n_features)
    true_b = rng.uniform(-2, 2)
    y = X @ true_w + true_b + noise * rng.normal(size=n_samples)
    return X, y, true_w, true_b


def closed_form_ols(X, y):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return theta[:-1], theta[-1]


# ---------------------------------------------------------------------------
# Stage 1: Loss function
# ---------------------------------------------------------------------------
def mse_loss(y_pred, y_true):
    """
    Mean squared error.

    Args:
        y_pred: np.ndarray, shape (n_samples,)
        y_true: np.ndarray, shape (n_samples,)

    Returns:
        float: mean squared error
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Gradients
# ---------------------------------------------------------------------------
def compute_gradients(X, y, weights, bias):
    """
    Gradients of MSE loss w.r.t. weights and bias for linear regression
    y_pred = X @ weights + bias.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        weights: np.ndarray, shape (n_features,)
        bias: float

    Returns:
        (grad_weights, grad_bias): grad_weights has shape (n_features,),
        grad_bias is a float.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Training loop (batch gradient descent)
# ---------------------------------------------------------------------------
def fit_linear_regression(X, y, lr=0.1, n_iters=1000):
    """
    Fit weights and bias via batch gradient descent.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        lr: learning rate
        n_iters: number of gradient descent steps

    Returns:
        (weights, bias): weights has shape (n_features,), bias is a float.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: L2 regularization (ridge regression)
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


def fit_ridge_regression(X, y, lr=0.1, n_iters=1000, alpha=0.1):
    """
    Fit weights and bias via batch gradient descent with L2 regularization.
    Should reuse compute_gradients_l2.

    Returns:
        (weights, bias)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Mini-batch SGD
# ---------------------------------------------------------------------------
def fit_sgd(X, y, lr=0.1, n_epochs=50, batch_size=16, alpha=0.0, seed=0):
    """
    Fit weights and bias via mini-batch stochastic gradient descent.
    Shuffle the data at the start of every epoch. Should reuse
    compute_gradients_l2 (alpha=0.0 gives plain SGD).

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        lr: learning rate
        n_epochs: number of passes through the dataset
        batch_size: number of samples per mini-batch
        alpha: L2 regularization strength
        seed: random seed for reproducibility

    Returns:
        (weights, bias)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
def r_squared(y_pred, y_true):
    """
    Coefficient of determination (R^2).

    Args:
        y_pred: np.ndarray, shape (n_samples,)
        y_true: np.ndarray, shape (n_samples,)

    Returns:
        float: R^2 score (1.0 = perfect fit, 0.0 = predicting the mean)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1 ===
    if test_name == "mse_basic":
        assert np.isclose(mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])), 0.0)
        assert np.isclose(mse_loss(np.array([0.0, 0.0]), np.array([1.0, -1.0])), 1.0)
        assert np.isclose(mse_loss(np.array([4.0, 0.0]), np.array([1.0, 0.0])), 4.5)
        return True

    # === Stage 2 ===
    if test_name == "gradients_zero_at_optimum":
        X, y, _, _ = make_synthetic_data(noise=0.0)
        w_star, b_star = closed_form_ols(X, y)
        grad_w, grad_b = compute_gradients(X, y, w_star, b_star)
        assert np.allclose(grad_w, 0.0, atol=1e-6)
        assert np.isclose(grad_b, 0.0, atol=1e-6)
        return True

    if test_name == "gradients_numeric_check":
        X, y, _, _ = make_synthetic_data(n_samples=30, n_features=2, seed=1)
        weights = np.array([0.3, -0.7])
        bias = 0.2
        grad_w, grad_b = compute_gradients(X, y, weights, bias)
        eps = 1e-5

        def loss_at(w, b):
            return mse_loss(X @ w + b, y)

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

    # === Stage 3 ===
    if test_name == "fit_recovers_true_params":
        X, y, true_w, true_b = make_synthetic_data(n_samples=500, noise=0.01)
        weights, bias = fit_linear_regression(X, y, lr=0.1, n_iters=2000)
        assert np.allclose(weights, true_w, atol=0.1)
        assert np.isclose(bias, true_b, atol=0.1)
        return True

    if test_name == "fit_matches_closed_form":
        X, y, _, _ = make_synthetic_data(n_samples=300, noise=0.05, seed=2)
        w_ols, b_ols = closed_form_ols(X, y)
        weights, bias = fit_linear_regression(X, y, lr=0.1, n_iters=3000)
        assert np.allclose(weights, w_ols, atol=0.05)
        assert np.isclose(bias, b_ols, atol=0.05)
        return True

    if test_name == "loss_decreases_monotonically_ish":
        X, y, _, _ = make_synthetic_data(seed=3)
        init_loss = mse_loss(np.zeros_like(y), y)
        weights, bias = fit_linear_regression(X, y, lr=0.1, n_iters=500)
        final_loss = mse_loss(X @ weights + bias, y)
        assert final_loss < init_loss * 0.05
        return True

    # === Stage 4 ===
    if test_name == "l2_gradient_reduces_to_plain_at_alpha_zero":
        X, y, _, _ = make_synthetic_data(n_samples=30, seed=4)
        weights = np.array([0.5, -0.2, 1.1])
        bias = 0.1
        gw_plain, gb_plain = compute_gradients(X, y, weights, bias)
        gw_l2, gb_l2 = compute_gradients_l2(X, y, weights, bias, alpha=0.0)
        assert np.allclose(gw_plain, gw_l2)
        assert np.isclose(gb_plain, gb_l2)
        return True

    if test_name == "ridge_shrinks_weights":
        X, y, _, _ = make_synthetic_data(n_samples=200, n_features=5, noise=0.5, seed=5)
        w_plain, _ = fit_linear_regression(X, y, lr=0.05, n_iters=2000)
        w_ridge, _ = fit_ridge_regression(X, y, lr=0.05, n_iters=2000, alpha=5.0)
        assert np.linalg.norm(w_ridge) < np.linalg.norm(w_plain)
        return True

    # === Stage 5 ===
    if test_name == "sgd_converges_reasonably":
        X, y, true_w, true_b = make_synthetic_data(n_samples=500, noise=0.05, seed=6)
        weights, bias = fit_sgd(X, y, lr=0.05, n_epochs=100, batch_size=32)
        assert np.allclose(weights, true_w, atol=0.3)
        assert np.isclose(bias, true_b, atol=0.3)
        return True

    if test_name == "sgd_reproducible_with_seed":
        X, y, _, _ = make_synthetic_data(n_samples=200, seed=7)
        w1, b1 = fit_sgd(X, y, n_epochs=20, seed=42)
        w2, b2 = fit_sgd(X, y, n_epochs=20, seed=42)
        assert np.allclose(w1, w2)
        assert np.isclose(b1, b2)
        return True

    # === Stage 6 ===
    if test_name == "r_squared_perfect_fit":
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(r_squared(y_true, y_true), 1.0)
        return True

    if test_name == "r_squared_mean_baseline":
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert np.isclose(r_squared(y_pred, y_true), 0.0)
        return True

    if test_name == "r_squared_on_trained_model":
        X, y, _, _ = make_synthetic_data(n_samples=300, noise=0.05, seed=8)
        weights, bias = fit_linear_regression(X, y, lr=0.1, n_iters=2000)
        y_pred = X @ weights + bias
        r2 = r_squared(y_pred, y)
        assert r2 > 0.95
        return True

    raise ValueError(f"Unknown test: {test_name}")
