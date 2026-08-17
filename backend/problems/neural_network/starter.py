import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def xor_data():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    return X, y


# ---------------------------------------------------------------------------
# Stage 1: Activations
# ---------------------------------------------------------------------------
def relu(z):
    """Elementwise ReLU. Returns array same shape as z."""
    # TODO: implement
    raise NotImplementedError


def relu_derivative(z):
    """
    Elementwise derivative of ReLU w.r.t. its input z.
    (Use any convention you like at z == 0; it won't be tested there.)
    """
    # TODO: implement
    raise NotImplementedError


def sigmoid(z):
    """Numerically stable logistic sigmoid, applied elementwise."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Forward pass
# ---------------------------------------------------------------------------
def init_params(n_input, n_hidden, n_output=1, seed=0):
    """
    Initialize weights with small random values and biases with zeros.

    Returns:
        dict with keys "W1", "b1", "W2", "b2".
        W1: (n_input, n_hidden)   b1: (n_hidden,)
        W2: (n_hidden, n_output)  b2: (n_output,)

    Note: initializing weights to all zeros will make both hidden units
    learn identically -- use random initialization for weight matrices.
    """
    # TODO: implement
    raise NotImplementedError


def forward(X, params):
    """
    Run the forward pass.

    Args:
        X: np.ndarray, shape (n_samples, n_input)
        params: dict as returned by init_params

    Returns:
        (y_pred, cache): y_pred has shape (n_samples,) -- the flattened,
        sigmoid-activated output. cache is a dict containing whatever
        intermediate values (X, Z1, A1, Z2, A2, ...) you'll need in the
        backward pass.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Loss
# ---------------------------------------------------------------------------
def bce_loss(y_pred, y_true):
    """
    Mean binary cross-entropy. Same as in the logistic regression
    exercise -- remember to clip y_pred away from exactly 0 or 1.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Backward pass
# ---------------------------------------------------------------------------
def backward(y_true, params, cache):
    """
    Backpropagate the gradient of mean BCE loss through the network.

    Args:
        y_true: np.ndarray, shape (n_samples,)
        params: dict as returned by init_params
        cache: dict as returned by forward

    Returns:
        grads: dict with keys "dW1", "db1", "dW2", "db2", matching the
        shapes of "W1", "b1", "W2", "b2" respectively.

    Hint: dL/dZ2 (pre-sigmoid, output layer) simplifies to
    (A2 - y) / n_samples. From there, propagate backward through W2,
    through the ReLU derivative, and finally through W1.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Training loop
# ---------------------------------------------------------------------------
def train(X, y, n_hidden=4, lr=0.5, n_iters=5000, seed=0):
    """
    Train a 1-hidden-layer network via batch gradient descent.

    Args:
        X: np.ndarray, shape (n_samples, n_input)
        y: np.ndarray, shape (n_samples,), labels in {0, 1}
        n_hidden: number of hidden units
        lr: learning rate
        n_iters: number of gradient descent steps
        seed: passed to init_params for reproducibility

    Returns:
        (params, loss_history): params is the trained dict, loss_history
        is a list/array of the loss at each iteration.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Prediction + evaluation
# ---------------------------------------------------------------------------
def predict(X, params, threshold=0.5):
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


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1: Activations ===
    if test_name == "relu_basic":
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 0.5, 2.0])
        assert np.allclose(relu(z), expected)
        return True

    if test_name == "relu_derivative_basic":
        z = np.array([-2.0, -0.5, 0.5, 2.0])
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        assert np.allclose(relu_derivative(z), expected)
        return True

    if test_name == "sigmoid_basic_and_stable":
        assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
        z = np.array([-1000.0, 1000.0])
        out = sigmoid(z)
        assert np.all(np.isfinite(out))
        assert np.isclose(out[0], 0.0, atol=1e-6)
        assert np.isclose(out[1], 1.0, atol=1e-6)
        return True

    # === Stage 2: Forward pass ===
    if test_name == "init_params_shapes":
        params = init_params(n_input=3, n_hidden=5, n_output=1, seed=0)
        assert params["W1"].shape == (3, 5)
        assert params["b1"].shape == (5,)
        assert params["W2"].shape == (5, 1)
        assert params["b2"].shape == (1,)
        return True

    if test_name == "init_params_not_all_zero_weights":
        params = init_params(n_input=3, n_hidden=5, n_output=1, seed=0)
        assert not np.allclose(params["W1"], 0.0)
        assert not np.allclose(params["W2"], 0.0)
        return True

    if test_name == "forward_output_shape_and_range":
        params = init_params(n_input=2, n_hidden=4, n_output=1, seed=0)
        X = np.random.default_rng(1).normal(size=(10, 2))
        y_pred, cache = forward(X, params)
        assert y_pred.shape == (10,)
        assert np.all(y_pred > 0.0) and np.all(y_pred < 1.0)
        return True

    if test_name == "forward_hand_computed":
        params = {
            "W1": np.array([[2.0]]),
            "b1": np.array([-1.0]),
            "W2": np.array([[3.0]]),
            "b2": np.array([0.5]),
        }
        X = np.array([[1.0]])
        y_pred, cache = forward(X, params)
        expected = 1.0 / (1.0 + np.exp(-3.5))
        assert np.isclose(y_pred[0], expected, atol=1e-6)
        return True

    # === Stage 3: Loss ===
    if test_name == "bce_known_value":
        y_true = np.array([1.0])
        y_pred = np.array([0.5])
        assert np.isclose(bce_loss(y_pred, y_true), np.log(2))
        return True

    if test_name == "bce_finite_at_extremes":
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])
        assert np.isfinite(bce_loss(y_pred, y_true))
        return True

    # === Stage 4: Backward pass ===
    if test_name == "backward_gradient_check":
        rng = np.random.default_rng(2)
        n_input, n_hidden, n_output = 3, 4, 1
        params = init_params(n_input, n_hidden, n_output, seed=2)
        X = rng.normal(size=(6, n_input))
        y = rng.integers(0, 2, size=6).astype(float)

        y_pred, cache = forward(X, params)
        grads = backward(y, params, cache)

        def loss_with_params(p):
            pred, _ = forward(X, p)
            return bce_loss(pred, y)

        eps = 1e-5
        rng_check = np.random.default_rng(3)
        for pname, gname in [("W1", "dW1"), ("b1", "db1"), ("W2", "dW2"), ("b2", "db2")]:
            arr = params[pname]
            grad_arr = grads[gname]
            assert grad_arr.shape == arr.shape, f"{gname} shape mismatch"

            flat_idx = list(np.ndindex(arr.shape))
            sample_size = min(5, len(flat_idx))
            sampled = rng_check.choice(len(flat_idx), size=sample_size, replace=False)

            for k in sampled:
                idx = flat_idx[k]
                orig = arr[idx]

                params_plus = {key: (v.copy() if key == pname else v) for key, v in params.items()}
                params_minus = {key: (v.copy() if key == pname else v) for key, v in params.items()}
                params_plus[pname][idx] = orig + eps
                params_minus[pname][idx] = orig - eps

                numeric_grad = (loss_with_params(params_plus) - loss_with_params(params_minus)) / (2 * eps)
                analytic_grad = grad_arr[idx]

                assert np.isclose(numeric_grad, analytic_grad, atol=1e-3, rtol=1e-2), (
                    f"{gname}{idx}: numeric={numeric_grad:.6f} analytic={analytic_grad:.6f}"
                )
        return True

    # === Stage 5: Training loop ===
    if test_name == "loss_history_decreases":
        X, y = xor_data()
        params, loss_history = train(X, y, n_hidden=4, lr=0.5, n_iters=3000, seed=0)
        assert len(loss_history) == 3000
        assert loss_history[-1] < loss_history[0] * 0.2
        return True

    if test_name == "solves_xor":
        X, y = xor_data()
        params, _ = train(X, y, n_hidden=4, lr=0.5, n_iters=8000, seed=0)
        preds = predict(X, params)
        assert np.array_equal(preds, y.astype(int))
        return True

    # === Stage 6: Prediction + evaluation ===
    if test_name == "predict_returns_ints_in_range":
        X, y = xor_data()
        params, _ = train(X, y, n_hidden=4, lr=0.5, n_iters=2000, seed=0)
        preds = predict(X, params)
        assert preds.dtype.kind in ("i", "u")
        assert set(np.unique(preds)).issubset({0, 1})
        return True

    if test_name == "accuracy_on_solved_xor":
        X, y = xor_data()
        params, _ = train(X, y, n_hidden=4, lr=0.5, n_iters=8000, seed=0)
        preds = predict(X, params)
        assert np.isclose(accuracy(preds, y), 1.0)
        return True

    raise ValueError(f"Unknown test: {test_name}")
