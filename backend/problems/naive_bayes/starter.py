import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array(centers, dtype=float)
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=spread, size=(n_per_cluster, len(c)))
        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, i))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Stage 1: Class priors
# ---------------------------------------------------------------------------
def compute_priors(y, n_classes):
    """
    Compute prior probability of each class = fraction of samples.

    Args:
        y: np.ndarray, shape (n_samples,), integer labels in [0, n_classes)
        n_classes: int

    Returns:
        np.ndarray, shape (n_classes,), priors[c] = P(y=c)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Class statistics
# ---------------------------------------------------------------------------
def compute_class_stats(X, y, n_classes):
    """
    Compute per-class mean and variance for each feature.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        n_classes: int

    Returns:
        (means, variances):
            means: np.ndarray, shape (n_classes, n_features)
            variances: np.ndarray, shape (n_classes, n_features)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Log-likelihood
# ---------------------------------------------------------------------------
def gaussian_log_likelihood(x, mean, var):
    """
    Log of Gaussian PDF: log N(x | mean, var) for each feature.

    Args:
        x: np.ndarray, shape (n_features,)
        mean: np.ndarray, shape (n_features,)
        var: np.ndarray, shape (n_features,)

    Returns:
        float: sum of per-feature log-likelihoods

    Note: add a small epsilon (e.g. 1e-9) to var for stability.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Log-posterior
# ---------------------------------------------------------------------------
def compute_log_posteriors(X, priors, means, variances):
    """
    Compute unnormalized log-posterior for each sample and each class.

    log P(y=c | x) ~ log P(y=c) + sum_j log P(x_j | y=c)

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        priors: np.ndarray, shape (n_classes,)
        means: np.ndarray, shape (n_classes, n_features)
        variances: np.ndarray, shape (n_classes, n_features)

    Returns:
        np.ndarray, shape (n_samples, n_classes)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Fit and predict
# ---------------------------------------------------------------------------
def fit(X, y, n_classes):
    """
    Fit a Gaussian Naive Bayes model.

    Returns:
        dict with keys "priors", "means", "variances", "n_classes"
    """
    # TODO: implement
    raise NotImplementedError


def predict(X, model):
    """
    Predict class labels by argmax of log-posteriors.

    Returns:
        np.ndarray, shape (n_samples,), dtype int
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
def accuracy(y_pred, y_true):
    """Fraction of predictions that match the true label."""
    # TODO: implement
    raise NotImplementedError


def fit_and_evaluate(X_train, y_train, X_test, y_test, n_classes):
    """
    Fit on training data, predict on test data, return test accuracy.

    Returns:
        float
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1 ===
    if test_name == "priors_uniform":
        y = np.array([0, 1, 0, 1])
        priors = compute_priors(y, 2)
        assert np.allclose(priors, [0.5, 0.5])
        return True

    if test_name == "priors_imbalanced":
        y = np.array([0, 0, 0, 1])
        priors = compute_priors(y, 2)
        assert np.isclose(priors[0], 0.75)
        assert np.isclose(priors[1], 0.25)
        return True

    if test_name == "priors_sum_to_one":
        y = np.array([0, 1, 2, 2, 1, 0, 2, 2])
        priors = compute_priors(y, 3)
        assert np.isclose(priors.sum(), 1.0)
        return True

    # === Stage 2 ===
    if test_name == "class_stats_shape":
        X = np.random.default_rng(0).normal(size=(20, 3))
        y = np.array([0]*10 + [1]*10)
        means, variances = compute_class_stats(X, y, 2)
        assert means.shape == (2, 3)
        assert variances.shape == (2, 3)
        return True

    if test_name == "class_stats_hand_computed":
        X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [12.0, 22.0]])
        y = np.array([0, 0, 1, 1])
        means, variances = compute_class_stats(X, y, 2)
        assert np.allclose(means[0], [2.0, 3.0])
        assert np.allclose(means[1], [11.0, 21.0])
        assert np.allclose(variances[0], [1.0, 1.0])
        assert np.allclose(variances[1], [1.0, 1.0])
        return True

    # === Stage 3 ===
    if test_name == "gaussian_log_likelihood_known_value":
        # log N(0 | 0, 1) = -0.5 * log(2*pi)
        ll = gaussian_log_likelihood(np.array([0.0]), np.array([0.0]), np.array([1.0]))
        expected = -0.5 * np.log(2 * np.pi)
        assert np.isclose(ll, expected, atol=1e-5)
        return True

    if test_name == "gaussian_log_likelihood_peak_at_mean":
        mean = np.array([3.0, 5.0])
        var = np.array([1.0, 2.0])
        ll_at_mean = gaussian_log_likelihood(mean, mean, var)
        ll_away = gaussian_log_likelihood(mean + 5.0, mean, var)
        assert ll_at_mean > ll_away
        return True

    if test_name == "gaussian_log_likelihood_finite_small_var":
        ll = gaussian_log_likelihood(np.array([1.0]), np.array([1.0]), np.array([0.0]))
        assert np.isfinite(ll)
        return True

    # === Stage 4 ===
    if test_name == "log_posteriors_shape":
        X, y = make_blobs(centers=[(0,0), (5,5)], n_per_cluster=20, seed=0)
        priors = compute_priors(y, 2)
        means, variances = compute_class_stats(X, y, 2)
        lp = compute_log_posteriors(X, priors, means, variances)
        assert lp.shape == (40, 2)
        return True

    if test_name == "log_posteriors_correct_class_highest":
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        y_train = np.array([0, 1])
        X_train = np.array([[0.0, 0.0], [0.1, -0.1], [10.0, 10.0], [9.9, 10.1]])
        y_full = np.array([0, 0, 1, 1])
        priors = compute_priors(y_full, 2)
        means, variances = compute_class_stats(X_train, y_full, 2)
        lp = compute_log_posteriors(X, priors, means, variances)
        assert np.argmax(lp[0]) == 0
        assert np.argmax(lp[1]) == 1
        return True

    # === Stage 5 ===
    if test_name == "fit_returns_correct_keys":
        X, y = make_blobs(centers=[(0,0), (5,5)], n_per_cluster=20, seed=0)
        model = fit(X, y, 2)
        assert "priors" in model
        assert "means" in model
        assert "variances" in model
        return True

    if test_name == "predict_shape_and_range":
        X, y = make_blobs(centers=[(0,0), (5,5)], n_per_cluster=30, seed=1)
        model = fit(X, y, 2)
        preds = predict(X, model)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({0, 1})
        return True

    if test_name == "predict_separable_blobs":
        X, y = make_blobs(centers=[(0,0), (10,10)], n_per_cluster=50, spread=0.5, seed=2)
        model = fit(X, y, 2)
        preds = predict(X, model)
        assert np.mean(preds == y) > 0.98
        return True

    if test_name == "predict_three_class":
        X, y = make_blobs(centers=[(0,0), (10,0), (5,10)], n_per_cluster=40, spread=0.5, seed=3)
        model = fit(X, y, 3)
        preds = predict(X, model)
        assert np.mean(preds == y) > 0.95
        return True

    # === Stage 6 ===
    if test_name == "accuracy_basic":
        y_pred = np.array([0, 1, 1, 0, 1])
        y_true = np.array([0, 1, 0, 0, 1])
        assert np.isclose(accuracy(y_pred, y_true), 0.8)
        return True

    if test_name == "accuracy_perfect":
        y = np.array([0, 1, 2, 1])
        assert np.isclose(accuracy(y, y), 1.0)
        return True

    if test_name == "fit_and_evaluate_separable":
        X_train, y_train = make_blobs(centers=[(0,0), (10,10)], n_per_cluster=50, spread=0.5, seed=10)
        X_test, y_test = make_blobs(centers=[(0,0), (10,10)], n_per_cluster=30, spread=0.5, seed=11)
        acc = fit_and_evaluate(X_train, y_train, X_test, y_test, 2)
        assert acc > 0.95
        return True

    if test_name == "fit_and_evaluate_three_class":
        X_train, y_train = make_blobs(centers=[(0,0), (10,0), (5,10)], n_per_cluster=50, spread=0.5, seed=20)
        X_test, y_test = make_blobs(centers=[(0,0), (10,0), (5,10)], n_per_cluster=30, spread=0.5, seed=21)
        acc = fit_and_evaluate(X_train, y_train, X_test, y_test, 3)
        assert acc > 0.90
        return True

    raise ValueError(f"Unknown test: {test_name}")
