from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_boost_data(n: int, seed: int, noise_frac: float = 0.0):
    """
    Create a synthetic binary classification dataset for boosting practice.

    The label is positive only when both conditions are met:
    - x_0 > 5
    - x_1 < 4
    Otherwise the label is negative.

    Args:
        n: Number of samples.
        seed: Random seed for reproducibility.
        noise_frac: Fraction of labels to flip uniformly at random.

    Returns:
        A tuple (X, y):
        - X: Feature matrix of shape (n, 2).
        - y: Label vector of shape (n,) with values in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 2))
    y01 = ((X[:, 0] > 5) & (X[:, 1] < 4)).astype(int)
    y = np.where(y01 == 1, 1, -1)
    if noise_frac > 0:
        flip = rng.random(n) < noise_frac
        y = np.where(flip, -y, y)
    return X, y


# ---------------------------------------------------------------------------
# Stage 1: Weighted error
# ---------------------------------------------------------------------------
def weighted_error(y_true, y_pred, w) -> float:
    """
    Weighted classification error.

    Args:
        y_true: np.ndarray of shape (n,), labels in {-1, +1}
        y_pred: np.ndarray of shape (n,), labels in {-1, +1}
        w: np.ndarray of shape (n,), nonnegative weights summing to 1

    Returns:
        float in [0, 1]

    Notes:
        This is the objective minimized when selecting the next weak learner.
        Misclassified points contribute their current sample weight; correctly
        classified points contribute zero.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Stump prediction
# ---------------------------------------------------------------------------
def stump_predict(
    X,
    feature: int,
    threshold: float,
    polarity: int,
):
    """
    Predict with a single decision stump.

    Rule:
    - Base rule predicts +1 if X[:, feature] <= threshold else -1.
    - If polarity == -1, flip all predictions.

    Returns:
        np.ndarray shape (n,), values in {-1, +1}

    Args:
        X: Feature matrix of shape (n, d).
        feature: Column index used for the split.
        threshold: Split threshold.
        polarity: Either +1 (base rule) or -1 (flipped rule).
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Best stump under current weights
# ---------------------------------------------------------------------------
def best_stump(X, y, w) -> tuple[dict, float] | None:
    """
    Search all features, midpoint thresholds, and both polarities.

    Midpoint thresholds are generated from sorted unique feature values.
    For each candidate threshold and polarity, compute weighted
    classification error and retain the minimum.

    Args:
        X: Feature matrix of shape (n, d).
        y: Labels of shape (n,), values in {-1, +1}.
        w: Sample weights of shape (n,), nonnegative and typically summing to 1.

    Returns:
        (stump_dict, best_error), where stump_dict has keys:
        - feature (int)
        - threshold (float)
        - polarity (int)

    Return None if no valid split threshold exists for any feature.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Alpha weight
# ---------------------------------------------------------------------------
def compute_alpha(err: float, eps: float = 1e-12) -> float:
    """
    Compute learner weight alpha = 0.5 * log((1 - err)/err).
    Clip err into [eps, 1 - eps] to avoid inf.

    Args:
        err: Weighted error rate for a weak learner.
        eps: Numerical stability floor/ceiling value.

    Returns:
        Signed contribution of this weak learner in the additive ensemble.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Reweighting
# ---------------------------------------------------------------------------
def update_weights(w, y, y_pred, alpha):
    """
    Update and normalize sample weights:
        w_i gets updated to w_i * exp(-alpha * y_i * y_pred_i)
    You must normalize the resulting weights to make sure they sum to 1.

    Intuition:
    - Correct predictions (y_i * y_pred_i = +1) get down-weighted.
    - Mistakes (y_i * y_pred_i = -1) get up-weighted.
    - Final normalization keeps weights as a probability distribution.

    Args:
        w: Current weights, shape (n,), summing to 1.
        y: True labels, shape (n,), values in {-1, +1}.
        y_pred: Weak learner predictions, shape (n,), values in {-1, +1}.
        alpha: Weak learner coefficient.

    Returns:
        np.ndarray shape (n,), sums to 1
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Fit AdaBoost
# ---------------------------------------------------------------------------
def fit_adaboost(X, y, n_estimators: int = 20):
    """
    Train AdaBoost with decision stumps.

    At each round:
    1. Find the stump minimizing weighted error.
    2. Convert error to alpha.
    3. Reweight samples to emphasize hard examples.

    Args:
        X: Feature matrix of shape (n, d).
        y: Labels of shape (n,), values in {-1, +1}.
        n_estimators: Number of boosting rounds.

    Returns model dict with keys:
      - "stumps": list of stump dicts
      - "alphas": list of float
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 7: Predict and evaluate
# ---------------------------------------------------------------------------
def predict_scores(X, model):
    """
    Return additive margin scores sum_t alpha_t * h_t(x).

    Args:
        X: Feature matrix of shape (n, d).
        model: Trained model containing matched stumps and alpha weights.
               model["stumps"] is a list of stump dicts (each with keys
               feature, threshold, polarity); model["alphas"] is a list of
               floats, same length, where alphas[t] is the vote weight for
               stumps[t].

    Returns:
        Score vector of shape (n,). Positive scores favor class +1.

    Hints:
        - Start with an accumulator of zeros, shape (n,) — one running
          score per sample, not per stump.
        - You need exactly one loop: over the T (stump, alpha) pairs
          (zip(model["stumps"], model["alphas"]) is the natural way to walk
          them together). You do NOT need a second loop over samples —
          stump_predict already returns predictions for every row of X at
          once, so each iteration is `accumulator += alpha_t * stump_predict(X, **stump_t)`.
        - stump_predict expects feature/threshold/polarity as separate
          arguments, not a dict — either unpack the stump dict's values
          positionally or use **stump_t if its keys exactly match
          stump_predict's parameter names.
        - The output is a real-valued margin, not a label — don't clip,
          round, or sign() anything here. That happens one level up, in
          predict_adaboost.
    """
    # TODO: implement
    raise NotImplementedError


def predict_adaboost(X, model):
    """
    Return class predictions in {-1, +1} using sign of score.

    Args:
        X: Feature matrix of shape (n, d).
        model: Trained AdaBoost model.

    Returns:
        Predicted labels of shape (n,), values in {-1, +1}.

    Hints:
        - This is a thin wrapper around predict_scores — get the margin
          first, then convert margin -> label.
        - Watch out for np.sign specifically: np.sign(0) returns 0, not
          +1 or -1 — so if a score ever lands exactly on 0, np.sign alone
          would produce a value outside {-1, +1} and violate this
          function's contract. Pick an explicit tie-breaking rule instead
          of relying on np.sign directly, e.g. np.where(scores >= 0, 1, -1)
          treats a zero margin as class +1.
        - This should end up as one or two lines — no loop needed here,
          the loop already happened inside predict_scores.
    """
    # TODO: implement
    raise NotImplementedError


def accuracy(y_true, y_pred) -> float:
    """
    Classification accuracy.

    Args:
        y_true: Ground-truth labels, shape (n,).
        y_pred: Predicted labels, shape (n,).

    Returns:
        Fraction of matching labels in [0, 1].
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    test_name = input_data["test"]

    # === Stage 1 ===
    if test_name == "weighted_error_zero":
        y = np.array([1, -1, 1, -1])
        p = np.array([1, -1, 1, -1])
        w = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.isclose(weighted_error(y, p, w), 0.0)
        return True

    if test_name == "weighted_error_known":
        y = np.array([1, -1, 1, -1])
        p = np.array([-1, -1, 1, 1])
        w = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.isclose(weighted_error(y, p, w), 0.5)
        return True

    # === Stage 2 ===
    if test_name == "stump_predict_polarity_plus":
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        pred = stump_predict(X, feature=0, threshold=2.5, polarity=1)
        assert np.array_equal(pred, np.array([1, 1, -1, -1]))
        return True

    if test_name == "stump_predict_polarity_minus":
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        pred = stump_predict(X, feature=0, threshold=2.5, polarity=-1)
        assert np.array_equal(pred, np.array([-1, -1, 1, 1]))
        return True

    # === Stage 3 ===
    if test_name == "best_stump_obvious":
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1, 1, -1, -1])
        w = np.full(4, 0.25)
        stump, err = best_stump(X, y, w)
        assert stump is not None
        assert err is not None
        stump, err = result
        pred = stump_predict(X, stump["feature"], stump["threshold"], stump["polarity"])
        assert np.isclose(weighted_error(y, pred, w), 0.0)
        assert np.isclose(err, 0.0)
        return True

    if test_name == "best_stump_none_when_no_threshold":
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1, -1, 1])
        w = np.full(3, 1 / 3)
        best_tree, best_error = best_stump(X, y, w)
        assert best_tree is None
        assert best_error is None
        return True

    # === Stage 4 ===
    if test_name == "compute_alpha_known":
        a = compute_alpha(0.25)
        assert np.isclose(a, 0.5 * np.log(3.0))
        return True

    if test_name == "compute_alpha_clips_extremes":
        a1 = compute_alpha(0.0)
        a2 = compute_alpha(1.0)
        assert np.isfinite(a1)
        assert np.isfinite(a2)
        assert a1 > 0
        assert a2 < 0
        return True

    # === Stage 5 ===
    if test_name == "update_weights_emphasizes_mistakes":
        y = np.array([1, 1, -1, -1])
        p = np.array([1, -1, -1, -1])
        w = np.full(4, 0.25)
        alpha = 0.4
        new_w = update_weights(w, y, p, alpha)
        assert np.isclose(new_w.sum(), 1.0)
        assert new_w[1] > new_w[0]
        return True

    # === Stage 6 ===
    if test_name == "fit_adaboost_model_shapes":
        X, y = make_boost_data(80, seed=10, noise_frac=0.0)
        model = fit_adaboost(X, y, n_estimators=7)
        assert set(model.keys()) == {"stumps", "alphas"}
        assert len(model["stumps"]) == len(model["alphas"]) == 7
        return True

    if test_name == "fit_adaboost_improves_training_accuracy":
        X, y = make_boost_data(150, seed=11, noise_frac=0.0)

        # Weak learner only (1 stump)
        m1 = fit_adaboost(X, y, n_estimators=1)
        acc1 = accuracy(y, predict_adaboost(X, m1))

        # Boosted ensemble
        mT = fit_adaboost(X, y, n_estimators=15)
        accT = accuracy(y, predict_adaboost(X, mT))

        assert accT >= acc1
        assert accT > 0.9
        return True

    # === Stage 7 ===
    if test_name == "predict_scores_and_labels_shapes":
        X, y = make_boost_data(60, seed=12, noise_frac=0.0)
        model = fit_adaboost(X, y, n_estimators=6)
        scores = predict_scores(X, model)
        preds = predict_adaboost(X, model)
        assert scores.shape == (60,)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({-1, 1})
        return True

    if test_name == "generalization_reasonable_on_clean_data":
        X_train, y_train = make_boost_data(250, seed=20, noise_frac=0.1)
        X_test, y_test = make_boost_data(250, seed=21, noise_frac=0.0)
        model = fit_adaboost(X_train, y_train, n_estimators=20)
        test_acc = accuracy(y_test, predict_adaboost(X_test, model))
        assert test_acc > 0.85
        return True

    raise ValueError(f"Unknown test: {test_name}")
