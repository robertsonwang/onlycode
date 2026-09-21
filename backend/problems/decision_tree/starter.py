from collections import Counter
import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_rect_data(n, seed, noise_frac=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 2))
    y = ((X[:, 0] > 5) & (X[:, 1] < 3)).astype(int)
    if noise_frac > 0:
        flip = rng.random(n) < noise_frac
        y = np.where(flip, 1 - y, y)
    return X, y


def tree_depth(node):
    if node["leaf"]:
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


# ---------------------------------------------------------------------------
# Stage 1: Impurity
# ---------------------------------------------------------------------------
def gini_impurity(y):
    """
    Gini impurity of a set of labels: 1 - sum_c(p_c^2).
    Empty y has impurity 0.0.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Splitting
# ---------------------------------------------------------------------------
def split_dataset(X, y, feature, threshold):
    """
    Partition (X, y) into left (x[feature] <= threshold) and right
    (x[feature] > threshold) subsets.

    Returns:
        (X_left, y_left, X_right, y_right)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Scoring a split
# ---------------------------------------------------------------------------
def weighted_impurity(y_left, y_right):
    """
    Sample-size-weighted average Gini impurity of the two sides.

    Returns:
        float
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Finding the best split
# ---------------------------------------------------------------------------
def best_split(X, y):
    """
    Search every feature and midpoint threshold for the split that
    minimizes weighted_impurity.

    Returns:
        (feature, threshold, impurity) or None if no valid split exists.

    Hints:
    - Outer loop: iterate over every feature (column) in X — you're
      searching all n_features candidate features, not just one.

    - For each feature, you need candidate thresholds to test. A common
      approach: sort the unique values of that feature, then use the
      midpoints between consecutive sorted unique values as candidate
      thresholds (rather than testing every raw value directly, or
      testing values that don't actually separate any points).

    - For each (feature, threshold) candidate:
        - Partition y into y_left and y_right based on whether each
          sample's value for that feature is below/above the threshold
          (think boolean masking on X[:, feature] compared to threshold)
        - Watch for invalid splits — what happens if the threshold
          produces an empty y_left or y_right? That's not a usable split
          and should probably be skipped rather than scored.
        - Compute weighted_impurity(y_left, y_right) using the function
          you already have (or are about to write) for weighted Gini.

    - Track the best (lowest) impurity seen so far across all candidates,
      along with which (feature, threshold) produced it — classic
      "running minimum" pattern, initialize your best-so-far impurity to
      something like infinity before the loops start.

    - Return None if no valid split was ever found (e.g. every feature
      has only one unique value, so no threshold can split the data —
      this can happen at a leaf-like node).
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Building the tree
# ---------------------------------------------------------------------------
def majority_class(y):
    """
    Most common label in y. Ties broken by smaller label.
    """
    # TODO: implement
    raise NotImplementedError


def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    """
    Recursively build a decision tree. Stop and create a leaf if:
    - depth >= max_depth
    - len(y) < min_samples_split
    - gini_impurity(y) == 0
    - best_split returns None

    Returns:
        node dict: {"leaf": True, "value": ...} or
        {"leaf": False, "feature": ..., "threshold": ..., "left": ..., "right": ...}

    Hints:
    - Check all four stopping conditions before calling best_split —
      no point searching for a split you won't use.
    - Leaf "value" is typically the majority class in y (most frequent
      label) — think about which numpy/collections call gets that.
    - On a valid split, partition X and y into left/right subsets using
      the same feature/threshold mask idea from best_split, then recurse
      on each half with depth+1.
    - Don't forget to actually increment depth on the recursive calls —
      easy off-by-one/infinite-recursion bug if forgotten.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Prediction
# ---------------------------------------------------------------------------
def predict_one(x, node):
    """
    Traverse the tree for a single sample x.

    Hints:
    - Base case: if node["leaf"] is True, you're done — return
      node["value"] directly.
    - Recursive case: compare x[node["feature"]] against
      node["threshold"] — same comparison direction used when the tree
      was built (<=  goes left, > goes right, matching split_dataset).
    - Recurse into predict_one(x, node["left"]) or
      predict_one(x, node["right"]) depending on that comparison, and
      return its result directly (don't forget the return).
    """
    # TODO: implement
    raise NotImplementedError


def predict(X, node):
    """
    Predict classes for every row of X.

    Hints:
    - This is just predict_one applied to every row of X — think about
      how to loop over rows of a 2D array (X[i] for each i, or iterating
      directly over X).
    - Collect the per-row results into a single array to return — same
      shape/order as y would be (one prediction per sample), so the
      caller can compare directly against true labels.
    - Consider whether a list comprehension + np.array(...) wrapping is
      cleaner than manually appending in a loop — either works, but one
      is more idiomatic numpy style.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 7: Bagging ensemble
# ---------------------------------------------------------------------------
def bootstrap_sample(X, y, seed):
    """Sample len(y) rows from (X, y) with replacement."""
    rng = np.random.default_rng(seed)
    # TODO: implement
    raise NotImplementedError


def fit_bagged_trees(X, y, n_trees=15, max_depth=5, min_samples_split=2, seed=0):
    """
    Fit n_trees independent decision trees, each on a bootstrap resample.
    Tree i uses seed + i.

    Returns:
        list of node dicts
    """
    # TODO: implement
    raise NotImplementedError


def predict_bagged(X, trees):
    """
    Majority vote across trees. Ties broken by smaller label.

    Returns:
        np.ndarray, shape (n_samples,), dtype int
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data):
    test_name = input_data["test"]

    # === Stage 1 ===
    if test_name == "gini_pure":
        assert np.isclose(gini_impurity(np.array([1, 1, 1, 1])), 0.0)
        return True

    if test_name == "gini_even_split":
        assert np.isclose(gini_impurity(np.array([0, 0, 1, 1])), 0.5)
        return True

    if test_name == "gini_known_value":
        assert np.isclose(gini_impurity(np.array([0, 0, 0, 1])), 0.375)
        return True

    if test_name == "gini_empty":
        assert np.isclose(gini_impurity(np.array([])), 0.0)
        return True

    # === Stage 2 ===
    if test_name == "split_dataset_hand_computed":
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        Xl, yl, Xr, yr = split_dataset(X, y, feature=0, threshold=2.5)
        assert np.allclose(Xl, [[1.0], [2.0]])
        assert np.array_equal(yl, [0, 0])
        assert np.allclose(Xr, [[3.0], [4.0]])
        assert np.array_equal(yr, [1, 1])
        return True

    if test_name == "split_dataset_boundary_inclusive_left":
        X = np.array([[2.5]])
        y = np.array([9])
        Xl, yl, Xr, yr = split_dataset(X, y, feature=0, threshold=2.5)
        assert len(yl) == 1 and len(yr) == 0
        return True

    # === Stage 3 ===
    if test_name == "weighted_impurity_hand_computed":
        y_left = np.array([0, 0])
        y_right = np.array([0, 1])
        assert np.isclose(weighted_impurity(y_left, y_right), 0.25)
        return True

    if test_name == "weighted_impurity_unequal_sizes":
        y_left = np.array([0, 0, 0])
        y_right = np.array([0, 1])
        assert np.isclose(weighted_impurity(y_left, y_right), 0.2)
        return True

    # === Stage 4 ===
    if test_name == "best_split_obvious_case":
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        y = np.array([0, 0, 1, 1])
        result = best_split(X, y)
        assert result is not None
        feature, threshold, impurity = result
        assert feature == 0
        assert np.isclose(threshold, 2.5)
        assert np.isclose(impurity, 0.0)
        return True

    if test_name == "best_split_none_when_identical_rows":
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = np.array([0, 1, 0])
        result = best_split(X, y)
        assert result is None
        return True

    if test_name == "best_split_picks_informative_feature":
        rng = np.random.default_rng(0)
        n = 40
        feat0 = rng.uniform(0, 10, size=n)
        y = (rng.random(n) < 0.5).astype(int)
        feat1 = y.astype(float) * 10 + rng.uniform(-0.1, 0.1, size=n)
        X = np.column_stack([feat0, feat1])
        result = best_split(X, y)
        assert result is not None
        feature, threshold, impurity = result
        assert feature == 1
        assert impurity < 0.05
        return True

    # === Stage 5 ===
    if test_name == "majority_class_basic":
        assert majority_class(np.array([0, 0, 1])) == 0
        return True

    if test_name == "majority_class_tiebreak_smaller_label":
        assert majority_class(np.array([0, 1])) == 0
        assert majority_class(np.array([2, 3])) == 2
        return True

    if test_name == "build_tree_perfectly_fits_noiseless_rectangle":
        X, y = make_rect_data(300, seed=1, noise_frac=0.0)
        tree = build_tree(X, y, max_depth=6, min_samples_split=2)
        preds = predict(X, tree)
        assert np.mean(preds == y) > 0.99
        return True

    if test_name == "build_tree_respects_max_depth":
        X, y = make_rect_data(200, seed=2, noise_frac=0.0)
        tree = build_tree(X, y, max_depth=1, min_samples_split=2)
        assert tree_depth(tree) <= 1
        return True

    if test_name == "build_tree_respects_min_samples_split":
        X, y = make_rect_data(50, seed=3, noise_frac=0.0)
        tree = build_tree(X, y, max_depth=10, min_samples_split=1000)
        assert tree["leaf"] is True
        return True

    if test_name == "build_tree_stops_at_pure_node":
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 0, 0])
        tree = build_tree(X, y, max_depth=10, min_samples_split=2)
        assert tree["leaf"] is True
        assert tree["value"] == 0
        return True

    # === Stage 6 ===
    if test_name == "predict_one_traverses_correctly":
        tree = {
            "leaf": False, "feature": 0, "threshold": 2.5,
            "left": {"leaf": True, "value": 0},
            "right": {"leaf": True, "value": 1},
        }
        assert predict_one(np.array([1.0]), tree) == 0
        assert predict_one(np.array([3.0]), tree) == 1
        assert predict_one(np.array([2.5]), tree) == 0
        return True

    if test_name == "predict_shape_and_dtype":
        X, y = make_rect_data(20, seed=4)
        tree = build_tree(X, y, max_depth=4, min_samples_split=2)
        preds = predict(X, tree)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})
        return True

    # === Stage 7 ===
    if test_name == "bootstrap_sample_shape_and_reproducibility":
        X, y = make_rect_data(30, seed=5)
        Xs1, ys1 = bootstrap_sample(X, y, seed=42)
        Xs2, ys2 = bootstrap_sample(X, y, seed=42)
        assert Xs1.shape == X.shape
        assert ys1.shape == y.shape
        assert np.array_equal(Xs1, Xs2)
        assert np.array_equal(ys1, ys2)
        return True

    if test_name == "bootstrap_sample_has_repeats_typically":
        X, y = make_rect_data(50, seed=6)
        Xs, ys = bootstrap_sample(X, y, seed=0)
        n_unique_rows = len(np.unique(Xs, axis=0))
        assert n_unique_rows < len(Xs)
        return True

    if test_name == "fit_bagged_trees_returns_n_trees":
        X, y = make_rect_data(60, seed=7)
        trees = fit_bagged_trees(X, y, n_trees=8, max_depth=4, min_samples_split=2, seed=0)
        assert len(trees) == 8
        for t in trees:
            assert "leaf" in t
        return True

    if test_name == "predict_bagged_shape_and_range":
        X, y = make_rect_data(60, seed=8)
        trees = fit_bagged_trees(X, y, n_trees=8, max_depth=4, min_samples_split=2, seed=0)
        preds = predict_bagged(X, trees)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({0, 1})
        return True

    if test_name == "bagging_reduces_overfitting_vs_single_deep_tree":
        wins = 0
        n_trials = 6
        for trial in range(n_trials):
            X_train, y_train = make_rect_data(300, seed=100 + trial, noise_frac=0.15)
            X_test, y_test = make_rect_data(300, seed=200 + trial, noise_frac=0.0)
            tree = build_tree(X_train, y_train, max_depth=10, min_samples_split=2)
            tree_acc = np.mean(predict(X_test, tree) == y_test)
            forest = fit_bagged_trees(X_train, y_train, n_trees=15, max_depth=10, min_samples_split=2, seed=0)
            forest_acc = np.mean(predict_bagged(X_test, forest) == y_test)
            if forest_acc > tree_acc:
                wins += 1
        assert wins >= n_trials - 1, f"forest only beat single tree in {wins}/{n_trials} trials"
        return True

    raise ValueError(f"Unknown test: {test_name}")
