import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def make_blobs(centers, n_per_cluster=60, spread=0.3, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array(centers, dtype=float)
    X_parts, label_parts = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=spread, size=(n_per_cluster, 2))
        X_parts.append(pts)
        label_parts.append(np.full(n_per_cluster, i))
    X = np.vstack(X_parts)
    true_labels = np.concatenate(label_parts)
    perm = rng.permutation(len(X))
    return X[perm], true_labels[perm]


def clusters_match_ground_truth(pred_labels, true_labels):
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    for true_group in np.unique(true_labels):
        pred_for_group = pred_labels[true_labels == true_group]
        if len(np.unique(pred_for_group)) != 1:
            return False
    representative_preds = [
        pred_labels[true_labels == g][0] for g in np.unique(true_labels)
    ]
    if len(set(representative_preds)) != len(np.unique(true_labels)):
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 1: Pairwise squared distances
# ---------------------------------------------------------------------------
def squared_distances(X, centroids):
    """
    Squared Euclidean distance from every point to every centroid.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        centroids: np.ndarray, shape (k, n_features)

    Returns:
        np.ndarray, shape (n_samples, k), where entry [i, j] is
        ||X[i] - centroids[j]||^2
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Initialization
# ---------------------------------------------------------------------------
def init_centroids(X, k, seed=0):
    """
    Initialize k centroids by sampling k distinct points from X uniformly
    at random (no replacement).

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        k: int, number of clusters
        seed: RNG seed for reproducibility

    Returns:
        np.ndarray, shape (k, n_features)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Assignment step
# ---------------------------------------------------------------------------
def assign_clusters(X, centroids):
    """
    Assign each point to its nearest centroid.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        centroids: np.ndarray, shape (k, n_features)

    Returns:
        np.ndarray, shape (n_samples,), dtype int, values in [0, k)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Update step
# ---------------------------------------------------------------------------
def update_centroids(X, labels, k, old_centroids):
    """
    Recompute each centroid as the mean of the points currently assigned
    to it. If cluster j has no points assigned, leave its centroid
    unchanged (use old_centroids[j]).

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        labels: np.ndarray, shape (n_samples,), current cluster assignment
        k: int, number of clusters
        old_centroids: np.ndarray, shape (k, n_features)

    Returns:
        np.ndarray, shape (k, n_features)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Full training loop
# ---------------------------------------------------------------------------
def _fit_kmeans_single_run(X, k, n_iters, tol, seed):
    """
    A single assign/update run to (approximate) convergence.

    Returns:
        (centroids, labels, n_iters_run)
    """
    # TODO: implement
    raise NotImplementedError


def fit_kmeans(X, k, n_iters=100, tol=1e-6, seed=0, n_init=10):
    """
    Run k-means with multiple random restarts -- keep whichever run
    achieves the lowest final inertia.

    Args:
        X: np.ndarray, shape (n_samples, n_features)
        k: number of clusters
        n_iters: maximum iterations per restart
        tol: convergence tolerance
        seed: base seed (restart i uses seed + i)
        n_init: number of independent random restarts

    Returns:
        (centroids, labels, n_iters_run): the result from the restart
        with the lowest inertia.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
def inertia(X, labels, centroids):
    """
    Within-cluster sum of squared distances: for each point, the squared
    distance to its assigned centroid, summed over all points.

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

    # === Stage 1: Squared distances ===
    if test_name == "squared_distances_hand_computed":
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        centroids = np.array([[0.0, 0.0], [0.0, 4.0]])
        dists = squared_distances(X, centroids)
        expected = np.array([[0.0, 16.0], [25.0, 9.0]])
        assert dists.shape == (2, 2)
        assert np.allclose(dists, expected)
        return True

    if test_name == "squared_distances_symmetry_with_single_centroid":
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 3))
        centroid = X[0:1]
        dists = squared_distances(X, centroid)
        assert dists.shape == (10, 1)
        assert np.isclose(dists[0, 0], 0.0, atol=1e-10)
        return True

    # === Stage 2: Initialization ===
    if test_name == "init_centroids_shape":
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 4))
        centroids = init_centroids(X, k=5, seed=0)
        assert centroids.shape == (5, 4)
        return True

    if test_name == "init_centroids_are_actual_points":
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 2))
        centroids = init_centroids(X, k=3, seed=0)
        for c in centroids:
            assert np.any(np.all(np.isclose(X, c), axis=1)), "centroid is not one of X's rows"
        return True

    if test_name == "init_centroids_distinct_points":
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 2))
        centroids = init_centroids(X, k=4, seed=0)
        assert len(np.unique(centroids, axis=0)) == 4
        return True

    # === Stage 3: Assignment step ===
    if test_name == "assign_clusters_hand_computed":
        X = np.array([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5], [9.0, 9.0]])
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_clusters(X, centroids)
        assert list(labels) == [0, 1, 0, 1]
        return True

    if test_name == "assign_clusters_output_shape_and_range":
        rng = np.random.default_rng(5)
        X = rng.normal(size=(25, 3))
        centroids = X[:4]
        labels = assign_clusters(X, centroids)
        assert labels.shape == (25,)
        assert labels.min() >= 0
        assert labels.max() < 4
        return True

    # === Stage 4: Update step ===
    if test_name == "update_centroids_hand_computed":
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
        labels = np.array([0, 0, 1, 1])
        old_centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        new_centroids = update_centroids(X, labels, k=2, old_centroids=old_centroids)
        expected = np.array([[1.0, 0.0], [11.0, 10.0]])
        assert np.allclose(new_centroids, expected)
        return True

    if test_name == "update_centroids_empty_cluster_keeps_old":
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0, 0])
        old_centroids = np.array([[0.0, 0.0], [99.0, 99.0]])
        new_centroids = update_centroids(X, labels, k=2, old_centroids=old_centroids)
        assert np.allclose(new_centroids[0], [1.0, 0.0])
        assert np.allclose(new_centroids[1], [99.0, 99.0])
        assert not np.any(np.isnan(new_centroids))
        return True

    # === Stage 5: Full training loop ===
    if test_name == "single_run_terminates_early_on_convergence":
        X, _ = make_blobs(centers=[(0, 0), (20, 20)], n_per_cluster=40, spread=0.2, seed=12)
        _, _, n_iters_run = _fit_kmeans_single_run(X, k=2, n_iters=300, tol=1e-6, seed=3)
        assert n_iters_run < 300
        return True

    if test_name == "single_run_hits_iteration_cap_when_capped_low":
        X, _ = make_blobs(centers=[(0, 0), (20, 20)], n_per_cluster=40, spread=0.2, seed=12)
        _, _, n_iters_run = _fit_kmeans_single_run(X, k=2, n_iters=1, tol=1e-6, seed=3)
        assert n_iters_run <= 1
        return True

    if test_name == "fit_kmeans_recovers_well_separated_blobs":
        X, true_labels = make_blobs(
            centers=[(0, 0), (10, 10), (0, 10)], n_per_cluster=50, spread=0.4, seed=10
        )
        centroids, pred_labels, _ = fit_kmeans(X, k=3, n_iters=100, seed=1, n_init=10)
        assert clusters_match_ground_truth(pred_labels, true_labels)
        return True

    if test_name == "fit_kmeans_centroids_close_to_true_centers":
        true_centers = [(0, 0), (10, 10), (0, 10)]
        X, true_labels = make_blobs(centers=true_centers, n_per_cluster=50, spread=0.4, seed=11)
        centroids, pred_labels, _ = fit_kmeans(X, k=3, n_iters=100, seed=2, n_init=10)
        true_centers = np.array(true_centers, dtype=float)
        for c in centroids:
            dists_to_true = np.sum((true_centers - c) ** 2, axis=1)
            assert np.min(dists_to_true) < 1.0
        return True

    if test_name == "fit_kmeans_reliable_across_many_datasets":
        n_fail = 0
        n_trials = 15
        for data_seed in range(n_trials):
            X, true_labels = make_blobs(
                centers=[(0, 0), (10, 10), (0, 10)], n_per_cluster=50, spread=0.4, seed=data_seed
            )
            _, pred_labels, _ = fit_kmeans(X, k=3, n_iters=100, seed=0, n_init=10)
            if not clusters_match_ground_truth(pred_labels, true_labels):
                n_fail += 1
        assert n_fail <= 1, f"{n_fail}/{n_trials} datasets failed"
        return True

    if test_name == "fit_kmeans_picks_lower_inertia_than_typical_single_run":
        X, _ = make_blobs(centers=[(0, 0), (10, 10), (0, 10)], n_per_cluster=50, spread=0.4, seed=14)
        multi_centroids, multi_labels, _ = fit_kmeans(X, k=3, n_iters=100, seed=0, n_init=10)
        multi_inertia = inertia(X, multi_labels, multi_centroids)
        single_inertias = []
        for s in range(10):
            c, l, _ = _fit_kmeans_single_run(X, k=3, n_iters=100, tol=1e-6, seed=s)
            single_inertias.append(inertia(X, l, c))
        assert multi_inertia <= min(single_inertias) + 1e-6
        return True

    # === Stage 6: Evaluation ===
    if test_name == "inertia_hand_computed":
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[1.0, 0.0], [11.0, 0.0]])
        assert np.isclose(inertia(X, labels, centroids), 4.0)
        return True

    if test_name == "inertia_lower_after_fitting_than_at_init":
        X, _ = make_blobs(centers=[(0, 0), (10, 10), (0, 10)], n_per_cluster=50, spread=0.4, seed=13)
        ic = init_centroids(X, k=3, seed=4)
        il = assign_clusters(X, ic)
        init_val = inertia(X, il, ic)
        fc, fl, _ = fit_kmeans(X, k=3, n_iters=100, seed=4)
        fitted_val = inertia(X, fl, fc)
        assert fitted_val <= init_val
        return True

    raise ValueError(f"Unknown test: {test_name}")
