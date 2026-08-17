# K-Means Clustering
Difficulty: Medium

Implement k-means clustering from scratch using NumPy, progressing through six stages.

Cluster labels are arbitrary — tests check clustering quality using permutation-invariant comparisons (grouping correctness, not label numbers).

Implement the functions below. The `solve` dispatcher and helper utilities are provided — do not modify them.

## Stage 1: Pairwise Squared Distances

Implement `squared_distances(X, centroids)` — returns an `(n_samples, k)` array of squared Euclidean distances from every point to every centroid.

## Stage 2: Initialization

Implement `init_centroids(X, k, seed)` — sample `k` distinct points from `X` uniformly at random (no replacement).

## Stage 3: Assignment Step

Implement `assign_clusters(X, centroids)` — assign each point to its nearest centroid.

## Stage 4: Update Step

Implement `update_centroids(X, labels, k, old_centroids)` — recompute each centroid as the mean of its assigned points. If a cluster has no points, keep the old centroid.

## Stage 5: Full Training Loop

Implement `_fit_kmeans_single_run(X, k, n_iters, tol, seed)` — run assign/update until convergence or iteration cap.

Implement `fit_kmeans(X, k, n_iters, tol, seed, n_init)` — run multiple random restarts and keep the result with lowest inertia.

## Stage 6: Evaluation

Implement `inertia(X, labels, centroids)` — within-cluster sum of squared distances.

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
    # Implement the functions above; do not modify solve().
```
