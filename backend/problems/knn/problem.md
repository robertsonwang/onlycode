# K-Nearest Neighbors
Difficulty: Medium

Implement k-nearest neighbors (KNN) classification from scratch using NumPy, progressing through six stages.

## Stage 1: Distance Computation

Implement `euclidean_distances(X_query, X_train)` — returns an `(n_query, n_train)` array of Euclidean distances.

## Stage 2: Finding Neighbors

Implement `find_k_neighbors(distances, k)` — returns the indices of the k nearest training points for each query point.

## Stage 3: Prediction

Implement `predict(X_query, X_train, y_train, k)` — predict class labels via majority vote among k nearest neighbors. Break ties by choosing the smaller label.

## Stage 4: Accuracy

Implement `accuracy(y_pred, y_true)` — fraction of correct predictions.

## Stage 5: Cross-Validation

Implement `kfold_split(n_samples, n_folds, seed)` — return a list of `(train_indices, val_indices)` tuples for k-fold CV.

Implement `cross_val_accuracy(X, y, k, n_folds, seed)` — average accuracy across folds.

## Stage 6: Choosing k

Implement `best_k(X, y, k_candidates, n_folds, seed)` — return the k with the highest cross-validation accuracy (ties broken by smaller k).

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
```
