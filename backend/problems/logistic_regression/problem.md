# Logistic Regression
Difficulty: Medium

Implement binary logistic regression from scratch using NumPy, progressing through six stages.

Implement the functions below. The `solve` dispatcher and helper utilities are provided — do not modify them.

## Stage 1: Sigmoid

Implement `sigmoid(z)` — the logistic sigmoid function, applied elementwise. Must be numerically stable for very large/small inputs (no overflow or NaN).

## Stage 2: Binary Cross-Entropy Loss

Implement `bce_loss(y_pred, y_true)` — mean binary cross-entropy. Handle edge cases where `y_pred` is at 0 or 1 (clip to avoid `log(0)`).

## Stage 3: Gradients

Implement `compute_gradients(X, y, weights, bias)` — gradients of mean BCE loss w.r.t. weights and bias, where `y_pred = sigmoid(X @ weights + bias)`.

## Stage 4: Training Loop (Batch Gradient Descent)

Implement `fit_logistic_regression(X, y, lr, n_iters)` — fits weights and bias via batch gradient descent, returning `(weights, bias)`.

## Stage 5: L2 Regularization

Implement `compute_gradients_l2(X, y, weights, bias, alpha)` — same as `compute_gradients` but adds L2 penalty `alpha * ||weights||^2`. Bias is **not** regularized.

Implement `fit_ridge_logistic_regression(X, y, lr, n_iters, alpha)` — fits with L2 regularization.

## Stage 6: Evaluation

Implement:
- `predict_proba(X, weights, bias)` — predicted probabilities
- `predict(X, weights, bias, threshold=0.5)` — hard labels (0 or 1)
- `accuracy(y_pred, y_true)` — fraction correct
- `precision_recall_f1(y_pred, y_true)` — precision, recall, F1 for the positive class

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
    # Implement the functions above; do not modify solve().
```
