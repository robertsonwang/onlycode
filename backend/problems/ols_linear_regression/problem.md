# OLS Linear Regression
Difficulty: Medium

Implement ordinary least squares (OLS) linear regression from scratch using NumPy, progressing through six stages.

Implement the functions below. The `solve` dispatcher and helper utilities are provided — do not modify them.

## Stage 1: Loss Function

Implement `mse_loss(y_pred, y_true)` — returns the mean squared error between predictions and true values.

## Stage 2: Gradients

Implement `compute_gradients(X, y, weights, bias)` — returns `(grad_weights, grad_bias)` for the linear model `y_pred = X @ weights + bias`.

## Stage 3: Training Loop (Batch Gradient Descent)

Implement `fit_linear_regression(X, y, lr, n_iters)` — fits weights and bias via batch gradient descent, returning `(weights, bias)`.

## Stage 4: L2 Regularization (Ridge Regression)

Implement `compute_gradients_l2(X, y, weights, bias, alpha)` — same as `compute_gradients` but adds an L2 penalty `alpha * ||weights||^2` to the loss. The bias is **not** regularized.

Implement `fit_ridge_regression(X, y, lr, n_iters, alpha)` — fits weights and bias with L2 regularization.

## Stage 5: Mini-batch SGD

Implement `fit_sgd(X, y, lr, n_epochs, batch_size, alpha, seed)` — trains via mini-batch stochastic gradient descent. Shuffle the data at the start of every epoch using the seeded RNG. Reuse `compute_gradients_l2`.

## Stage 6: Evaluation

Implement `r_squared(y_pred, y_true)` — returns the R² (coefficient of determination).

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
    # Implement the functions above; do not modify solve().
```
