# Softmax Regression
Difficulty: Medium

Implement multiclass softmax (multinomial logistic) regression from scratch using NumPy, progressing through six stages.

Implement the functions below. The `solve` dispatcher and helper utilities are provided — do not modify them.

## Stage 1: Softmax

Implement `softmax(logits)` — converts a matrix of logits to probabilities. Each row should sum to 1. Must be numerically stable (subtract the row max before exponentiating).

## Stage 2: One-Hot Encoding

Implement `one_hot_encode(y, num_classes)` — converts integer class labels to one-hot vectors.

## Stage 3: Predict Probabilities

Implement `predict_proba(X, W)` — compute class probabilities via `softmax(X @ W)`.

## Stage 4: Prediction

Implement `predict(X, W)` — return the class with the highest probability for each example.

## Stage 5: Cross-Entropy Loss

Implement `cross_entropy_loss(y, probabilities)` — mean cross-entropy loss. Clip probabilities to avoid `log(0)`.

## Stage 5b: Gradient Computation

Implement `compute_gradient(X, y, W)` — gradient of mean cross-entropy loss w.r.t. W:

```
gradient = (1 / n_samples) * X.T @ (probabilities - y_one_hot)
```

## Stage 6: Training

Implement `fit(X, y, num_classes, learning_rate, epochs)` — train via gradient descent, returning the learned weight matrix W.

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
    # Implement the functions above; do not modify solve().
```
