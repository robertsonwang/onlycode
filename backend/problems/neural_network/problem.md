# Neural Network (1 Hidden Layer)
Difficulty: Hard

Implement a small neural network with one hidden layer (ReLU activation) and a sigmoid output, trained with backpropagation for binary classification.

**Architecture:**
```
X (n_samples, n_input)
  -> Z1 = X @ W1 + b1        (n_samples, n_hidden)
  -> A1 = relu(Z1)           (n_samples, n_hidden)
  -> Z2 = A1 @ W2 + b2       (n_samples, n_output)   n_output = 1
  -> A2 = sigmoid(Z2)        (n_samples, n_output)
  -> y_pred = A2 flattened to (n_samples,)
```

Parameters are carried as a dict: `params = {"W1": ..., "b1": ..., "W2": ..., "b2": ...}`

Implement the functions below. The `solve` dispatcher is provided — do not modify it.

## Stage 1: Activations

Implement `relu(z)`, `relu_derivative(z)`, and `sigmoid(z)`. Sigmoid must be numerically stable.

## Stage 2: Forward Pass

Implement `init_params(n_input, n_hidden, n_output, seed)` — initialize weights randomly (not zeros!) and biases to zeros.

Implement `forward(X, params)` — returns `(y_pred, cache)` where cache stores intermediate values for backprop.

## Stage 3: Loss

Implement `bce_loss(y_pred, y_true)` — mean binary cross-entropy with clipping.

## Stage 4: Backward Pass

Implement `backward(y_true, params, cache)` — backpropagate BCE loss through the network, returning gradients `{"dW1", "db1", "dW2", "db2"}`.

## Stage 5: Training Loop

Implement `train(X, y, n_hidden, lr, n_iters, seed)` — batch gradient descent, returning `(params, loss_history)`.

## Stage 6: Prediction + Evaluation

Implement `predict(X, params, threshold)` and `accuracy(y_pred, y_true)`.

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
    # Implement the functions above; do not modify solve().
```
