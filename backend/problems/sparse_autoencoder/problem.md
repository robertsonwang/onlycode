# Sparse Autoencoder for Toy Activations
Difficulty: Hard

Build and train a sparse autoencoder (SAE) on small synthetic activation vectors. The model has a ReLU encoder, an overcomplete latent dictionary, and a linear decoder:

- `pre = X @ W_enc + b_enc`
- `features = relu(pre)`
- `reconstruction = features @ W_dec + b_dec`

Optimize:

`mean((reconstruction - X) ** 2) + l1_coeff * mean(abs(features))`

The L1 term encourages sparse feature activations. Reconstruction quality and sparsity must be reported separately; combining them into one number hides the tradeoff.

## Stage 1: Initialization and Forward Pass

Implement `init_sae(d_input, d_hidden, seed=0)` and `sae_forward(X, params)`. Parameter shapes are:
- `W_enc`: `(d_input, d_hidden)`
- `b_enc`: `(d_hidden,)`
- `W_dec`: `(d_hidden, d_input)`
- `b_dec`: `(d_input,)`

Return `(reconstruction, cache)` from the forward pass. The cache must include `X`, `pre`, and `features`.

## Stage 2: Losses

Implement `sae_loss(X, reconstruction, features, l1_coeff)`, returning a dict with `mse`, `l1`, and `total`.

## Stage 3: Backpropagation

Implement `sae_gradients(X, params, cache, l1_coeff)`. Gradients must match every parameter shape and include the gradient of both reconstruction loss and the L1 activation penalty.

## Stage 4: Decoder Normalization

Implement `normalize_decoder(params, eps=1e-8)` so every row of `W_dec` has unit L2 norm. This removes an easy scaling loophole where encoder features shrink while decoder weights grow. Return a new parameter dict; do not mutate the input.

## Stage 5: Training

Implement full-batch gradient descent in `train_sae(...)`. Return `(params, history)`, where each history entry is the loss dict for that step. Keep execution deterministic.

## Stage 6: Feature Statistics

Implement `feature_statistics(features, threshold=1e-8)` returning:
- `fraction_active`: fraction of all feature entries above the threshold
- `mean_l0`: mean number of active features per example
- `mean_activation`: mean absolute feature activation

## Laptop scope

Tests use fewer than 128 examples, 8 input dimensions, and 16 SAE features. This exercises the architecture, L1 gradient, scaling issue, and evaluation logic without storing real transformer activations or requiring a GPU.
