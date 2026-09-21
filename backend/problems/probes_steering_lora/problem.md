# Probes, Steering Vectors, and LoRA Shifts
Difficulty: Hard

Practice three related representation-analysis tools on tiny NumPy activations:
- train a held-out linear probe without leaking test statistics
- construct and causally intervene with a mean-difference steering direction
- compute the activation shift induced by a low-rank adapter

The tests use synthetic matrices, so the full drill runs on CPU without a model download.

## Stage 1: Leakage-Safe Standardization

Implement:
- `fit_standardizer(X_train)` returning training-set mean and standard deviation
- `apply_standardizer(X, stats)` using those saved statistics

Replace zero standard deviations with `1.0`. Never fit preprocessing on held-out data.

## Stage 2: Linear Probe

Implement binary logistic regression in `train_linear_probe(X, y, lr=0.1, n_steps=500, l2=0.0)`. Return `{"weight": ..., "bias": ...}`. Implement `probe_predict` and `accuracy`.

The probe demonstrates decodability (correlation), not causality.

## Stage 3: Steering and Ablation

Implement:
- `mean_difference_direction(X, y)`: normalized class-1 mean minus class-0 mean
- `steer_activations(X, direction, strength)`: add the direction to every example
- `project_out_direction(X, direction)`: remove each example's component along the direction
- `probe_accuracy_drop_after_ablation(X, y, probe, direction)`

A drop after intervention is a stronger causal check than held-out probe accuracy alone, though it can still be confounded by distributed or entangled features.

## Stage 4: LoRA Representation Shift

For a base linear layer `Y = X @ W`, a LoRA adapter contributes:

`delta = (alpha / rank) * X @ A @ B`

Implement `lora_delta`, then `representation_shift_metrics(base, adapted)`, returning:
- `mean_l2`: mean row-wise L2 shift
- `relative_l2`: global shift norm divided by base norm
- `mean_cosine`: mean row-wise cosine similarity

Use epsilon-safe denominators.

## Stage 5: Layerwise Shift Scan

Implement `layerwise_lora_shifts(activations, adapters)`. Each activation matrix corresponds to one layer; each adapter is a dict containing `A`, `B`, and `alpha`. Return one metrics dict per layer.

## Laptop scope

Everything is linear algebra over at most a few hundred rows and tens of dimensions. This isolates methodology and tensor-shape reasoning before moving to a real checkpoint on the remote 3090.
