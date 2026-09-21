# Activation Patching and Causal Tracing
Difficulty: Hard

Implement causal interventions on a tiny deterministic residual sequence model. No checkpoint download or GPU is needed: all tests use NumPy arrays smaller than a few kilobytes.

You will compare:
- a **clean** input, where the model has the information needed for the target behavior
- a **corrupted** input, where that information is changed
- a **patched** run, where selected corrupted-run activations are replaced by clean-run activations

The metric is the next-token logit difference between a positive and negative token. A large recovery after patching is evidence that the patched activation is causally relevant to this behavior; it is not, by itself, a complete explanation of the model.

## Stage 1: Behavioral Metric

Implement `logit_difference(logits, positive_token_id, negative_token_id)`. Return one score per batch item from the final sequence position.

## Stage 2: Position Patching

Implement `patch_positions(corrupted_activation, clean_activation, position_mask)` without mutating either input. The position mask has shape `(batch, sequence)` and broadcasts across `d_model`.

## Stage 3: One Causal Intervention

Implement `activation_patch_effect(...)`:
1. cache the clean activation at the requested layer
2. run the corrupted baseline
3. run the corrupted input again, replacing selected activations with the clean values
4. return baseline, patched, and clean logit differences plus recovery

Define recovery as `patched_logit_diff - corrupted_logit_diff` for each example.

## Stage 4: Layer Scan

Implement `scan_layers(...)`, running the intervention at every layer and returning a `(n_layers, batch)` array of recoveries.

## Stage 5: Normalized Recovery

Implement `normalized_recovery(clean_score, corrupted_score, patched_score, eps=1e-8)`:

`(patched - corrupted) / (clean - corrupted)`

Use a signed epsilon-safe denominator so near-zero denominators remain finite. Do not clip the result: interventions can overshoot or move in the wrong direction.

## Laptop scope

The supplied model has four layers, eight hidden dimensions, and a tiny vocabulary. This preserves the mechanics of clean/corrupted caching, position-specific replacement, and causal effect measurement while avoiding Hugging Face downloads and expensive inference.
