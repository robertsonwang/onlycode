# Residual Stream Caching on a Tiny Transformer
Difficulty: Medium

Practice the engineering mechanics behind extracting transformer activations without downloading a model or requiring a GPU. The supplied NumPy model is deliberately tiny, but its hook lifecycle mirrors PyTorch: register a callback, run a forward pass, and always remove the handle.

This laptop-sized drill uses at most a few kilobytes per test. In production, the same structure applies to a Hugging Face model, with tensors detached and moved to CPU before caching.

## Stage 1: Pad and Mask

Implement `pad_sequences(sequences, pad_id=0)`:
- right-pad to the longest sequence
- return integer `tokens` and boolean `attention_mask`, both shape `(batch, max_length)`
- reject an empty batch or an empty sequence with `ValueError`

## Stage 2: Masked Pooling

Implement `masked_mean(hidden, attention_mask)` for hidden states of shape `(batch, sequence, d_model)`. Padding must not affect the result.

## Stage 3: One-Layer Extraction

Implement `extract_residual_stream(model, tokens, attention_mask, layer)`:
- register one forward hook at `layer`
- run inference once
- cache an independent copy of that layer's output
- remove the hook in a `finally` block so exceptions cannot leak hooks

The layer index is zero-based and refers to the residual stream after that block.

## Stage 4: Batched Caching

Implement `cache_residual_stream(model, sequences, layers, batch_size=8, pad_id=0)`:
- pad once, then process examples in batches
- collect every requested layer during the same forward pass per batch
- return `{layer: activation_array}`
- preserve input order and return arrays shaped `(n_examples, padded_length, d_model)`
- never leave hooks registered

Do not perform one forward pass per layer. That works numerically but is unnecessarily expensive.

## Why this is scoped down

The exercise avoids Hugging Face downloads, tokenizer variability, autograd memory, and multi-gigabyte checkpoints. Once this passes, porting it to PyTorch mainly means replacing NumPy copies with `output.detach().cpu()` and registering hooks on actual transformer blocks.
