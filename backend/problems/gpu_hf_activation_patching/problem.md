# [GPU] Hugging Face Activation Patching
Difficulty: Hard
Requires: NVIDIA CUDA GPU (12 GB+ VRAM)

> **GPU REQUIRED:** Tests intentionally fail when CUDA is unavailable. They instantiate a random GPT-2-style model locally, so no checkpoint or network access is required.

Implement activation patching against a real Hugging Face causal language-model interface. Unlike the NumPy drill, transformer blocks return tuples and model calls return structured output objects.

## Stage 1: Last-Token Metric

Implement `last_token_logit_difference(logits, attention_mask, positive_token_id, negative_token_id)`:
- select the final non-padding position independently for each batch item
- return positive-minus-negative logits with shape `(batch,)`
- keep computation on the input device

## Stage 2: Tuple-Preserving Intervention

Implement `replace_positions(block_output, clean_hidden, position_mask)`:
- accept either a tensor or a block-output tuple
- replace only masked `(batch, sequence)` positions
- preserve all non-hidden tuple elements unchanged
- do not mutate inputs

This return contract matters: a forward hook replaces the module output when it returns a non-`None` value.

## Stage 3: Residual Capture

Implement `capture_block_residual(model, input_ids, attention_mask, layer)`:
- hook `model.transformer.h[layer]`
- run with `use_cache=False` under `torch.inference_mode()`
- return a detached clone of the block's hidden-state output
- remove the hook even after exceptions

## Stage 4: Causal Effect

Implement `hf_activation_patch_effect(...)`:
1. cache the clean residual
2. measure clean and corrupted next-token logit differences
3. patch selected clean positions into the corrupted run
4. return detached CPU `float32` vectors for `clean`, `corrupted`, `patched`, and `recovery`

Define `recovery = patched - corrupted`.

## Stage 5: Efficient Layer Scan

Implement `scan_hf_layers(...)` returning shape `(n_layers, batch)` on CPU. Cache every clean-layer residual in one clean forward pass, compute the corrupted baseline once, and then perform one patched pass per layer. The expected total is `n_layers + 2` model forwards, not three forwards per layer.

## Remote setup

Run `uv sync --extra gpu` on the CUDA host and ensure that the installed PyTorch build detects the 3090. Start the backend there. The model is intentionally small enough for rapid tests, but the same code applies to larger Hugging Face causal LMs.
