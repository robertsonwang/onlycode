# [GPU] PyTorch Activation Streaming
Difficulty: Hard
Requires: NVIDIA CUDA GPU (8 GB+ VRAM)

> **GPU REQUIRED:** Tests intentionally fail when `torch.cuda.is_available()` is false. Run the application on the 3090 with the `gpu` dependency group installed.

Implement a production-style activation extraction pipeline using real PyTorch hooks. The supplied transformer-like model is randomly initialized and requires no model download.

## Stage 1: CUDA Padding and Masking

Implement `pad_token_batch(sequences, pad_id=0, device="cuda")`:
- right-pad variable-length token sequences
- return `torch.long` tokens and `torch.bool` attention masks
- place both tensors on the requested device
- reject empty batches and empty sequences

## Stage 2: Masked Pooling

Implement `masked_mean(hidden, attention_mask)`. Inputs remain on GPU and padding must not affect the result.

## Stage 3: Safe Hook Extraction

Implement `extract_layer_activation(model, tokens, attention_mask, layer)`:
- hook `model.blocks[layer]`
- run under `torch.inference_mode()`
- handle block outputs that are either tensors or tuples
- return a detached CPU `float32` tensor
- remove the hook in `finally`, including when forward raises

## Stage 4: Streaming Multiple Layers

Implement `stream_layer_activations(model, sequences, layers, batch_size=16, pad_id=0)`:
- run exactly one forward pass per batch, not one per layer
- keep model inputs on its CUDA device
- immediately detach, convert to `float32`, and move each captured activation to CPU
- preserve example order
- return `{layer: tensor}` with shape `(n_examples, global_padded_length, d_model)`
- leave no hooks registered

The test model counts forward calls and checks hook cleanup. A solution that retains GPU tensors or autograd graphs defeats the memory-bounded design.

## Remote setup

Install the optional dependencies with `uv sync --extra gpu`, verify that the CUDA PyTorch build is selected, then start the backend on the GPU machine. This problem receives a 120-second runner timeout and disables the generic virtual-memory cap because CUDA reserves large virtual address ranges.
