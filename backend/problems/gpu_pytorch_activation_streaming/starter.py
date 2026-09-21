from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------------
# CUDA model used by tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This exercise requires an NVIDIA CUDA GPU")


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, tuple_output: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(4 * d_model, d_model)
        self.tuple_output = tuple_output

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, None]:
        result = x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))
        return (result, None) if self.tuple_output else result


class TinyGPUTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=192, n_layers=4, tuple_layer=None):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, tuple_output=(i == tuple_layer)) for i in range(n_layers)]
        )
        self.forward_calls = 0
        self.fail_on_forward = False

    @property
    def active_hook_count(self) -> int:
        return sum(len(block._forward_hooks) for block in self.blocks)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.forward_calls += 1
        if self.fail_on_forward:
            raise RuntimeError("simulated CUDA forward failure")
        x = self.embedding(tokens)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        for block in self.blocks:
            output = block(x)
            x = output[0] if isinstance(output, tuple) else output
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)
        return x


# ---------------------------------------------------------------------------
# Stage 1: CUDA token batching
# ---------------------------------------------------------------------------
def pad_token_batch(
    sequences: Sequence[Sequence[int]],
    pad_id: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right-padded `(tokens, attention_mask)` tensors on device."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Mask-aware GPU pooling
# ---------------------------------------------------------------------------
def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average `(batch, sequence, d_model)` over unmasked positions."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Extract one layer without retaining a graph or hook
# ---------------------------------------------------------------------------
def extract_layer_activation(
    model: TinyGPUTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Return a detached CPU float32 copy of `model.blocks[layer]` output."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Stream selected layers to CPU in bounded batches
# ---------------------------------------------------------------------------
def stream_layer_activations(
    model: TinyGPUTransformer,
    sequences: Sequence[Sequence[int]],
    layers: Sequence[int],
    batch_size: int = 16,
    pad_id: int = 0,
) -> dict[int, torch.Tensor]:
    """Return `{layer: CPU tensor}` using one CUDA forward pass per batch."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test dispatcher (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def solve(input_data: dict[str, Any]) -> bool:
    _require_cuda()
    test_name = input_data["test"]

    if test_name == "cuda_is_required":
        assert torch.cuda.is_available()
        return True

    if test_name == "padding_device_dtype_and_values":
        tokens, mask = pad_token_batch([[3, 4, 5], [6]], pad_id=0)
        assert tokens.is_cuda and mask.is_cuda
        assert tokens.dtype == torch.long and mask.dtype == torch.bool
        assert tokens.cpu().tolist() == [[3, 4, 5], [6, 0, 0]]
        assert mask.cpu().tolist() == [[True, True, True], [True, False, False]]
        return True

    if test_name == "masked_mean_ignores_padding":
        hidden = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]], [[2.0, 6.0], [50.0, 50.0], [60.0, 60.0]]],
            device="cuda",
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool, device="cuda")
        expected = torch.tensor([[2.0, 3.0], [2.0, 6.0]], device="cuda")
        assert torch.allclose(masked_mean(hidden, mask), expected)
        return True

    if test_name == "extract_is_detached_cpu_float32":
        model = TinyGPUTransformer().cuda().half().eval()
        tokens, mask = pad_token_batch([[1, 2, 3], [4, 5]])
        activation = extract_layer_activation(model, tokens, mask, layer=2)
        assert activation.device.type == "cpu"
        assert activation.dtype == torch.float32
        assert not activation.requires_grad and activation.grad_fn is None
        assert model.active_hook_count == 0
        return True

    if test_name == "extract_handles_tuple_output":
        model = TinyGPUTransformer(tuple_layer=1).cuda().eval()
        tokens, mask = pad_token_batch([[1, 2], [3, 4]])
        activation = extract_layer_activation(model, tokens, mask, layer=1)
        assert activation.shape == (2, 2, 192)
        assert model.active_hook_count == 0
        return True

    if test_name == "hook_cleanup_after_failure":
        model = TinyGPUTransformer().cuda().eval()
        model.fail_on_forward = True
        tokens, mask = pad_token_batch([[1, 2]])
        try:
            extract_layer_activation(model, tokens, mask, layer=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forward failure must propagate")
        assert model.active_hook_count == 0
        return True

    if test_name == "streaming_shapes_order_and_forward_count":
        model = TinyGPUTransformer(d_model=128).cuda().eval()
        sequences = [[i + 1] * ((i % 5) + 1) for i in range(19)]
        cached = stream_layer_activations(model, sequences, [0, 2, 3], batch_size=6)
        assert set(cached) == {0, 2, 3}
        assert all(value.shape == (19, 5, 128) for value in cached.values())
        assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in cached.values())
        assert model.forward_calls == 4
        assert model.active_hook_count == 0
        return True

    if test_name == "streaming_matches_full_batch":
        sequences = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11, 12]]
        batched_model = TinyGPUTransformer(d_model=96).cuda().eval()
        full_model = TinyGPUTransformer(d_model=96).cuda().eval()
        batched = stream_layer_activations(batched_model, sequences, [1, 3], batch_size=2)
        full = stream_layer_activations(full_model, sequences, [1, 3], batch_size=20)
        assert torch.allclose(batched[1], full[1], atol=1e-5)
        assert torch.allclose(batched[3], full[3], atol=1e-5)
        return True

    raise ValueError(f"Unknown test: {test_name}")
