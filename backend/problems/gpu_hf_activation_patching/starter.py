from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel


# ---------------------------------------------------------------------------
# Offline Hugging Face model used by tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This exercise requires an NVIDIA CUDA GPU")


def make_tiny_hf_model(
    n_layer: int = 3,
    n_embd: int = 128,
    n_head: int = 4,
    vocab_size: int = 128,
    seed: int = 0,
) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_ctx=32,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    return GPT2LMHeadModel(config).cuda().eval()


def _model_logits(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


# ---------------------------------------------------------------------------
# Stage 1: Mask-aware next-token metric
# ---------------------------------------------------------------------------
def last_token_logit_difference(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> torch.Tensor:
    """Return positive-minus-negative logits at each row's last real token."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: Replace hidden states while preserving block output structure
# ---------------------------------------------------------------------------
def replace_positions(
    block_output: torch.Tensor | tuple[Any, ...],
    clean_hidden: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor | tuple[Any, ...]:
    """Patch hidden states and preserve any remaining tuple elements."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: Safely capture one Hugging Face block residual
# ---------------------------------------------------------------------------
def capture_block_residual(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Return a detached clone of `model.transformer.h[layer]` hidden output."""
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: Patch one layer and measure causal recovery
# ---------------------------------------------------------------------------
def hf_activation_patch_effect(
    model: GPT2LMHeadModel,
    clean_input_ids: torch.Tensor,
    clean_attention_mask: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    corrupted_attention_mask: torch.Tensor,
    layer: int,
    position_mask: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> dict[str, torch.Tensor]:
    """
    Return CPU float32 vectors with keys clean, corrupted, patched, recovery.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: Efficiently scan every transformer block
# ---------------------------------------------------------------------------
def scan_hf_layers(
    model: GPT2LMHeadModel,
    clean_input_ids: torch.Tensor,
    clean_attention_mask: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    corrupted_attention_mask: torch.Tensor,
    position_mask: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> torch.Tensor:
    """Return CPU recoveries shaped `(n_layers, batch)` in n_layers + 2 passes."""
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

    if test_name == "last_token_metric_uses_mask":
        logits = torch.zeros(2, 4, 10, device="cuda")
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], device="cuda")
        logits[0, 1, 7], logits[0, 1, 3] = 5.0, 2.0
        logits[1, 2, 7], logits[1, 2, 3] = -1.0, 4.0
        result = last_token_logit_difference(logits, mask, 7, 3)
        assert torch.allclose(result, torch.tensor([3.0, -5.0], device="cuda"))
        return True

    if test_name == "replace_positions_preserves_tuple":
        corrupted = torch.zeros(2, 3, 4, device="cuda")
        clean = torch.arange(24, device="cuda").reshape(2, 3, 4).float()
        auxiliary = (torch.tensor(9, device="cuda"), "sentinel")
        output = (corrupted, *auxiliary)
        mask = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool, device="cuda")
        patched = replace_positions(output, clean, mask)
        assert isinstance(patched, tuple) and patched[1:] == auxiliary
        assert torch.equal(patched[0][0, 0], clean[0, 0])
        assert torch.equal(patched[0][1, 1], clean[1, 1])
        assert torch.equal(corrupted, torch.zeros_like(corrupted))
        return True

    if test_name == "capture_matches_direct_hook":
        model = make_tiny_hf_model(seed=2)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        mask = torch.ones_like(ids)
        direct = {}
        def capture_hook(module, inputs, output):
            direct["value"] = output[0].detach().clone()

        handle = model.transformer.h[1].register_forward_hook(capture_hook)
        with torch.inference_mode():
            _model_logits(model, ids, mask)
        handle.remove()
        actual = capture_block_residual(model, ids, mask, 1)
        assert torch.allclose(actual, direct["value"])
        return True

    if test_name == "capture_removes_hook":
        model = make_tiny_hf_model(seed=3)
        ids = torch.tensor([[1, 2, 3]], device="cuda")
        mask = torch.ones_like(ids)
        before = len(model.transformer.h[0]._forward_hooks)
        capture_block_residual(model, ids, mask, 0)
        after = len(model.transformer.h[0]._forward_hooks)
        assert before == after == 0
        return True

    if test_name == "patch_effect_matches_manual_intervention":
        model = make_tiny_hf_model(seed=4)
        clean = torch.tensor([[2, 4, 6, 8], [1, 3, 5, 7]], device="cuda")
        corrupt = torch.tensor([[9, 4, 6, 8], [10, 3, 5, 7]], device="cuda")
        attention = torch.ones_like(clean)
        positions = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool, device="cuda")
        result = hf_activation_patch_effect(
            model, clean, attention, corrupt, attention, 1, positions, 11, 12
        )
        clean_hidden = capture_block_residual(model, clean, attention, 1)
        box = {}
        def patch_hook(module, inputs, output):
            box["seen"] = True
            return replace_positions(output, clean_hidden, positions)
        handle = model.transformer.h[1].register_forward_hook(patch_hook)
        with torch.inference_mode():
            patched_logits = _model_logits(model, corrupt, attention)
        handle.remove()
        with torch.inference_mode():
            corrupt_logits = _model_logits(model, corrupt, attention)
        expected_patched = last_token_logit_difference(patched_logits, attention, 11, 12).cpu()
        expected_corrupt = last_token_logit_difference(corrupt_logits, attention, 11, 12).cpu()
        assert box["seen"]
        assert torch.allclose(result["patched"], expected_patched.float(), atol=1e-5)
        assert torch.allclose(result["recovery"], (expected_patched - expected_corrupt).float(), atol=1e-5)
        return True

    if test_name == "empty_patch_has_zero_recovery":
        model = make_tiny_hf_model(seed=5)
        clean = torch.tensor([[1, 2, 3]], device="cuda")
        corrupt = torch.tensor([[4, 2, 3]], device="cuda")
        attention = torch.ones_like(clean)
        positions = torch.zeros_like(clean, dtype=torch.bool)
        result = hf_activation_patch_effect(
            model, clean, attention, corrupt, attention, 2, positions, 9, 10
        )
        assert torch.allclose(result["recovery"], torch.zeros(1), atol=1e-6)
        return True

    if test_name == "layer_scan_is_correct_and_efficient":
        model = make_tiny_hf_model(n_layer=3, seed=6)
        clean = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        corrupt = torch.tensor([[7, 2, 3], [8, 5, 6]], device="cuda")
        attention = torch.ones_like(clean)
        positions = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool, device="cuda")
        counter = {"calls": 0}
        count_handle = model.register_forward_pre_hook(
            lambda module, inputs: counter.__setitem__("calls", counter["calls"] + 1)
        )
        scan = scan_hf_layers(model, clean, attention, corrupt, attention, positions, 13, 14)
        count_handle.remove()
        assert scan.shape == (3, 2) and scan.device.type == "cpu"
        assert counter["calls"] == 5

        for layer in range(3):
            one = hf_activation_patch_effect(
                model, clean, attention, corrupt, attention, layer, positions, 13, 14
            )
            assert torch.allclose(scan[layer], one["recovery"], atol=1e-5)
        return True

    raise ValueError(f"Unknown test: {test_name}")
