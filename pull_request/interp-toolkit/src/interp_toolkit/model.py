"""A minimal decoder-only transformer with named hook points for interpretability work.

This is intentionally small (toy-sized) so experiments run instantly on CPU.
Hook points let callers cache or patch intermediate activations during a
forward pass, in the style of activation-patching / causal-tracing experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from interp_toolkit.hooks import HookPoint


@dataclass
class ModelConfig:
    n_layers: int = 2
    d_model: int = 32
    n_heads: int = 4
    d_vocab: int = 64
    n_ctx: int = 16

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        cfg = self.cfg
        qkv = self.qkv(x).reshape(b, t, 3, cfg.n_heads, cfg.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: [b, t, n_heads, d_head]
        q = q.transpose(1, 2)  # [b, n_heads, t, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (cfg.d_head**0.5)
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        pattern = F.softmax(scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        z = torch.einsum("bhqk,bhkd->bhqd", pattern, v)  # [b, n_heads, t, d_head]
        z = self.hook_z(z)
        z = z.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(z)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc_out = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.hook_pre(self.fc_in(x))
        post = self.hook_post(F.gelu(pre))
        return self.fc_out(post)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        x = x + self.attn(self.ln1(x))
        x = self.hook_resid_mid(x)
        x = x + self.mlp(self.ln2(x))
        x = self.hook_resid_post(x)
        return x


class MiniTransformer(nn.Module):
    """Toy decoder-only transformer. Hook names follow `blocks.{i}.<name>`."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

    def named_hook_points(self) -> dict[str, HookPoint]:
        points: dict[str, HookPoint] = {}
        for i, block in enumerate(self.blocks):
            points[f"blocks.{i}.hook_resid_pre"] = block.hook_resid_pre
            points[f"blocks.{i}.hook_resid_mid"] = block.hook_resid_mid
            points[f"blocks.{i}.hook_resid_post"] = block.hook_resid_post
            points[f"blocks.{i}.attn.hook_pattern"] = block.attn.hook_pattern
            points[f"blocks.{i}.attn.hook_z"] = block.attn.hook_z
            points[f"blocks.{i}.mlp.hook_pre"] = block.mlp.hook_pre
            points[f"blocks.{i}.mlp.hook_post"] = block.mlp.hook_post
        return points

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.unembed(x)
