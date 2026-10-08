"""Causal multi-head attention written by hand, with RoPE and an explicit KV cache.

Full-sequence forward materializes the (B, H, T, T) score matrix: O(T^2) time and memory.
Incremental decoding appends this step's K, V to a preallocated cache and attends over
all cached positions: O(T) time per token and an O(T) cache that grows with context.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import apply_rope, rope_cos_sin


class KVCache:
    """Preallocated per-layer key/value buffers of shape (B, H, max_len, head_dim)."""

    def __init__(self, batch, n_head, head_dim, max_len, device, dtype):
        self.k = torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def nbytes(self, length=None):
        per_pos = 2 * self.k[:, :, :1].numel() * self.k.element_size()
        return per_pos * (self.k.shape[2] if length is None else length)


class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, impl="manual", rope_base=10000.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.impl = impl
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._residual_out = True
        self.dropout = dropout

    def init_state(self, batch, max_len, device, dtype):
        return KVCache(batch, self.n_head, self.head_dim, max_len, device, dtype)

    def forward(self, x, state=None, pos=0):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = rope_cos_sin(torch.arange(pos, pos + T, device=x.device), self.head_dim,
                                self.rope_base, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if state is not None:                       # write this step's K/V, attend to the prefix
            state.k[:, :, pos:pos + T] = k.to(state.k.dtype)
            state.v[:, :, pos:pos + T] = v.to(state.v.dtype)
            k, v = state.k[:, :, :pos + T].to(q.dtype), state.v[:, :, :pos + T].to(q.dtype)
        S = k.shape[2]
        drop = self.dropout if self.training else 0.0
        if self.impl == "manual":
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))   # (B, H, T, S)
            if T > 1:   # query i sits at absolute position pos+i and may see keys <= pos+i
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device).triu_(pos + 1)
                scores = scores.masked_fill(mask, float("-inf"))
            att = F.softmax(scores.to(torch.promote_types(q.dtype, torch.float32)), dim=-1).to(q.dtype)
            att = F.dropout(att, drop, self.training)
            y = att @ v
        elif self.impl == "sdpa":   # PyTorch fused kernel, used only as a reference point
            # (contiguous: torch 2.5 CPU SDPA returns NaN for strided float64 inputs)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            if T > 1 and S != T:
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril_(pos)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, dropout_p=drop)
        else:
            raise ValueError(self.impl)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))
