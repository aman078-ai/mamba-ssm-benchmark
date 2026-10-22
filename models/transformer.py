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


def alibi_slopes(n_head):
    """ALiBi head slopes (Press et al., 2022): a geometric sequence 2^(-8/H), 2^(-16/H), ..."""
    start = 2.0 ** (-8.0 / n_head)
    return torch.tensor([start ** (h + 1) for h in range(n_head)])


class Attention(nn.Module):
    """pos_encoding: 'rope' (rotary, default), 'none' (NoPE: causal mask only) or 'alibi' (linear distance penalty).
    impl: 'manual' (full T x S score matrix), 'chunked' (manual math, queries in blocks of query_chunk so memory is
    O(query_chunk * S); used to evaluate long contexts) or 'sdpa' (PyTorch's fused kernel, reference only)."""

    def __init__(self, d_model, n_head, dropout=0.0, impl="manual", rope_base=10000.0, pos_encoding="rope",
                 query_chunk=1024):
        super().__init__()
        assert d_model % n_head == 0
        assert pos_encoding in ("rope", "none", "alibi"), pos_encoding
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.impl, self.rope_base, self.pos_encoding, self.query_chunk = impl, rope_base, pos_encoding, query_chunk
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._residual_out = True
        self.dropout = dropout
        if pos_encoding == "alibi":
            self.register_buffer("slopes", alibi_slopes(n_head), persistent=False)

    def init_state(self, batch, max_len, device, dtype):
        return KVCache(batch, self.n_head, self.head_dim, max_len, device, dtype)

    def _scores_to_probs(self, scores, q_start, offset):
        """Causal mask (+ ALiBi) for queries q_start.. of this call; key j sits `offset` positions before query j."""
        Tq, S = scores.shape[-2:]
        qi = torch.arange(q_start, q_start + Tq, device=scores.device)[:, None] + offset
        dist = qi - torch.arange(S, device=scores.device)[None, :]          # query position - key position
        ct = torch.promote_types(scores.dtype, torch.float32)
        scores = scores.to(ct)
        if self.pos_encoding == "alibi":
            scores = scores - self.slopes.to(ct)[:, None, None] * dist.to(ct)
        scores = scores.masked_fill(dist < 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(self, x, state=None, pos=0):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.pos_encoding == "rope":
            cos, sin = rope_cos_sin(torch.arange(pos, pos + T, device=x.device), self.head_dim,
                                    self.rope_base, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if state is not None:                       # write this step's K/V, attend to the prefix
            state.k[:, :, pos:pos + T] = k.to(state.k.dtype)
            state.v[:, :, pos:pos + T] = v.to(state.v.dtype)
            k, v = state.k[:, :, :pos + T].to(q.dtype), state.v[:, :, :pos + T].to(q.dtype)
        S = k.shape[2]
        offset = S - T                              # absolute position of query 0 minus that of key 0
        drop = self.dropout if self.training else 0.0
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.impl == "manual":
            att = self._scores_to_probs((q @ k.transpose(-2, -1)) * scale, 0, offset).to(q.dtype)
            y = F.dropout(att, drop, self.training) @ v
        elif self.impl == "chunked":
            outs = []
            for q0 in range(0, T, self.query_chunk):
                qb = q[:, :, q0:q0 + self.query_chunk]
                att = self._scores_to_probs((qb @ k.transpose(-2, -1)) * scale, q0, offset).to(q.dtype)
                outs.append(F.dropout(att, drop, self.training) @ v)
            y = torch.cat(outs, dim=2)
        elif self.impl == "sdpa":   # PyTorch fused kernel, used only as a reference point
            # (contiguous: torch 2.5 CPU SDPA returns NaN for strided float64 inputs)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            if self.pos_encoding == "alibi":
                dist = (torch.arange(T, device=x.device)[:, None] + offset) - torch.arange(S, device=x.device)[None, :]
                bias = (-self.slopes[:, None, None] * dist).masked_fill(dist < 0, float("-inf")).to(q.dtype)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=drop)
            elif T > 1 and S != T:
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril_(offset)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, dropout_p=drop)
        else:
            raise ValueError(self.impl)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))
