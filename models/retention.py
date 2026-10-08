"""Retention (RetNet-style linear attention with exponential decay), written from scratch.

    o_t = sum_{s<=t} gamma^(t-s) (q_t . k_s) v_s          (one decay gamma per head)

Three equivalent forms:
  parallel   (Q K^T ⊙ D) V with D_ts = gamma^(t-s): O(T^2), like attention without softmax
  recurrent  S_t = gamma S_{t-1} + k_t^T v_t,  o_t = q_t S_t:  O(1) state (head_dim x head_dim)
  chunkwise  parallel inside chunks of size C, recurrent across chunks: O(T C) total.

It sits between the two other architectures: like attention it uses query-key matching, and
like an SSM it compresses the past into a fixed-size state with a (fixed, not selective) decay.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import apply_rope, rope_cos_sin


class Retention(nn.Module):
    def __init__(self, d_model, n_head, chunk_size=64):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim, self.chunk_size = n_head, d_model // n_head, chunk_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._residual_out = True
        self.norm_weight = nn.Parameter(torch.ones(d_model))
        gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(n_head, dtype=torch.float32))
        self.register_buffer("log_gamma", torch.log(gamma), persistent=False)

    def init_state(self, batch, max_len, device, dtype):
        return SimpleNamespace(S=torch.zeros(batch, self.n_head, self.head_dim, self.head_dim,
                                             device=device,
                                             dtype=torch.promote_types(dtype, torch.float32)))

    @staticmethod
    def state_nbytes(state):
        return state.S.numel() * state.S.element_size()

    def _qkv(self, x, pos):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = rope_cos_sin(torch.arange(pos, pos + T, device=x.device), self.head_dim, device=x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        ct = torch.promote_types(x.dtype, torch.float32)
        return q.to(ct), k.to(ct) * self.head_dim ** -0.5, v.to(ct)

    def _finish(self, o, x):
        B, H, T, hd = o.shape
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)          # per-head RMS norm
        o = o.transpose(1, 2).reshape(B, T, H * hd).to(x.dtype) * self.norm_weight
        return self.proj(o * F.silu(self.gate(x)))

    def forward_parallel(self, x, pos=0):
        q, k, v = self._qkv(x, pos)
        T = x.shape[1]
        idx = torch.arange(T, device=x.device)
        rel = (idx[:, None] - idx[None, :]).float()
        expo = (rel * self.log_gamma[:, None, None]).masked_fill(rel < 0, float("-inf"))
        o = ((q @ k.transpose(-2, -1)) * torch.exp(expo)) @ v
        return self._finish(o, x)

    def forward(self, x, state=None, pos=0):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, pos)
        lg = self.log_gamma.to(q.dtype)                                       # (H,)
        if T == 1 and state is not None:                                      # recurrent step
            S = torch.exp(lg)[None, :, None, None] * state.S + k.transpose(-2, -1) @ v
            state.S.copy_(S)
            return self._finish(q @ S, x)

        C = min(self.chunk_size, T)
        nc = -(-T // C)
        pad = nc * C - T
        if pad:   # left-pad with zeros: contributes nothing and keeps the final state exact
            q, k, v = (F.pad(t, (0, 0, pad, 0)) for t in (q, k, v))
        H, hd = self.n_head, self.head_dim
        q, k, v = (t.view(B, H, nc, C, hd) for t in (q, k, v))
        i = torch.arange(C, device=x.device).float()
        rel = i[:, None] - i[None, :]
        intra = torch.exp((rel * lg[:, None, None]).masked_fill(rel < 0, float("-inf")))   # (H, C, C)
        o = ((q @ k.transpose(-2, -1)) * intra[None, :, None]) @ v                         # in-chunk

        k_dec = k * torch.exp((C - 1 - i)[None, :, None] * lg[:, None, None])[None, :, None]
        kv = k_dec.transpose(-2, -1) @ v                                    # (B, H, nc, hd, hd)
        c = torch.arange(nc, device=x.device).float()
        crel = c[:, None] - 1 - c[None, :]                                  # chunk c sees c' < c
        W = torch.exp((crel * C * lg[:, None, None]).masked_fill(crel < 0, float("-inf")))  # (H, nc, nc)
        S_in = torch.einsum("hcd,bhdxy->bhcxy", W, kv)                      # state entering chunk
        if state is not None:   # undo the decay over the `pad` virtual zero tokens
            S0 = state.S * torch.exp(-pad * lg)[None, :, None, None]
            S_in = S_in + torch.exp(c * C * lg[:, None])[None, :, :, None, None] * S0.unsqueeze(2)
        q_dec = q * torch.exp((i + 1)[None, :, None] * lg[:, None, None])[None, :, None]
        o = o + q_dec @ S_in                                                # cross-chunk
        if state is not None:
            state.S.copy_(torch.exp(C * lg)[None, :, None, None] * S_in[:, :, -1] + kv[:, :, -1])
        o = o.reshape(B, H, nc * C, hd)[:, :, pad:]
        return self._finish(o, x)
