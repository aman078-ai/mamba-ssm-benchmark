"""Mamba (S6) mixer written from scratch.

    u ──in_proj──► x ──causal depthwise conv──► SiLU ──► selective SSM ──► (+ D·x) ──► ⊙ ──► out_proj
          └──────► z ─────────────────────────────────────────── SiLU ────────────────┘

Selection: Δ (step size), B and C are linear functions of the current token, so the model
decides per token how much to remember (large Δ: reset towards the input; small Δ: keep the
state).  A is a learned, input-independent diagonal matrix A = -exp(A_log), initialized to
S4D-real (A_n = -(n+1)).  `selective=False` freezes Δ, B, C to learned constants, which turns
the layer into a linear time-invariant S4D-style SSM: the ablation from the Mamba paper.

Inference keeps a fixed-size state per layer: the last (d_conv - 1) conv inputs and the
(d_inner, d_state) SSM state.  Its size does not depend on how many tokens have been seen.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scan import discretize, selective_scan


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_rank=None, selective=True,
                 discretization="zoh", scan="chunked", d_block=None, chunk_size=None,
                 dt_min=1e-3, dt_max=0.1):
        super().__init__()
        self.d_inner = d_inner = expand * d_model
        self.d_state, self.d_conv = d_state, d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.selective, self.discretization = selective, discretization
        self.scan, self.d_block, self.chunk_size = scan, d_block, chunk_size

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, bias=True)

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        if selective:
            self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
            self.dt_proj._no_reinit = True
            std = self.dt_rank ** -0.5
            with torch.no_grad():
                nn.init.uniform_(self.dt_proj.weight, -std, std)
                self.dt_proj.bias.copy_(inv_softplus_dt)
        else:
            self.dt_bias = nn.Parameter(inv_softplus_dt)
            self.B = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
            self.C = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.out_proj._residual_out = True

    # -- helpers ---------------------------------------------------------------------------
    def ssm_params(self, x):
        """x: (B, L, d_inner) -> Δ (B, L, d_inner), A (d_inner, N), B, C (B, L, N)."""
        Bsz, L, _ = x.shape
        ct = torch.promote_types(x.dtype, torch.float32)
        if self.selective:
            dt, Bm, Cm = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(dt).to(ct))
        else:
            delta = F.softplus(self.dt_bias).expand(Bsz, L, -1)
            Bm = self.B.expand(Bsz, L, -1)
            Cm = self.C.expand(Bsz, L, -1)
        return delta, -torch.exp(self.A_log.to(ct)), Bm, Cm

    def init_state(self, batch, max_len, device, dtype):
        return SimpleNamespace(
            conv=torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            ssm=torch.zeros(batch, self.d_inner, self.d_state, device=device,
                            dtype=torch.promote_types(dtype, torch.float32)),
        )

    @staticmethod
    def state_nbytes(state):
        return state.conv.numel() * state.conv.element_size() + state.ssm.numel() * state.ssm.element_size()

    # -- forward ---------------------------------------------------------------------------
    def forward(self, u, state=None, pos=0, return_delta=False):
        Bsz, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = x.transpose(1, 2)                                            # (B, D, L)
        if state is None:
            xc = F.pad(xc, (self.d_conv - 1, 0))
        else:
            xc = torch.cat([state.conv.to(xc.dtype), xc], dim=-1)
            state.conv.copy_(xc[..., -(self.d_conv - 1):])
        x = F.silu(F.conv1d(xc, self.conv.weight, self.conv.bias, groups=self.d_inner)).transpose(1, 2)

        delta, A, Bm, Cm = self.ssm_params(x)
        if getattr(self, "record_delta", False):        # used by analysis/delta_analysis.py
            self.last_delta = delta.detach()
        h0 = None if state is None else state.ssm
        if L == 1 and state is not None:                                  # recurrent decode step
            ct = A.dtype
            a, b = discretize(delta.to(ct), A, Bm.to(ct), x.to(ct), self.discretization)
            h = a[:, 0] * h0 + b[:, 0]
            y = torch.einsum("bdn,bn->bd", h, Cm[:, 0].to(ct)).unsqueeze(1).to(x.dtype)
            last = h
        else:
            y, last = selective_scan(x, delta, A, Bm, Cm, h0=h0, method=self.discretization,
                                     scan=self.scan, d_block=self.d_block, chunk_size=self.chunk_size)
        if state is not None:
            state.ssm.copy_(last)
        y = (y + x * self.D.to(x.dtype)) * F.silu(z)
        out = self.out_proj(y)
        return (out, delta) if return_delta else out
