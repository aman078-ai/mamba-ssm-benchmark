"""A byte-level language model whose layers can be any mix of attention, Mamba and retention.

Layer pattern letters:
    A  pre-norm residual attention block + MLP block   (GPT layer, ~12 d^2 params)
    M  pre-norm residual Mamba block                    (~6.6 d^2 params, no separate MLP)
    R  pre-norm residual retention block + MLP block

    transformer = "AAAAAA",  mamba = "MMMMMMMMMMM",  retnet = "RRRRRR",
    hybrid      = "MMMMMAMMMM" (Jamba-style: mostly Mamba, one attention layer)

Interfaces:
    logits = model(idx)                                    # training / scoring, (B, T, V)
    states = model.init_state(B, max_len, device)          # KV caches / SSM states
    logits = model(prompt, states=states, pos=0)           # prefill
    logits = model(next_tok[:, None], states, pos=t)       # one decode step
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import MLP, RMSNorm
from models.mamba import Mamba
from models.retention import Retention
from models.transformer import Attention, KVCache


@dataclass
class LMConfig:
    layers: str = "AAAAAA"
    vocab_size: int = 256
    d_model: int = 256
    n_head: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_impl: str = "manual"
    pos_encoding: str = "rope"          # attention position encoding: rope | none | alibi
    attn_query_chunk: int = 1024        # query block size for attn_impl="chunked"
    # Mamba
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    selective: bool = True
    a_mode: str = "s4d"                 # Mamba decay parameterization: s4d (per state) | scalar (shared, Mamba-2-style)
    discretization: str = "zoh"
    scan: str = "auto"
    d_block: int = 0
    chunk_size: int = 0
    # Retention
    retention_chunk: int = 64

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, kind, cfg: LMConfig):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        if kind == "A":
            self.mixer = Attention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.attn_impl,
                                   pos_encoding=cfg.pos_encoding, query_chunk=cfg.attn_query_chunk)
        elif kind == "M":
            self.mixer = Mamba(cfg.d_model, cfg.d_state, cfg.expand, cfg.d_conv,
                               selective=cfg.selective, discretization=cfg.discretization,
                               scan=cfg.scan, d_block=cfg.d_block or None,
                               chunk_size=cfg.chunk_size or None, a_mode=cfg.a_mode)
        elif kind == "R":
            self.mixer = Retention(cfg.d_model, cfg.n_head, cfg.retention_chunk)
        else:
            raise ValueError(kind)
        self.mlp = None
        if kind in "AR":
            self.norm2 = RMSNorm(cfg.d_model)
            self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, state=None, pos=0):
        x = x + self.drop(self.mixer(self.norm1(x), state, pos))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(k, cfg) for k in cfg.layers])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                       # tied embeddings
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear) and not getattr(m, "_no_reinit", False):
            std = 0.02 / math.sqrt(2 * len(self.cfg.layers)) if getattr(m, "_residual_out", False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def init_state(self, batch, max_len, device, dtype=torch.float32):
        return [b.mixer.init_state(batch, max_len, device, dtype) for b in self.blocks]

    @staticmethod
    def state_nbytes(states, length=None):
        total = 0
        for s in states:
            total += s.nbytes(length) if isinstance(s, KVCache) else sum(
                t.numel() * t.element_size() for t in vars(s).values())
        return total

    def forward(self, idx, states=None, pos=0):
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, None if states is None else states[i], pos)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, prompt, n_new, temperature=1.0, top_k=None, max_len=None):
        """Autoregressive sampling with caches/states.  prompt: (B, T) byte ids."""
        B, T = prompt.shape
        states = self.init_state(B, max_len or T + n_new, prompt.device)
        logits = self(prompt, states, 0)[:, -1]
        out = [prompt]
        for t in range(n_new):
            logits = logits.float() / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            out.append(nxt)
            if t < n_new - 1:
                logits = self(nxt, states, T + t)[:, -1]
        return torch.cat(out, 1)


# Matched-size configurations used throughout the experiments (d_model = 256, ~4.7M params).
PRESETS = {
    "transformer": dict(layers="A" * 6),
    "mamba": dict(layers="M" * 11),
    "retnet": dict(layers="R" * 6, mlp_ratio=3.5),
    "hybrid": dict(layers="MMMMMAMMMM"),
}


def build(name="transformer", **overrides):
    kw = dict(PRESETS.get(name, {}))
    kw.update(overrides)
    return LM(LMConfig(**kw))
