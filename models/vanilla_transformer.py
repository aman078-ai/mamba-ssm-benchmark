"""Vanilla decoder-only Transformer (GPT-style) implemented manually in PyTorch.

Key design choices:
- No nn.MultiheadAttention — implemented attention manually via QKV projection + softmax
- Causal mask for auto-regressive decoding
- No positional embeddings by default (can be added externally)
- Standard residual + norm pattern

Expected usage:
    model = VanillaTransformer(vocab_size=256, d_model=64, n_head=4, d_state=16, n_layer=2)
    x, _ = model(x)  # x: (B, T) -> (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (upper-triangular) mask of shape (1, 1, seq_len, seq_len)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf")).unsqueeze(0).unsqueeze(0)


class VanillaTransformer(nn.Module):
    """Decoder-onlyTransformer (GPT-style) implemented manually.

    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of model embeddings
        n_head: Number of attention heads
        d_state: Dimension of state per head (for compatibility with SSM comparison)
        n_layer: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_head: int = 4,
        d_state: int = 16,
        n_layer: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.d_state = d_state
        self.n_layer = n_layer

        # Token embeddings and position embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers (manual attention + FFN)
        self.layers = nn.ModuleList([
           TransformerLayer(d_model, n_head, d_state, dropout) for _ in range(n_layer)
        ])

        # Final lay norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape
        # Forward token embeddings: (B, T, d_model)
        h = self.embedding(x)

        # Apply each transformer layer
        for layer in self.layers:
            h = layer(h)  # (B, T, d_model)

        # Final norm and head
        h = self.ln_f(h)
        logits = self.head(h)  # (B, T, vocab_size)
        return logits


class TransformerLayer(nn.Module):
    """A single transformer layer with manual multi-head attention and FFN."""

    def __init__(self, d_model: int, n_head: int, d_state: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        # Layer norms
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        B, T, _ = x.shape
        # --- Pre-norm residual branch ---
        # Norm -> attention -> proj -> dropout -> add + norm -> FFN
        normed = self.ln_1(x)
        # Compute QKV projection: (B, T, 3*d_model) -> split into (B, T, 3, n_head, head_dim)
        qkv = self.qkv_proj(normed).reshape(B, -1, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_head, head_dim)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # attn_scores = q @ k^T / sqrt(d_head)  -> (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask (no need for padding mask in causal-only)
        # causal_mask shape: (1, 1, T, T) - broadcasts over B and n_head
        causal = causal_mask(T, x.device)
        attn_scores = attn_scores + causal

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        attn_out = attn_weights @ v  # (B, n_head, T, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)  # (B, T, d_model)

        # Projection + dropout
        attn_out = self.attn_out(attn_out)
        attn_out = self.dropout(attn_out)

        # Residual connection
        x = x + attn_out

        # FFN branch
        x = x + self.ffn(self.ln_2(x))

        return x