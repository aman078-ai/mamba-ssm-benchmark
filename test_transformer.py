"""Quick test script for vanilla transformer."""

import sys
sys.path.insert(0, '.')

from models.vanilla_transformer import VanillaTransformer, TransformerLayer, causal_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.enwik8 import Enwik8
import torch.optim as optim


def test_basic_forward():
    """Test 1: Basic model instantiation and forward pass."""
    print('=== Test 1: Basic forward pass ===')
    model = VanillaTransformer(vocab_size=256, d_model=64, n_head=4, d_state=16, n_layer=2)
    x = torch.randint(0, 256, (2, 32))  # (B, T)
    logits = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Logits shape: {logits.shape}')
    assert logits.shape == (2, 32, 256), f'Expected (2,32,256) got {logits.shape}'
    print('PASS')
    return True


def test_single_layer():
    """Test 2: Single transformer layer."""
    print('\n=== Test 2: Single transformer layer ===')
    layer = TransformerLayer(d_model=64, n_head=4, d_state=16, dropout=0.0)
    x = torch.randn(2, 32, 64)
    out = layer(x)
    print(f'Input: {x.shape} -> Output: {out.shape}')
    assert out.shape == (2, 32, 64)
    print('PASS')
    return True


def test_causal_mask():
    """Test 3: Causal mask correctness."""
    print('\n=== Test 3: Causal mask ===')
    device = torch.device('cpu')
    mask = causal_mask(4, device)
    print(f'Causal mask shape: {mask.shape}')
    print(mask)
    assert mask.shape == (1, 1, 4, 4), f'Expected (1,1,4,4) got {mask.shape}'
    assert mask[0, 0, 0, 1].item() == float('-inf'), 'Mask should have -inf above diagonal'
    assert mask[0, 0, 0, 0].item() == 0.0, 'Mask should be 0 on diagonal'
    # Test that it actually zeroes out future positions
    B, n_head, T = 2, 4, 4
    attn_scores = torch.randn(B, n_head, T, T)
    masked_scores = attn_scores + mask
    # Check that diagonal and lower triangle are preserved (not -inf)
    # Upper triangle should be -inf
    assert masked_scores[0, 0, 0, 1].item() == float('-inf')
    print('PASS')
    return True


def test_toy_training():
    """Test 4: Toy training on enwik8."""
    print('\n=== Test 4: Toy training on enwik8 ===')
    data = Enwik8(root='data/enwik8', train=True, seqlen=64, batch_size=2)
    model = VanillaTransformer(vocab_size=256, d_model=64, n_head=4, d_state=16, n_layer=2)
    opt = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for step in range(5):
        x, y = data.get()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'step {step}: loss = {loss.item():.4f}')
    print('PASS - toy training works')
    return True


if __name__ == '__main__':
    test_basic_forward()
    test_single_layer()
    test_causal_mask()
    test_toy_training()
    print('\n=== All tests passed! ===')