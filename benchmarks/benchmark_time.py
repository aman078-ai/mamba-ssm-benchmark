"""Benchmark script: wall-clock time vs sequence length.

Sweeps sequence length and measures forward-pass time for both
Transformer and SSM architectures, demonstrating O(n^2) vs O(n) scaling.
Also measures peak GPU memory usage.
"""

import argparse
import time
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.vanilla_transformer import VanillaTransformer
from models.ssm_block import SSMBlock
from data.enwik8 import Enwik8


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward_time():
    """Benchmark forward-pass time vs sequence length."""
    parser = argparse.ArgumentParser(description='Benchmark forward time')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--seqlens', type=str, default='64,128,256,512,1024', help='Comma-separated sequence lengths')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    seqlens = [int(x) for x in args.seqlens.split(',')]
    device = torch.device(args.device)

    results = {
        'forward_time_sec': {},
        'memory_peak_mb': {},
        'params': {}
    }

    # Create models
    transformer = VanillaTransformer(
        vocab_size=256,
        d_model=args.d_model,
        n_head=args.n_head,
        d_state=args.d_state,
        n_layer=args.n_layer,
    ).to(device)

    ssm = SSMBlock(d_model=args.d_model, d_state=args.d_state, dropout=0.0, mode='naive').to(device)

    transformer_params = count_parameters(transformer)
    ssm_params = count_parameters(ssm)

    results['params']['transformer'] = transformer_params
    results['params']['ssm'] = ssm_params

    print(f'=== Forward-Time Benchmark ===')
    print(f'Transformer: d_model={args.d_model}, n_layer={args.n_layer}, n_head={args.n_head}')
    print(f'SSM: d_model={args.d_model}, d_state={args.d_state}')
    print(f'Batch size: {args.batch_size}')
    print(f'Sequence lengths: {seqlens}')
    print(f'Device: {device}')
    print()

    for seq_len in seqlens:
        print(f'--- Sequence length: {seq_len} ---')

        T = seq_len
        B = args.batch_size

        # --- Transformer benchmark ---
        # Create random token indices (B, T)
        tokens = torch.randint(0, 256, (B, T), device=device)

        # Transformer forward
        t_start = time.time()
        with torch.no_grad():
            t_logits = transformer(tokens)  # (B, T, vocab_size)
        t_end = time.time()
        t_time = t_end - t_start

        # Measure memory on CUDA if available
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            _ = transformer(tokens)
            t_mem_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
        else:
            t_mem_peak = 0.0  # Approximate CPU memory

        results['forward_time_sec'][f'transformer_{seq_len}'] = t_time
        results['memory_peak_mb'][f'transformer_{seq_len}'] = t_mem_peak

        print(f'  Transformer: time={t_time*1000:.2f}ms, mem_peak={t_mem_peak:.1f}MB')

        # --- SSM benchmark ---
        # SSM expects (B, T, d_model) - we need to embed tokens first
        # Create embeddings for the token indices
        s_tokens = torch.randint(0, 256, (B, T), device=device)
        s_emb = torch.nn.Embedding(256, args.d_model).to(device)(s_tokens)  # (B, T, d_model)

        s_start = time.time()
        with torch.no_grad():
            s_out = ssm(s_emb)  # (B, T, d_model)
        s_end = time.time()
        s_time = s_end - s_start

        # Measure memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            _ = ssm(s_emb)
            s_mem_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
        else:
            s_mem_peak = 0.0

        results['forward_time_sec'][f'ssm_{seq_len}'] = s_time
        results['memory_peak_mb'][f'ssm_{seq_len}'] = s_mem_peak

        print(f'  SSM: time={s_time*1000:.2f}ms, mem_peak={s_mem_peak:.1f}MB')

        # Clean up intermediate embedding
        del s_emb

        print()

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_forward_time.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'=== Results saved to results/benchmark_forward_time.json ===')
    print(f'=== Benchmark complete! ===')


if __name__ == '__main__':
    benchmark_forward_time()