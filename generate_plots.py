#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
with open('results/benchmark_forward_time.json', 'r') as f:
    ft = json.load(f)

with open('results/benchmark_kv_cache.json', 'r') as f:
    kv = json.load(f)

# Sequence lengths from both benchmarks
seqlens_ft = [64, 128, 256, 512]
all_seqlens = sorted(set(seqlens_ft + [1024]))

# ===== Plot 1: Forward time vs sequence length (linear scale) =====
plt.figure(figsize=(10, 6))
valid_seqlens = [s for s in all_seqlens if f'transformer_{s}' in ft and f'ssm_{s}' in ft]
t_times = [ft['forward_time_sec'][f'transformer_{s}'] * 1000 for s in valid_seqlens]  # ms
s_times = [ft['forward_time_sec'][f'ssm_{s}'] * 1000 for s in valid_seqlens]

plt.plot(valid_seqlens, t_times, 'o-', label='Transformer (QKV attention)', linewidth=2, markersize=6, color='blue')
plt.plot(valid_seqlens, s_times, 's-', label='SSM (selective scan)', linewidth=2, markersize=6, color='red')
plt.xlabel('Sequence length')
plt.ylabel('Forward pass time (ms)')
plt.title('Forward Pass Speed: Transformer vs SSM', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/time_vs_seqlen.png', dpi=150)
print('Saved: plots/time_vs_seqlen.png')

# ===== Plot 2: KV-cache growth vs sequence length =====
plt.figure(figsize=(10, 6))
# Transformer KV-cache values from benchmark
kv_t_vals = kv['kv_cache_mb_transformer']
# Map to the valid seqlens - use closest match
kv_t_interp = []
for s in valid_seqlens:
    closest = min(kv['seq_lens'], key=lambda x: abs(x - s))
    kv_t_interp.append(kv['kv_cache_mb_transformer'][kv['seq_lens'].index(closest)])

plt.plot(valid_seqlens, kv_t_interp, 'o-', label='Transformer KV-cache', linewidth=2, markersize=6, color='blue')
# SSM state - constant
ssm_state = kv['ssm_state_mb'][0] if kv['ssm_state_mb'] else 0.0001
plt.axhline(y=ssm_state, linestyle='--', label='SSM state (constant)', linewidth=2, markersize=6, color='red')
plt.xlabel('Sequence length')
plt.ylabel('Memory (MB)')
plt.title('Inference Memory: KV-cache vs SSM State', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/kv_cache_vs_ssm_state.png', dpi=150)
print('Saved: plots/kv_cache_vs_ssm_state.png')

# ===== Plot 3: Parameter count comparison =====
plt.figure(figsize=(8, 6))
labels = ['Transformer', 'SSM']
params = [ft['params']['transformer'], ft['params']['ssm']]
colors = ['skyblue', 'lightcoral']
bars = plt.bar(labels, params, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Number of trainable parameters')
plt.title(f'Model Size Comparison\n(Transformer: {ft["params"]["transformer"]:,} vs SSM: {ft["params"]["ssm"]:,})', fontsize=13)
for bar, param in zip(bars, params):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{param:,}', ha='center', va='bottom', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/model_size_comparison.png', dpi=150)
print('Saved: plots/model_size_comparison.png')

# ===== Plot 4: Time ratio (Transformer/SSM) vs sequence length =====
plt.figure(figsize=(10, 6))
if valid_seqlens:
    ratios = [t/s if s > 0 else float('inf') for t, s in zip(t_times, s_times)]
    plt.plot(valid_seqlens, ratios, 'o-', color='green', linewidth=2, markersize=6)
    plt.xlabel('Sequence length')
    plt.ylabel('Transformer time / SSM time')
    plt.title('Relative Speed: Transformer/SSM Ratio', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/speed_ratio.png', dpi=150)
    print('Saved: plots/speed_ratio.png')

print('All plots generated successfully!')