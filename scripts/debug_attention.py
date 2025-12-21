#!/usr/bin/env python3
"""
Manual attention WITH YaRN RoPE AND mscale.
"""
import torch
from safetensors import safe_open
import json
import math

print("="*60)
print("MINISTRAL ATTENTION (WITH YaRN + mscale)")
print("="*60)

# Load config
with open("Ministral-Stuff/config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)
head_dim = text_cfg["head_dim"]
n_kv_heads = text_cfg["num_key_value_heads"]
dim = text_cfg["hidden_size"]
rope_params = text_cfg["rope_parameters"]
rope_theta = rope_params["rope_theta"]
factor = rope_params["factor"]
beta_slow = rope_params["beta_slow"]
beta_fast = rope_params["beta_fast"]

# Compute mscale (same as C)
mscale = math.sqrt(0.1 * math.log(factor) + 1.0)
print(f"mscale = {mscale:.6f}")

# Load weights
f = safe_open("Ministral-Stuff/consolidated.safetensors", framework="pt")
embeddings = f.get_tensor("tok_embeddings.weight").float()
wq = f.get_tensor("layers.0.attention.wq.weight").float()
wk = f.get_tensor("layers.0.attention.wk.weight").float()
attention_norm = f.get_tensor("layers.0.attention_norm.weight").float()

tokens = [1, 17, 2438, 2077, 5966]

def rmsnorm(x, w, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2) + eps)
    return (x / rms) * w

def get_yarn_theta(j, theta_base, head_dim, factor, beta_slow, beta_fast):
    freq_idx = j / head_dim
    theta = 1.0 / (theta_base ** freq_idx)
    if factor <= 1.0:
        return theta
    if freq_idx < beta_slow / head_dim:
        return theta
    elif freq_idx > beta_fast / head_dim:
        return 1.0 / ((theta_base * factor) ** freq_idx)
    else:
        start = beta_slow / head_dim
        end = beta_fast / head_dim
        alpha = (freq_idx - start) / (end - start)
        theta_scaled = 1.0 / ((theta_base * factor) ** freq_idx)
        return (1.0 - alpha) * theta + alpha * theta_scaled

def apply_rope_yarn_mscale(x, pos, theta_base, head_dim, factor, beta_slow, beta_fast, mscale):
    """Apply RoPE with YaRN + mscale (matches C exactly)"""
    x = x.clone()
    half = head_dim // 2
    n_vecs = len(x) // head_dim
    
    for h in range(n_vecs):
        offset = h * head_dim
        for j in range(half):
            theta = get_yarn_theta(j * 2, theta_base, head_dim, factor, beta_slow, beta_fast)
            angle = pos * theta
            
            v0 = x[offset + j].item() * mscale
            v1 = x[offset + j + half].item() * mscale
            
            x[offset + j]        = v0 * math.cos(angle) - v1 * math.sin(angle)
            x[offset + j + half] = v0 * math.sin(angle) + v1 * math.cos(angle)
    
    return x

print("\n" + "="*60)
print("K CACHE WITH YaRN + mscale")
print("="*60)

K_cache = []
for pos, tok_id in enumerate(tokens[:4]):
    x = embeddings[tok_id].clone()
    x = rmsnorm(x, attention_norm)
    k = wk @ x
    k = apply_rope_yarn_mscale(k, pos, rope_theta, head_dim, factor, beta_slow, beta_fast, mscale)
    K_cache.append(k)
    if pos < 3:
        print(f"Token {pos} (ID={tok_id}): K[0:4] = {k[0:4].tolist()}")

print("\nC K[0][0:4] = 0.1104 0.1104 -0.3320 -0.3848")
print("C K[1][0:4] = 0.6914 -0.8008 -0.0610 -0.3066")

print("\n" + "="*60)
print("RAW ATTENTION SCORES (pos=3, head=0)")
print("="*60)

pos = 3
tok_id = tokens[pos]
x = embeddings[tok_id].clone()
x = rmsnorm(x, attention_norm)
q = wq @ x
q = apply_rope_yarn_mscale(q, pos, rope_theta, head_dim, factor, beta_slow, beta_fast, mscale)

q_head0 = q[0:head_dim]
scale = 1.0 / math.sqrt(head_dim)

print(f"\nPython (YaRN + mscale):")
for ki, k in enumerate(K_cache[:pos+1]):
    kv_h = 0
    k_head0 = k[kv_h * head_dim : (kv_h+1) * head_dim]
    score = torch.dot(q_head0, k_head0).item() * scale
    print(f"  Q[pos={pos}] · K[pos={ki}] = {score:.2f}")

print("\nC OUTPUT:")
print("  pos=3: 8.09 0.74 -2.84 0.00")
