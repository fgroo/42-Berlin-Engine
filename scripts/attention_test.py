#!/usr/bin/env python3
"""
Check attention scores for "A B C D" tokens using manual forward pass.
"""
import torch
from safetensors import safe_open
import json
import math

print("="*60)
print("ATTENTION SCORE CHECK for 'A B C D' tokens")
print("="*60)

# Load config
with open("Ministral-Stuff/config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)
dim = text_cfg["hidden_size"]  # 3072
n_heads = text_cfg["num_attention_heads"]  # 32
n_kv_heads = text_cfg["num_key_value_heads"]  # 8
head_dim = text_cfg["head_dim"]  # 128
rope_params = text_cfg["rope_parameters"]
rope_theta = rope_params["rope_theta"]
factor = rope_params["factor"]
beta_slow = rope_params["beta_slow"]
beta_fast = rope_params["beta_fast"]
mscale = math.sqrt(0.1 * math.log(factor) + 1.0)

print(f"RoPE: theta={rope_theta}, factor={factor}, mscale={mscale:.4f}")

# Load weights
f = safe_open("Ministral-Stuff/consolidated.safetensors", framework="pt")
embeddings = f.get_tensor("tok_embeddings.weight").float()
wq = f.get_tensor("layers.0.attention.wq.weight").float()
wk = f.get_tensor("layers.0.attention.wk.weight").float()
attention_norm = f.get_tensor("layers.0.attention_norm.weight").float()

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

def apply_rope_mscale(x, pos, head_dim, mscale):
    """Apply RoPE with YaRN + mscale"""
    x = x.clone()
    half = head_dim // 2
    n_vecs = len(x) // head_dim
    
    for h in range(n_vecs):
        offset = h * head_dim
        for j in range(half):
            theta = get_yarn_theta(j * 2, rope_theta, head_dim, factor, beta_slow, beta_fast)
            angle = pos * theta
            
            v0 = x[offset + j].item() * mscale
            v1 = x[offset + j + half].item() * mscale
            
            x[offset + j]        = v0 * math.cos(angle) - v1 * math.sin(angle)
            x[offset + j + half] = v0 * math.sin(angle) + v1 * math.cos(angle)
    
    return x

# Token IDs: BOS=1, A=32 (likely), B=33, C=34, D=35
# Actually check what these tokens encode to
# For simplicity: use token IDs 1, 2, 3, 4, 5
token_ids = [1, 2, 3, 4, 5]  # BOS + dummy tokens

print(f"\nProcessing {len(token_ids)} tokens...")

# Build K cache for all positions
K_cache = []
for pos, tok_id in enumerate(token_ids):
    x = embeddings[tok_id].clone()
    x = rmsnorm(x, attention_norm)
    k = wk @ x  # [n_kv_heads * head_dim]
    k = apply_rope_mscale(k, pos, head_dim, mscale)
    K_cache.append(k)

# Compute Q for last position
last_pos = len(token_ids) - 1
last_tok = token_ids[-1]
x = embeddings[last_tok].clone()
x = rmsnorm(x, attention_norm)
q = wq @ x  # [n_heads * head_dim]
q = apply_rope_mscale(q, last_pos, head_dim, mscale)

print(f"\nQ computed for pos={last_pos}")

# Compute raw attention scores for head 0
print("\n" + "="*60)
print("RAW ATTENTION SCORES (L0H0) for last token:")
print("="*60)

q_head0 = q[0:head_dim]
scale = 1.0 / math.sqrt(head_dim)

for ki in range(len(K_cache)):
    kv_h = 0  # Head 0 uses kv_head 0
    k_head0 = K_cache[ki][kv_h * head_dim : (kv_h+1) * head_dim]
    raw_score = torch.dot(q_head0, k_head0).item() * scale
    print(f"  pos={ki}: raw_score = {raw_score:.2f}")

# Compute softmax
import numpy as np
scores = []
for ki in range(len(K_cache)):
    kv_h = 0
    k_head0 = K_cache[ki][kv_h * head_dim : (kv_h+1) * head_dim]
    raw_score = torch.dot(q_head0, k_head0).item() * scale
    scores.append(raw_score)

scores = np.array(scores)
exp_scores = np.exp(scores - np.max(scores))
probs = exp_scores / np.sum(exp_scores)

print("\nATTENTION WEIGHTS after softmax:")
for i, p in enumerate(probs):
    print(f"  pos={i}: {p*100:.1f}%")
