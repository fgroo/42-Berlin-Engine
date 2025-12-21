#!/usr/bin/env python3
"""Check FFN weights for Layer 0, 1, 2 - compare variance and range"""
from safetensors import safe_open

path = "Ministral-Stuff/consolidated.safetensors"

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    
    for layer in [0, 1, 2]:
        print(f"\n=== Layer {layer} ===")
        
        for w_name in ["w1", "w2", "w3"]:
            key = f"layers.{layer}.feed_forward.{w_name}.weight"
            if key in keys:
                w = f.get_tensor(key).float()  # Convert BF16 to F32
                print(f"{w_name}: shape={w.shape}, dtype=bf16")
                print(f"    mean={w.mean().item():.6f}, std={w.std().item():.6f}")
                print(f"    min={w.min().item():.6f}, max={w.max().item():.6f}")
                print(f"    first 5 values: {w.flatten()[:5].tolist()}")
            else:
                print(f"{w_name}: NOT FOUND ({key})")
