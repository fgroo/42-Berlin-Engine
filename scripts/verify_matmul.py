#!/usr/bin/env python3
"""Manually compute dot product for L2"""
from safetensors import safe_open

path = "Ministral-Stuff/consolidated.safetensors"

with safe_open(path, framework="pt") as f:
    w1_l2 = f.get_tensor("layers.2.feed_forward.w1.weight").float()
    
    # From C debug:
    # vec[0..4]=0.456963 -0.456990 0.457022 -0.457001 0.457013
    # row[0..4]=-0.016602 -0.067871 -0.075684 -0.109863 -0.059814
    
    row = w1_l2[0, :]
    
    # Create vec matching C's alternating pattern with sum=2.52
    import torch
    vec = torch.zeros(3072)
    for i in range(3072):
        if i % 2 == 0:
            vec[i] = 0.457
        else:
            vec[i] = -0.457
    
    # Compute partial sums
    partial_10 = (row[:10] * vec[:10]).sum().item()
    partial_100 = (row[:100] * vec[:100]).sum().item()
    partial_all = (row * vec).sum().item()
    
    print(f"Partial sum (10 elements): {partial_10:.6f}")
    print(f"Partial sum (100 elements): {partial_100:.6f}")
    print(f"Full dot product (3072): {partial_all:.6f}")
    print(f"C reports result: 5.122852")
    
    # What if the vec isn't alternating?
    vec_const_pos = torch.ones(3072) * 0.457
    print(f"\nWith all positive vec: {(row * vec_const_pos).sum().item():.6f}")
    
    # Check if vec_sum=2.52 can be achieved with alternating pattern
    vec_sum = vec.sum().item()
    print(f"Vec sum with alternating: {vec_sum:.6f}")
