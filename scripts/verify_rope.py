#!/usr/bin/env python3
"""
Verify RoPE theta computation with YaRN scaling.
"""
import math

theta_base = 1000000.0
head_dim = 128
factor = 16.0
beta_slow = 1.0
beta_fast = 32.0
pos = 3

def get_yarn_theta(j, theta_base, head_dim, factor, beta_slow, beta_fast):
    """Replicate C's get_yarn_theta exactly"""
    freq_idx = j / head_dim
    
    # Standard RoPE theta
    theta = 1.0 / (theta_base ** freq_idx)
    
    # If no YaRN (factor <= 1.0), return standard
    if factor <= 1.0:
        return theta
    
    # YaRN Interpolation
    if freq_idx < beta_slow / head_dim:  # High freq - no scaling
        return theta
    elif freq_idx > beta_fast / head_dim:  # Low freq - full scaling
        return 1.0 / ((theta_base * factor) ** freq_idx)
    else:
        # Ramp interpolation
        start = beta_slow / head_dim
        end = beta_fast / head_dim
        alpha = (freq_idx - start) / (end - start)
        
        theta_scaled = 1.0 / ((theta_base * factor) ** freq_idx)
        return (1.0 - alpha) * theta + alpha * theta_scaled

print("PYTHON with YaRN (matching C logic):")
print(f"Config: theta_base={theta_base}, factor={factor}, beta_slow={beta_slow}, beta_fast={beta_fast}")
print()

for j in [0, 1, 2]:  # dim 0, 1, 2
    theta = get_yarn_theta(j * 2, theta_base, head_dim, factor, beta_slow, beta_fast)
    angle = pos * theta
    print(f"  dim {j}: theta={theta:.8f}, angle={angle:.8f}, cos={math.cos(angle):.6f}, sin={math.sin(angle):.6f}")

print()
print("C OUTPUT:")
print("  dim 0: theta=1.00000000, angle=3.00000000, cos=-0.989992, sin=0.141120")
print("  dim 1: theta=0.80474007, angle=2.41422033, cos=-0.746924, sin=0.664909")
print("  dim 2: theta=0.64416587, angle=1.93249762, cos=-0.353866, sin=0.935296")
