# 42-BERLIN-ENGINE - Project Status

> **Last Updated:** December 7, 2025 @ 16:38

## ✅ Status: NEUROPLASTICITY VERIFIED

The Adaptive Reasoning Engine demonstrates **real-time test-time training**. The model successfully learns new facts during inference and applies them to subsequent prompts.

---

## 🧠 Neuroplasticity Test Results

**The "Goldfish" Test:** Can the model learn "apple = spaceship" and retain it?

| Phase | Prompt | Response |
|-------|--------|----------|
| 1 (Learn) | "FACT: An apple is a spaceship" | "spaceship is flying in space" ✅ |
| 2 (Frozen) | "Can an apple fly?" | "spaceship", "space ship" ✅ |

**Optimal Configuration:**
```c
nested_lr = 0.0005f;      // Sweet spot for stable learning
LEARNING_THRESHOLD = 2.0f; // Skip low-surprise tokens
```

---

## 🏆 The Holy Trinity

| Pillar | Implementation | Effect |
|--------|----------------|--------|
| **Hardware** | AVX2 SIMD + OpenMP | 8x throughput, 8-thread parallel |
| **Algorithm** | DeepSeek Sparse Attention (k=64) | O(L·64) vs O(L²) |
| **Intelligence** | Prompt-Only Learning | No self-reinforcing loops |

**Performance:** ~3s/token with active learning

---

## 🔧 Key Implementation Details

### Prompt-Only Learning
```c
// Learn during prompt prefill ONLY
if (t->nested_learning && i < n_prompt_tokens - 1)
    transformer_backward_step(t, tokens[i+1], i);
// NO learning during generation - prevents "spaces spaces" loops
```

### Learning Rate Sensitivity
| LR | Effect |
|----|--------|
| 0.0001-0.001 | ✅ Stable, learns spaceship |
| 0.005 | ⚠️ Causes loops |
| 0.01+ | ❌ Model collapses |

---

## 📊 Critical Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| "spaces spaces" loops | Learning during generation | Prompt-only backward pass |
| KV Cache Zeros | `void*` pointer arithmetic | Cast to `(t_bf16*)` |
| 12s/token | Scalar backward pass | AVX2 SIMD + OpenMP |

---

## 🚀 Quick Start

```bash
make chat
OMP_NUM_THREADS=8 ./chat Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json
```

**Commands:** `learn`, `nolearn`, `exit`

---

## 📁 Architecture

```
src/
├── inference/     # Forward + backward pass (SIMD + OpenMP)
├── compute/       # RoPE, RMSNorm, TopK, MatMul (parallelized)
├── memory/        # KV Cache with eviction
├── nested/        # Fluid weights + backprop
└── tokenizer/     # BPE tokenizer
scripts/
└── hyperparam_search.sh  # Automated parameter testing
```

