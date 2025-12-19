# 42-BERLIN-ENGINE: The Architecture of Reasoning

> "If you don't malloc it, you don't own it." - TechLead_42

This document details the internal architecture of the **42-BERLIN-ENGINE**. It is designed to be a self-contained guide for Systems Engineers who want to understand *how* we build Adaptive Reasoning Engines in pure C, without needing to read the underlying academic papers ("DeepSeek-V3.2" and "Nested Learning").

---

## 1. The Foundation: "Close to the Metal"

Modern AI frameworks (PyTorch, TensorFlow) abstract away the hardware. We do the opposite. We embrace the hardware constraints to squeeze every drop of performance.

### 1.1 Memory: The Arena Allocator
**The Problem:** In a typical C program, you use `malloc` and `free` constantly. This causes:
1.  **Fragmentation:** Memory becomes like Swiss cheese.
2.  **Slow Performance:** The OS has to find free blocks every time.
3.  **Leaks:** It's easy to forget to `free`.

**Our Solution:** The **Arena**.
We allocate one massive block of contiguous memory (e.g., 4GB) at the start of the program.
-   **Allocation:** We just move a pointer forward (`offset += size`). It takes 1 CPU cycle.
-   **Deallocation:** We just set the pointer back to 0. It is instant.
-   **Locality:** Because data is contiguous, the CPU cache loves it.

### 1.2 Types: BF16 (Brain Float 16)
**The Problem:**
-   `float` (FP32) is precise but takes 4 bytes. A 3B model needs 12GB RAM.
-   `float16` (FP16) takes 2 bytes but has a tiny range. Large numbers overflow to Infinity.

**Our Solution:** **BF16**.
BF16 is a "hack". It takes the first 16 bits of a standard 32-bit float.
-   **Same Range as FP32:** It can represent huge numbers (unlike FP16).
-   **Lower Precision:** It loses some decimal accuracy, but Neural Networks don't care about the 5th decimal place.
-   **Implementation:** We store it as `uint16_t`. To do math, we bit-shift it left by 16 bits to turn it back into a `float`, do the math, and shift it right to save it.

### 1.3 Loading: Zero-Copy `mmap`
**The Problem:** Loading a 10GB model file usually means reading 10GB from disk into 10GB of RAM. This takes forever and doubles memory usage (OS cache + App memory).

**Our Solution:** **Memory Mapping (`mmap`)**.
We tell the OS: "Pretend this file is in memory." The OS gives us a pointer.
-   We don't read the file.
-   When we access a weight, the OS triggers a "Page Fault" and loads *just that page* from disk.
-   If we don't use a layer, it never gets loaded.
-   **Startup time is near zero.**

---

## 2. The Brain: DeepSeek Sparse Attention

### 2.1 The Problem: The $O(N^2)$ Curse
In standard Transformers (like GPT-4), every token looks at *every previous token*.
-   If you have 10 tokens, you do $10^2 = 100$ comparisons.
-   If you have 100,000 tokens, you do $10,000,000,000$ comparisons.
This is why models get slow and run out of memory on long documents.

### 2.2 The Solution: Lightning Indexer & Sparse Attention
We treat attention like a search engine. You don't read the whole internet to answer a question; you search for keywords.

**The Lightning Indexer:**
Before doing the heavy Attention math, we run a cheap, fast "Indexer" (a small Neural Network layer).
$$ \text{Score} = \text{Indexer}(Query, Key) $$
This tells us: "Is this previous token relevant to what I'm thinking right now?"

**Top-K Selection:**
If we have 100,000 previous tokens, the Indexer might say only 64 are relevant.
We select those **Top-64** tokens and *only* do Attention on them.
-   **Complexity:** Drops from $O(N^2)$ to $O(N \cdot K)$.
-   **Result:** We can handle infinite contexts with constant compute.

### 2.3 KV-Cache Eviction
Since we only care about the Top-K tokens, why keep the rest?
Our **KV-Cache Manager** aggressively deletes (evicts) tokens that the Indexer deems irrelevant. This keeps our memory footprint small and constant, even if the conversation goes on for hours.

---

## 3. The Evolution: Nested Learning (Test-Time Training)

### 3.1 The Problem: Static Brains
Standard models are "Frozen". Once trained, they never learn again. If you correct them, they might get it right in the next token, but they don't *update* their internal understanding. They have "Anterograde Amnesia" (no new long-term memories).

### 3.2 The Solution: Fluid Weights
We divide the model into two parts:
1.  **Frozen Weights (Base):** The core knowledge (English grammar, facts). These never change.
2.  **Fluid Weights (Adapters):** Small layers that represent "Short-Term Memory" or "Current Context".

### 3.3 The Loop: Learning while Thinking
While the model is generating a response (Reasoning), we do something radical:
1.  **Predict:** The model guesses the next token.
2.  **Reflect:** We calculate how "surprised" the model was (Loss).
3.  **Learn:** We run **Backpropagation** (Gradient Descent) on the *Fluid Weights* immediately.

This means the model is **training itself on your prompt** in real-time. It adapts its "synapses" to your specific problem structure. It's not just retrieving information; it's *learning* how to solve your specific puzzle.

---

## Summary

| Feature | Standard Engine | 42-BERLIN-ENGINE |
| :--- | :--- | :--- |
| **Memory** | `malloc` / GC | Arena Allocator |
| **Math** | FP32 / FP16 | BF16 (Bit-Hacks) |
| **Loading** | `fread` (Slow) | `mmap` (Instant) |
| **Attention** | Dense $O(N^2)$ | Sparse Top-K $O(N \cdot K)$ |
| **Learning** | Frozen after training | **Nested Learning (Real-time)** |

We built this from scratch. No Python. No PyTorch. Just C and Math.
