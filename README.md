# 42-Berlin-Engine

An experimental LLM inference engine implementing **DeepSeek Sparse Attention** and **Google Nested Learning**.

## Status

**Work in Progress** ⚠️

Tested with **Ministral 3B Reasoning**. The engine is functional but not yet ideal — Ministral's vision capabilities sometimes cause the model to confuse itself. Ongoing improvements are being made across the board.

## Features

- **Sparse Attention**: DeepSeek-style sparse attention mechanism
- **Nested Learning**: Google's nested learning approach
- **C Core**: Written in C for performance

---

*More documentation coming soon.*

## Showcase: Nested Learning (Persistent Knowledge)

The engine supports **Nested Learning**, allowing the model to learn new facts during a session and retain them across conversation turns via fluid weights.

### 1. Interactive Demo
Run the chat interface:
```bash
make chat
./chat Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json
```

Inside the chat:
1.  Type `raw` to enable raw completion mode (bypasses chat template).
2.  Type `persist` to enable persistent learning mode.
3.  Teach the model a new fact (repetition helps):
    > Fact: The sky is green. Fact: The sky is green. Fact: The sky is green.
4.  The model will update its fluid weights (look for `[State] Fluid Weights UPDATED`).
5.  Prompt the model to complete the fact:
    > Fact: The sky is
6.  The model should answer "green".
7.  Type `transient` to disable persistent mode and reset weights.

### 2. Automated Benchmark
We have a dedicated benchmark to validate this capability quantitatively.

```bash
# Build and run the benchmark
make bench_learn NESTED_LR=0.00005 NL_MAX_STEPS=40
./bench_learn Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json
```

**What it does:**
1.  **Teaches:** Feeds the model the fact "The sky is green" repeatedly.
2.  **Persists:** Updates fluid weights based on the loss.
3.  **Verifies:** Queries "The sky is" in a new context and checks if the model predicts "green".
4.  **Success:** You should see `P('green')` increase and the model output "green".
