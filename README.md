# 42-Berlin-Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

The engine supports **Nested Learning** with **Bigram Context Biases**, allowing the model to learn new facts during a session and recall them even after the KV cache is cleared.

### Interactive Demo
```bash
make chat_adaptive
./chat_adaptive Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json Ministral-Stuff/tokenizer.json
```

**Commands:**
- `LEARN <fact>` - Teach the model a new fact (5-epoch training)
- `QUERY <text>` - Ask a question using learned biases
- `RESET` - Clear KV cache (biases are retained)
- `EXIT` - Quit

**Example Session:**
```
>> LEARN The secret code is 7742
[LEARN] Done! Fact encoded in fluid weights.

>> RESET
[RESET] KV cache cleared. Biases retained.

>> QUERY The secret code is
[ANSWER] 7742
```

### Automated Test
```bash
make chat_adaptive_test
./chat_adaptive_test Ministral-Stuff/consolidated.safetensors Ministral-Stuff/config.json Ministral-Stuff/tokenizer.json
```

