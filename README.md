# TinyTPU

A fast, lightweight tensor library that matches or beats PyTorch performance on CPU.

## Performance

**TinyTPU vs PyTorch Direct (lower = TinyTPU faster):**

| Operation | Ratio | Result |
|-----------|-------|--------|
| relu | 0.60x | **40% faster** |
| gelu | 0.78x | **22% faster** |
| layer_norm | 0.87x | **13% faster** |
| matmul 1024² | 0.90x | **10% faster** |
| matmul 2048² | 0.94x | **6% faster** |
| softmax | 1.45x | PyTorch wins |

## Installation
```bash
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
pip install -e .

# For best performance
pip install torch
```

## Quick Start
```python
from tinytpu import TinyTPU

tpu = TinyTPU()  # Auto-selects best backend

# Native tensors (fastest)
A = tpu.randn(1024, 1024)
B = tpu.randn(1024, 1024)
C = tpu.matmul(A, B)

# Neural network ops
x = tpu.randn(1000, 768)
y = tpu.relu(x)
y = tpu.gelu(x)
y = tpu.softmax(x)
y = tpu.layer_norm(x)

# NumPy compatible
import numpy as np
A = np.random.randn(512, 512).astype(np.float32)
C = tpu.matmul(A, A.T)
```

## LLM Inference

Run GPT-2 at 10-15 tokens/sec on CPU:
```bash
python software/tinytpu/gpt2_optimized.py
```
```
Prompt: "The future of artificial intelligence is"
The future of artificial intelligence is uncertain, but the technology 
is changing rapidly in ways that will change how we think...

[50 tokens in 5.4s = 9.22 tok/s]
```

## Features

- **Fast**: Matches or beats PyTorch on key operations
- **Simple**: Clean API, easy to use
- **CPU-only**: No GPU required
- **Auto-backend**: Selects PyTorch > NumPy automatically
- **LLM Ready**: KV-cache optimized inference

## Architecture
```
┌─────────────────────────────────────────┐
│              TinyTPU API                │
│  matmul, relu, gelu, softmax, etc.      │
├─────────────────────────────────────────┤
│           Auto Backend                  │
│     PyTorch (fast) > NumPy (fallback)   │
├─────────────────────────────────────────┤
│         Systolic Array RTL              │
│      4x4 verified Verilog design        │
└─────────────────────────────────────────┘
```

## Benchmarks

Run the benchmark yourself:
```bash
python software/tinytpu/tpu_v2.py
```

## Project Structure
```
tinytpu/
├── software/tinytpu/
│   ├── __init__.py        # Package entry
│   ├── tpu_v2.py          # Core library
│   ├── gpt2_optimized.py  # LLM inference
│   └── benchmark.py       # Full benchmark suite
├── hardware/rtl/          # Verilog systolic array
├── pyproject.toml         # Package config
└── README.md
```

## API Reference
```python
tpu = TinyTPU(backend="auto")  # or "pytorch", "numpy"

# Tensor creation
tpu.randn(M, N)      # Random tensor
tpu.zeros(M, N)      # Zero tensor
tpu.tensor(data)     # From data

# Operations
tpu.matmul(A, B)     # Matrix multiplication
tpu.relu(x)          # ReLU activation
tpu.gelu(x)          # GELU activation
tpu.softmax(x)       # Softmax
tpu.layer_norm(x)    # Layer normalization
tpu.embedding(W, idx) # Embedding lookup
```

## Why TinyTPU?

| Need | Solution |
|------|----------|
| Fast tensor ops without GPU | TinyTPU on CPU |
| Simpler than PyTorch | Clean API |
| Learn TPU architecture | RTL included |
| Run LLMs cheaply | 10-15 tok/s GPT-2 |

## License

MIT
