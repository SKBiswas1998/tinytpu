# TinyTPU

A minimal Tensor Processing Unit implementation for learning and running LLM inference on cheap hardware.

![TinyTPU](https://img.shields.io/badge/TinyTPU-Educational-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

- **Real LLM Inference**: Run GPT-2 (124M params) at 10-15 tokens/sec on CPU
- **Systolic Array RTL**: Verified 4×4 systolic array in Verilog
- **Auto Backend**: Automatically selects PyTorch (4x faster) or NumPy
- **KV-Cache**: Optimized autoregressive generation
- **Educational**: Cycle-accurate simulation to understand TPU internals

## 📦 Installation
```bash
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
pip install torch transformers numpy
```

## ⚡ Quick Start

### Run GPT-2 Inference
```python
# Generate text with real GPT-2
python software/tinytpu/gpt2_optimized.py
```

Output:
```
Prompt: "The future of artificial intelligence is"
The future of artificial intelligence is uncertain, but the technology 
is changing rapidly in ways that will change how we think...

[50 tokens in 5.4s = 9.22 tok/s]
```

### Basic TPU Operations
```python
from tinytpu import TinyTPU
import numpy as np

# Create TPU (auto-selects best backend)
tpu = TinyTPU()
print(tpu)  # TinyTPU(backend='pytorch', device='cpu')

# Matrix multiplication
A = np.random.randint(-128, 127, (64, 128), dtype=np.int8)
B = np.random.randint(-128, 127, (128, 64), dtype=np.int8)
C = tpu.matmul(A, B)

# Neural network operations
x = np.random.randn(4, 768).astype(np.float32)
y = tpu.softmax(x)
y = tpu.gelu(x)
y = tpu.layer_norm(x)
```

### Benchmark
```python
tpu = TinyTPU()
result = tpu.benchmark(size=512, iterations=10)
print(f"{result['time_ms']:.2f}ms, {result['gops']:.2f} GOPS")
```

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         TinyTPU                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │  Python API     │  tpu.matmul(), tpu.softmax(), etc.        │
│  └────────┬────────┘                                           │
│           │                                                     │
│  ┌────────▼────────┐                                           │
│  │ Unified Backend │  Auto-selects: PyTorch > Numba > NumPy    │
│  └────────┬────────┘                                           │
│           │                                                     │
│  ┌────────▼────────┐                                           │
│  │  Systolic Array │  4×4 weight-stationary dataflow           │
│  │     (RTL)       │  Verified Verilog implementation          │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Systolic Array

The TPU uses a **weight-stationary** systolic array:
```
        Activations flow →
        ┌───┐ ┌───┐ ┌───┐ ┌───┐
        │ A │ │ A │ │ A │ │ A │
        └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
          ▼     ▼     ▼     ▼
Weights ┌───┐ ┌───┐ ┌───┐ ┌───┐
stay    │W×A│→│W×A│→│W×A│→│W×A│→ Results
in      ├───┤ ├───┤ ├───┤ ├───┤
place   │W×A│→│W×A│→│W×A│→│W×A│→
        ├───┤ ├───┤ ├───┤ ├───┤
        │W×A│→│W×A│→│W×A│→│W×A│→
        ├───┤ ├───┤ ├───┤ ├───┤
        │W×A│→│W×A│→│W×A│→│W×A│→
        └───┘ └───┘ └───┘ └───┘
```

Each Processing Element (PE):
1. Holds one weight (stationary)
2. Receives activation from left
3. Multiplies and accumulates
4. Passes activation right, partial sum down

## 📁 Project Structure
```
tinytpu/
├── software/
│   ├── tinytpu/
│   │   ├── __init__.py         # Package exports
│   │   ├── unified_backend.py  # Auto backend selection
│   │   ├── llm.py              # LLM engine
│   │   ├── gpt2_optimized.py   # GPT-2 with KV-cache
│   │   └── gpt2_kvcache.py     # KV-cache implementation
│   └── tests/
│       ├── brutal_test.py      # 40 edge case tests
│       └── production_validation.py  # 16 validation tests
├── hardware/
│   ├── rtl/
│   │   └── systolic_array.v    # 4×4 verified RTL
│   └── tb/
│       └── professional_tb.v   # 1033 test vectors
└── README.md
```

## 🧪 Testing
```bash
# Run all tests (56 total)
cd software
python -m pytest tests/ -v

# Run brutal edge case tests
python tests/brutal_test.py

# Run production validation
python tests/production_validation.py
```

## 🔧 Hardware

The RTL implementation is in `hardware/rtl/systolic_array.v`:

- **Size**: 4×4 processing elements
- **Data width**: 8-bit inputs, 32-bit accumulator
- **Dataflow**: Weight-stationary
- **Verified**: 1033 test vectors pass

### Simulate with Icarus Verilog
```bash
cd hardware
iverilog -o sim.vvp rtl/systolic_array.v tb/professional_tb.v
vvp sim.vvp
```

## 📊 Performance

| Configuration | Speed | Notes |
|--------------|-------|-------|
| NumPy (baseline) | 0.6 tok/s | No optimization |
| NumPy + KV-cache | 0.9 tok/s | 1.5x speedup |
| PyTorch + KV-cache | **10-15 tok/s** | **15-25x speedup** |

### Benchmark Results (512×512 matmul)

| Backend | Time | GOPS |
|---------|------|------|
| NumPy | 254ms | 1.1 |
| PyTorch CPU | 63ms | 4.3 |

## 🎯 Use Cases

1. **Education**: Learn how TPUs and systolic arrays work
2. **Cheap LLM Inference**: Run models without expensive GPU
3. **Hardware Prototyping**: Verified RTL for FPGA deployment
4. **Research**: Experiment with quantization, dataflow

## 📚 How It Works

### KV-Cache Optimization

Without KV-cache (slow):
```
Token 1: Compute K,V for position 0
Token 2: Compute K,V for position 0,1 (recompute!)
Token 3: Compute K,V for position 0,1,2 (recompute!)
→ O(n²) computation
```

With KV-cache (fast):
```
Token 1: Compute K,V for position 0, CACHE it
Token 2: Compute K,V for position 1 only, append
Token 3: Compute K,V for position 2 only, append
→ O(n) computation
```

### Auto Backend Selection
```python
Priority: CUDA > MPS > PyTorch CPU > Numba > NumPy

# Automatic
tpu = TinyTPU()  # Picks best available

# Manual
tpu = TinyTPU(backend="pytorch")
tpu = TinyTPU(backend="numpy")
```

## 🛣️ Roadmap

- [x] Systolic array RTL
- [x] Python API
- [x] Unified backend system
- [x] GPT-2 inference
- [x] KV-cache optimization
- [ ] INT8 quantization
- [ ] Larger models (TinyLlama, Phi-2)
- [ ] FPGA deployment
- [ ] Memory-mapped weights

## 📄 License

MIT License - feel free to use for learning and research!

## 🙏 Acknowledgments

- Google's TPU architecture papers
- HuggingFace for model weights
- The open-source hardware community
