# TinyTPU

An open-source tensor processing unit for memory-efficient LLM inference.

Run 70B LLM inference on a $150 FPGA board using layer-streaming architecture.

## Features

- INT8 Systolic Array - Hardware-accelerated matrix multiplication
- Layer Streaming - Process one transformer layer at a time
- PyTorch Integration - Drop-in replacement for GPU inference
- AirLLM Compatible - Works with AirLLM memory-efficient inference

## Quick Start
```bash
pip install tinytpu
```
```python
from tinytpu import TinyTPU
import numpy as np

tpu = TinyTPU()
A = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
B = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
C = tpu.matmul(A, B)
```

## License

Apache 2.0
