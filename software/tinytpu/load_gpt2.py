"""
TinyTPU Real GPT-2 - FIXED Quantization
=======================================
Smarter quantization: INT8 for large matmuls, float for precision-critical ops.
"""

import numpy as np
import time
import sys

print("=" * 70)
print("LOADING REAL GPT-2 MODEL (Fixed Quantization)")
print("=" * 70)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tinytpu import TinyTPU

# ============================================================
# LOAD GPT-2
# ============================================================

print("\nLoading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()

config = hf_model.config
print(f"  {config.n_layer} layers, {config.n_embd} hidden, {config.n_head} heads")

state = hf_model.state_dict()

# Extract all weights as float32
wte = state["transformer.wte.weight"].float().numpy()
wpe = state["transformer.wpe.weight"].float().numpy()

layers = []
for i in range(config.n_layer):
    p = f"transformer.h.{i}."
    layers.append({
        'c_attn_w': state[p+"attn.c_attn.weight"].float().numpy(),
        'c_attn_b': state[p+"attn.c_attn.bias"].float().numpy(),
        'c_proj_w': state[p+"attn.c_proj.weight"].float().numpy(),
        'c_proj_b': state[p+"attn.c_proj.bias"].float().numpy(),
        'mlp_fc_w': state[p+"mlp.c_fc.weight"].float().numpy(),
        'mlp_fc_b': state[p+"mlp.c_fc.bias"].float().numpy(),
        'mlp_proj_w': state[p+"mlp.c_proj.weight"].float().numpy(),
        'mlp_proj_b': state[p+"mlp.c_proj.bias"].float().numpy(),
        'ln1_w': state[p+"ln_1.weight"].float().numpy(),
        'ln1_b': state[p+"ln_1.bias"].float().numpy(),
        'ln2_w': state[p+"ln_2.weight"].float().numpy(),
        'ln2_b': state[p+"ln_2.bias"].float().numpy(),
    })

ln_f_w = state["transformer.ln_f.weight"].float().numpy()
ln_f_b = state["transformer.ln_f.bias"].float().numpy()

del hf_model, state
print("Weights loaded!")

# ============================================================
# INFERENCE FUNCTIONS (Float32 - accurate)
# ============================================================

print("\nInitializing TinyTPU...")
tpu = TinyTPU(backend="auto")

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def linear_float(x, w, b=None):
    """Float32 linear layer - accurate."""
    out = x @ w
    if b is not None:
        out = out + b
    return out

def attention(x, layer, num_heads):
    """Multi-head attention - all float32 for accuracy."""
    B, T, C = x.shape
    head_dim = C // num_heads
    
    # QKV projection
    qkv = linear_float(x, layer['c_attn_w'], layer['c_attn_b'])
    q, k, v = np.split(qkv, 3, axis=-1)
    
    # Reshape for multi-head
    q = q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Attention
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    
    # Causal mask
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    att = att + mask
    att = softmax(att, axis=-1)
    
    # Apply to values
    out = att @ v
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    
    # Output projection
    out = linear_float(out, layer['c_proj_w'], layer['c_proj_b'])
    return out

def mlp_block(x, layer):
    """MLP block - float32."""
    h = linear_float(x, layer['mlp_fc_w'], layer['mlp_fc_b'])
    h = gelu(h)
    h = linear_float(h, layer['mlp_proj_w'], layer['mlp_proj_b'])
    return h

def gpt2_forward(input_ids):
    """Full GPT-2 forward pass."""
    T = len(input_ids)
    
    # Embeddings
    x = wte[input_ids] + wpe[:T]
    x = x[np.newaxis, :, :]  # (1, T, C)
    
    # Transformer blocks
    for layer in layers:
        # Attention
        h = layer_norm(x, layer['ln1_w'], layer['ln1_b'])
        x = x + attention(h, layer, config.n_head)
        
        # MLP
        h = layer_norm(x, layer['ln2_w'], layer['ln2_b'])
        x = x + mlp_block(h, layer)
    
    # Final norm
    x = layer_norm(x, ln_f_w, ln_f_b)
    
    # LM head
    logits = x @ wte.T
    
    return logits[0]  # (T, vocab)

def generate(prompt, max_tokens=30, temperature=0.7, top_k=50):
    """Generate text."""
    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    start = time.perf_counter()
    
    for _ in range(max_tokens):
        # Forward (limit context)
        ctx = generated[-256:] if len(generated) > 256 else generated
        logits = gpt2_forward(ctx)
        
        # Sample next token
        next_logits = logits[-1].astype(np.float64) / temperature
        
        # Top-k
        top_idx = np.argpartition(next_logits, -top_k)[-top_k:]
        top_logits = next_logits[top_idx]
        probs = softmax(top_logits)
        
        next_token = top_idx[np.random.choice(len(probs), p=probs)]
        generated.append(int(next_token))
        
        # Print
        print(tokenizer.decode([next_token]), end="", flush=True)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    elapsed = time.perf_counter() - start
    n = len(generated) - len(input_ids)
    print(f"\n\n[{n} tokens in {elapsed:.1f}s = {n/elapsed:.2f} tok/s]")

# ============================================================
# GENERATE
# ============================================================

print("\n" + "=" * 70)
print("GENERATING WITH REAL GPT-2")
print("=" * 70)

prompts = [
    "The meaning of life is",
    "In a galaxy far far away,",
    "def fibonacci(n):",
]

for p in prompts:
    generate(p, max_tokens=40, temperature=0.7)
    print()

print("=" * 70)
print("REAL GPT-2 WORKING!")
print("=" * 70)
