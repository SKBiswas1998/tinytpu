"""
TinyTPU GPT-2 - Fully Optimized
===============================
KV-Cache + PyTorch backend = Maximum speed
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 70)
print("GPT-2 FULLY OPTIMIZED (KV-Cache + PyTorch)")
print("=" * 70)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
hf_model.eval()

cfg = hf_model.config
n_layer, n_head, n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
head_dim = n_embd // n_head
print(f"  {n_layer} layers, {n_embd} hidden, {n_head} heads")

state = hf_model.state_dict()

# Convert to tensors on device
wte = state["transformer.wte.weight"].to(device)
wpe = state["transformer.wpe.weight"].to(device)

layers = []
for i in range(n_layer):
    p = f"transformer.h.{i}."
    layers.append({
        'c_attn_w': state[p+"attn.c_attn.weight"].to(device),
        'c_attn_b': state[p+"attn.c_attn.bias"].to(device),
        'c_proj_w': state[p+"attn.c_proj.weight"].to(device),
        'c_proj_b': state[p+"attn.c_proj.bias"].to(device),
        'mlp_fc_w': state[p+"mlp.c_fc.weight"].to(device),
        'mlp_fc_b': state[p+"mlp.c_fc.bias"].to(device),
        'mlp_proj_w': state[p+"mlp.c_proj.weight"].to(device),
        'mlp_proj_b': state[p+"mlp.c_proj.bias"].to(device),
        'ln1_w': state[p+"ln_1.weight"].to(device),
        'ln1_b': state[p+"ln_1.bias"].to(device),
        'ln2_w': state[p+"ln_2.weight"].to(device),
        'ln2_b': state[p+"ln_2.bias"].to(device),
    })

ln_f_w = state["transformer.ln_f.weight"].to(device)
ln_f_b = state["transformer.ln_f.bias"].to(device)

del hf_model
print("Weights loaded to", device)

# ============================================================
# KV-CACHE (PyTorch version)
# ============================================================

class KVCache:
    def __init__(self, n_layers, n_heads, head_dim, device):
        self.cache = [None] * n_layers
        self.device = device
    
    def clear(self):
        self.cache = [None] * len(self.cache)
    
    def update(self, layer_idx, k, v):
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (k, v)
        else:
            old_k, old_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([old_k, k], dim=2),
                torch.cat([old_v, v], dim=2)
            )
        return self.cache[layer_idx]

# ============================================================
# OPTIMIZED FORWARD PASS
# ============================================================

@torch.no_grad()
def forward_optimized(input_ids, kv_cache, start_pos=0):
    """Optimized forward with PyTorch + KV-cache."""
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, device=device)
    
    T = input_ids.shape[0]
    
    # Embeddings
    x = wte[input_ids] + wpe[start_pos:start_pos + T]
    x = x.unsqueeze(0)  # (1, T, C)
    
    for i, layer in enumerate(layers):
        # Layer norm 1
        h = F.layer_norm(x, (n_embd,), layer['ln1_w'], layer['ln1_b'])
        
        # Attention
        qkv = h @ layer['c_attn_w'] + layer['c_attn_b']
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(1, T, n_head, head_dim).transpose(1, 2)
        k = k.view(1, T, n_head, head_dim).transpose(1, 2)
        v = v.view(1, T, n_head, head_dim).transpose(1, 2)
        
        # Update cache
        k, v = kv_cache.update(i, k, v)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Causal mask for prefill
        if T > 1:
            total_len = k.shape[2]
            mask = torch.triu(torch.ones(T, total_len, device=device), diagonal=total_len-T+1) * -1e9
            att = att + mask
        
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(1, T, n_embd)
        out = out @ layer['c_proj_w'] + layer['c_proj_b']
        x = x + out
        
        # Layer norm 2 + MLP
        h = F.layer_norm(x, (n_embd,), layer['ln2_w'], layer['ln2_b'])
        h = h @ layer['mlp_fc_w'] + layer['mlp_fc_b']
        h = F.gelu(h)
        h = h @ layer['mlp_proj_w'] + layer['mlp_proj_b']
        x = x + h
    
    # Final norm + LM head
    x = F.layer_norm(x, (n_embd,), ln_f_w, ln_f_b)
    logits = x @ wte.T
    
    return logits[0]

def generate_optimized(prompt, max_tokens=50, temperature=0.7, top_k=50):
    """Generate with full optimization."""
    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    kv_cache = KVCache(n_layer, n_head, head_dim, device)
    
    start = time.perf_counter()
    
    # Prefill
    prefill_start = time.perf_counter()
    with torch.no_grad():
        logits = forward_optimized(input_ids, kv_cache, start_pos=0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start
    
    # Generate
    gen_times = []
    
    for i in range(max_tokens):
        token_start = time.perf_counter()
        
        # Sample
        next_logits = logits[-1].float() / temperature
        
        # Top-k
        top_vals, top_idx = torch.topk(next_logits, top_k)
        probs = F.softmax(top_vals, dim=-1)
        
        idx = torch.multinomial(probs, 1)
        next_token = top_idx[idx].item()
        generated.append(next_token)
        
        # Print
        print(tokenizer.decode([next_token]), end="", flush=True)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        # Forward new token only
        with torch.no_grad():
            logits = forward_optimized([next_token], kv_cache, start_pos=len(generated)-1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        gen_times.append(time.perf_counter() - token_start)
    
    elapsed = time.perf_counter() - start
    n = len(generated) - len(input_ids)
    avg_gen = np.mean(gen_times) if gen_times else 0
    
    print(f"\n\n[{n} tokens in {elapsed:.1f}s = {n/elapsed:.2f} tok/s]")
    print(f"[Prefill: {prefill_time*1000:.0f}ms | Per token: {avg_gen*1000:.0f}ms]")
    
    return generated

# ============================================================
# BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("SPEED COMPARISON")
print("=" * 70)

# Warmup
print("\nWarmup...")
_ = generate_optimized("Hello", max_tokens=5, temperature=0.7)

print("\n" + "=" * 70)
print("GENERATING TEXT")
print("=" * 70)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a magical kingdom,",
    "The best way to learn programming is",
    "In the year 2050, humans will",
]

for p in prompts:
    generate_optimized(p, max_tokens=50, temperature=0.7)
    print()

print("=" * 70)
print("OPTIMIZATION COMPLETE!")
print("=" * 70)
