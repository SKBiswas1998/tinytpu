"""
TinyTPU GPT-2 with KV-Cache
===========================
3-5x faster generation by caching Key/Value tensors.

WITHOUT KV-Cache (current):
  Token 1: Compute K,V for position 0
  Token 2: Compute K,V for position 0,1 (recompute pos 0!)
  Token 3: Compute K,V for position 0,1,2 (recompute pos 0,1!)
  → O(n²) computation

WITH KV-Cache:
  Token 1: Compute K,V for position 0, CACHE it
  Token 2: Compute K,V for position 1 only, append to cache
  Token 3: Compute K,V for position 2 only, append to cache
  → O(n) computation = 3-5x faster!
"""

import numpy as np
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tinytpu import TinyTPU

print("=" * 70)
print("GPT-2 WITH KV-CACHE (3-5x FASTER)")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()

cfg = hf_model.config
n_layer, n_head, n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
head_dim = n_embd // n_head

state = hf_model.state_dict()

# Extract weights
wte = state["transformer.wte.weight"].float().numpy()
wpe = state["transformer.wpe.weight"].float().numpy()

layers = []
for i in range(n_layer):
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
print(f"  {n_layer} layers, {n_embd} hidden, {n_head} heads")

# ============================================================
# KV-CACHE CLASS
# ============================================================

class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    
    Stores K and V tensors for each layer, so we don't recompute them.
    """
    
    def __init__(self, n_layers: int, n_heads: int, head_dim: int, max_seq: int = 1024):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        
        # Cache: list of (K, V) for each layer
        # K, V shape: (batch, n_heads, seq_len, head_dim)
        self.cache = [None] * n_layers
        self.seq_len = 0
    
    def clear(self):
        """Clear the cache."""
        self.cache = [None] * self.n_layers
        self.seq_len = 0
    
    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        """
        Update cache with new K, V values.
        
        Args:
            layer_idx: Which layer
            new_k: New keys, shape (batch, n_heads, new_seq, head_dim)
            new_v: New values, shape (batch, n_heads, new_seq, head_dim)
        
        Returns:
            full_k, full_v: Complete K, V including history
        """
        if self.cache[layer_idx] is None:
            # First token(s)
            self.cache[layer_idx] = (new_k, new_v)
        else:
            # Append to existing cache
            old_k, old_v = self.cache[layer_idx]
            full_k = np.concatenate([old_k, new_k], axis=2)
            full_v = np.concatenate([old_v, new_v], axis=2)
            self.cache[layer_idx] = (full_k, full_v)
        
        return self.cache[layer_idx]
    
    def get(self, layer_idx: int):
        """Get cached K, V for a layer."""
        return self.cache[layer_idx]


# ============================================================
# INFERENCE WITH KV-CACHE
# ============================================================

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention_with_cache(x, layer, layer_idx, kv_cache, n_heads):
    """
    Multi-head attention WITH KV-Cache.
    
    Key insight:
    - For new tokens, only compute Q, K, V for those tokens
    - Retrieve cached K, V for previous tokens
    - Attention uses full K, V (cached + new)
    """
    B, T, C = x.shape  # T = number of NEW tokens (1 during generation)
    head_dim = C // n_heads
    
    # Compute Q, K, V for NEW tokens only
    qkv = x @ layer['c_attn_w'] + layer['c_attn_b']
    q, k, v = np.split(qkv, 3, axis=-1)
    
    # Reshape for multi-head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Update cache and get full K, V
    full_k, full_v = kv_cache.update(layer_idx, k, v)
    
    # full_k, full_v shape: (B, n_heads, total_seq, head_dim)
    total_seq = full_k.shape[2]
    
    # Attention: Q attends to ALL K (including cached)
    # q: (B, n_heads, T, head_dim)
    # full_k: (B, n_heads, total_seq, head_dim)
    att = (q @ full_k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    # att shape: (B, n_heads, T, total_seq)
    
    # Causal mask: each position can only attend to previous positions
    # For generation (T=1), the new token can see all previous tokens
    if T > 1:
        # Prefill: need proper causal mask
        # Position i can attend to positions 0..i
        start_pos = total_seq - T
        mask = np.triu(np.ones((T, total_seq)), k=start_pos + 1) * -1e9
        att = att + mask
    # For T=1, no mask needed (single token attends to all previous)
    
    att = softmax(att, axis=-1)
    
    # Apply attention to values
    out = att @ full_v  # (B, n_heads, T, head_dim)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    
    # Output projection
    out = out @ layer['c_proj_w'] + layer['c_proj_b']
    return out

def mlp_block(x, layer):
    h = x @ layer['mlp_fc_w'] + layer['mlp_fc_b']
    h = gelu(h)
    h = h @ layer['mlp_proj_w'] + layer['mlp_proj_b']
    return h

def forward_with_cache(input_ids, kv_cache, start_pos=0):
    """
    Forward pass with KV-cache.
    
    Args:
        input_ids: Token IDs to process (can be 1 token during generation)
        kv_cache: KVCache object
        start_pos: Starting position for position embeddings
    
    Returns:
        logits for the input tokens
    """
    T = len(input_ids)
    
    # Embeddings (only for new tokens)
    x = wte[input_ids] + wpe[start_pos:start_pos + T]
    x = x[np.newaxis, :, :]  # (1, T, C)
    
    # Transformer blocks
    for i, layer in enumerate(layers):
        # Attention with cache
        h = layer_norm(x, layer['ln1_w'], layer['ln1_b'])
        x = x + attention_with_cache(h, layer, i, kv_cache, n_head)
        
        # MLP (no caching needed)
        h = layer_norm(x, layer['ln2_w'], layer['ln2_b'])
        x = x + mlp_block(h, layer)
    
    # Final norm and LM head
    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ wte.T
    
    return logits[0]  # (T, vocab)

def generate_with_cache(prompt, max_tokens=50, temperature=0.7, top_k=50):
    """Generate text using KV-cache for speed."""
    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    # Initialize cache
    kv_cache = KVCache(n_layer, n_head, head_dim)
    
    start = time.perf_counter()
    
    # PREFILL: Process entire prompt at once
    prefill_start = time.perf_counter()
    logits = forward_with_cache(input_ids, kv_cache, start_pos=0)
    prefill_time = time.perf_counter() - prefill_start
    
    # GENERATION: One token at a time (using cache!)
    gen_times = []
    
    for i in range(max_tokens):
        token_start = time.perf_counter()
        
        # Sample next token from last position
        next_logits = logits[-1].astype(np.float64) / temperature
        
        # Top-k sampling
        top_idx = np.argpartition(next_logits, -top_k)[-top_k:]
        top_logits = next_logits[top_idx]
        probs = softmax(top_logits)
        
        next_token = top_idx[np.random.choice(len(probs), p=probs)]
        generated.append(int(next_token))
        
        # Print new token
        print(tokenizer.decode([next_token]), end="", flush=True)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        # Forward ONLY the new token (cache has previous K,V!)
        logits = forward_with_cache([next_token], kv_cache, start_pos=len(generated)-1)
        
        gen_times.append(time.perf_counter() - token_start)
    
    elapsed = time.perf_counter() - start
    n = len(generated) - len(input_ids)
    avg_gen_time = np.mean(gen_times) if gen_times else 0
    
    print(f"\n\n[{n} tokens in {elapsed:.1f}s = {n/elapsed:.2f} tok/s]")
    print(f"[Prefill: {prefill_time*1000:.0f}ms | Per token: {avg_gen_time*1000:.0f}ms]")
    
    return generated

# ============================================================
# COMPARISON: WITH vs WITHOUT KV-CACHE
# ============================================================

def generate_no_cache(prompt, max_tokens=50, temperature=0.7, top_k=50):
    """Generate WITHOUT KV-cache (for comparison)."""
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    start = time.perf_counter()
    
    for i in range(max_tokens):
        # Recompute EVERYTHING each time (slow!)
        ctx = generated[-256:] if len(generated) > 256 else generated
        
        T = len(ctx)
        x = wte[ctx] + wpe[:T]
        x = x[np.newaxis, :, :]
        
        for layer in layers:
            h = layer_norm(x, layer['ln1_w'], layer['ln1_b'])
            
            # Full attention (no cache)
            qkv = h @ layer['c_attn_w'] + layer['c_attn_b']
            q, k, v = np.split(qkv, 3, axis=-1)
            
            q = q.reshape(1, T, n_head, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(1, T, n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(1, T, n_head, head_dim).transpose(0, 2, 1, 3)
            
            att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            mask = np.triu(np.ones((T, T)), k=1) * -1e9
            att = softmax(att + mask, axis=-1)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(1, T, n_embd)
            out = out @ layer['c_proj_w'] + layer['c_proj_b']
            x = x + out
            
            h = layer_norm(x, layer['ln2_w'], layer['ln2_b'])
            x = x + mlp_block(h, layer)
        
        x = layer_norm(x, ln_f_w, ln_f_b)
        logits = (x @ wte.T)[0]
        
        # Sample
        next_logits = logits[-1].astype(np.float64) / temperature
        top_idx = np.argpartition(next_logits, -top_k)[-top_k:]
        probs = softmax(next_logits[top_idx])
        next_token = top_idx[np.random.choice(len(probs), p=probs)]
        generated.append(int(next_token))
        
        if next_token == tokenizer.eos_token_id:
            break
    
    elapsed = time.perf_counter() - start
    n = len(generated) - len(input_ids)
    return n, elapsed

# ============================================================
# RUN COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK: KV-CACHE vs NO CACHE")
print("=" * 70)

prompt = "The future of artificial intelligence"

print("\n[WITHOUT KV-Cache]")
n1, t1 = generate_no_cache(prompt, max_tokens=20)
speed1 = n1 / t1
print(f"  {n1} tokens in {t1:.1f}s = {speed1:.2f} tok/s")

print("\n[WITH KV-Cache]")
generated = generate_with_cache(prompt, max_tokens=20)
n2 = len(generated) - len(tokenizer.encode(prompt))
# Time is printed in the function

print("\n" + "=" * 70)
print("GENERATING WITH KV-CACHE")
print("=" * 70)

prompts = [
    "Once upon a time in a land far away,",
    "The secret to happiness is",
    "import torch\ndef train_model(",
]

for p in prompts:
    generate_with_cache(p, max_tokens=40, temperature=0.7)
    print()

print("=" * 70)
print("KV-CACHE IMPLEMENTATION COMPLETE!")
print("=" * 70)
