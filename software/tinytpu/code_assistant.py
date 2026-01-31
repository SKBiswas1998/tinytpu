"""
TinyTPU Code Assistant - Code Model
===================================
Uses a code-trained model for better results.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("TINYTPU CODE ASSISTANT (Code-Optimized)")
print("=" * 70)

device = torch.device('cpu')

# ============================================================
# LOAD CODE MODEL
# ============================================================

print("\nLoading code model... (downloads ~350MB first time)")

# Code-specific model - much better for programming
MODEL_NAME = "bigcode/tiny_starcoder_py"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

# ============================================================
# GENERATION
# ============================================================

def generate(prompt, max_tokens=150, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - start
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    return text, tokens, elapsed

# ============================================================
# DEMO
# ============================================================

print("\n" + "=" * 70)
print("CODE GENERATION DEMO")
print("=" * 70)

demos = [
    "def quicksort(arr):\n    ",
    "def is_palindrome(s: str) -> bool:\n    ",
    "def fibonacci(n: int) -> int:\n    ",
    "# Function to read a CSV file and return a list of dictionaries\ndef read_csv(filename):\n    ",
]

for prompt in demos:
    print(f"\n{'='*50}")
    print(f"PROMPT: {prompt.strip()}")
    print("=" * 50)
    
    result, tokens, elapsed = generate(prompt, max_tokens=120, temperature=0.2)
    print(result)
    print(f"\n[{tokens} tokens in {elapsed:.1f}s = {tokens/elapsed:.1f} tok/s]")

print("\n" + "=" * 70)
print("Interactive Mode (type code prompt, 'quit' to exit)")
print("=" * 70)

while True:
    try:
        prompt = input("\n>>> ").strip()
        if not prompt or prompt.lower() == 'quit':
            break
        result, tokens, elapsed = generate(prompt + "\n    ", max_tokens=150)
        print("\n" + result)
        print(f"\n[{tokens} tokens, {tokens/elapsed:.1f} tok/s]")
    except (KeyboardInterrupt, EOFError):
        break

print("\nDone!")
