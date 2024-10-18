"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_state_dict_with_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if 'wavelet_attention' not in name:  # Skip wavelet_attention parameters
                if model_state_dict[name].shape != param.shape:
                    print(f"Mismatch in {name}: checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
                    print(f"Skipping parameter {name} due to shape mismatch")
                else:
                    model_state_dict[name].copy_(param)
        else:
            print(f"Skipping {name} as it's not in the current model")
    model.load_state_dict(model_state_dict, strict=False)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Use the new function here
    load_state_dict_with_mismatch(model, state_dict)
    
    wavelet_dim = checkpoint['model_args'].get('wavelet_dim', 32)  # Default to 32 if not found
    model.config.wavelet_dim = wavelet_dim
    if 'vocab_size' in checkpoint['model_args']:
        model.config.vocab_size = checkpoint['model_args']['vocab_size']
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    wavelet_dim = 32
    model.config.wavelet_dim = wavelet_dim

def generate_with_wavelets(model, x, max_new_tokens, temperature, top_k, wavelet_dim):
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        
        # Create a dummy wavelet input
        dummy_wavelets = torch.zeros((idx_cond.size(0), idx_cond.size(1), wavelet_dim), device=device)
        
        # Forward the model to get the logits for the index in the sequence
        output = model(idx_cond, dummy_wavelets)
        
        # Check if the output is a tuple (it might be if the model returns both logits and loss)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Ensure we're only looking at the last token's logits
        logits = logits[:, -1, :]
        
        # Clip logits to the vocab size
        logits = logits[:, :model.config.vocab_size]
        
        # Pluck the logits at the final step and scale by desired temperature
        logits = logits / temperature
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # Ensure the sampled index is within the vocab size
        idx_next = torch.clamp(idx_next, 0, model.config.vocab_size - 1)
        # Append sampled index to the running sequence and continue
        x = torch.cat((x, idx_next), dim=1)
    
    return x

def safe_encode(s, vocab_size):
    encoded = encode(s)
    return [token for token in encoded if token < vocab_size]

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
start_ids = safe_encode(start, model.config.vocab_size)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f"Initial input shape: {x.shape}")
print(f"Model config: {model.config}")
print(f"Vocabulary size: {model.config.vocab_size}")

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            try:
                y = generate_with_wavelets(model, x, max_new_tokens, temperature, top_k, wavelet_dim)
                generated_text = decode(y[0].tolist())
                print(f"Generated text (sample {k+1}):")
                print(generated_text)
                print('---------------')
            except Exception as e:
                print(f"Error during generation (sample {k+1}): {str(e)}")
                print(f"Current sequence: {x}")
                print(f"Current sequence decoded: {decode(x[0].tolist())}")
                continue