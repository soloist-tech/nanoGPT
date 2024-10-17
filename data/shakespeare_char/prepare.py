"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import pywt  # Add this import

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode_ids(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode_ids(train_data)
val_ids = encode_ids(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

def generate_char_wavelets(chars, wavelet_length=32):
    char_wavelets = {}
    for i, char in enumerate(chars):
        wavelet = pywt.Wavelet('db1')
        phi, psi, x = wavelet.wavefun(level=5)
        psi = psi / np.max(np.abs(psi))
        psi_resized = np.interp(np.linspace(0, 1, wavelet_length), np.linspace(0, 1, len(psi)), psi)
        char_wavelets[char] = psi_resized
    return char_wavelets

char_wavelets = generate_char_wavelets(chars)

def encode_with_wavelets(s):
    ids = [stoi[c] for c in s]
    wavelets = [char_wavelets[c] for c in s]
    return ids, wavelets

train_ids, train_wavelets = encode_with_wavelets(train_data)
val_ids, val_wavelets = encode_with_wavelets(val_data)

# Export to files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_ids.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_ids.bin'))

train_wavelets = np.array(train_wavelets, dtype=np.float32)
val_wavelets = np.array(val_wavelets, dtype=np.float32)
train_wavelets.tofile(os.path.join(os.path.dirname(__file__), 'train_wavelets.bin'))
val_wavelets.tofile(os.path.join(os.path.dirname(__file__), 'val_wavelets.bin'))

# Update the meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'char_wavelets': char_wavelets,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)