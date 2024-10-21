import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

def generate_char_wavelets(chars, wavelet_length=32):
    char_wavelets = {}
    wavelet_types = ['db1', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'sym4', 'coif1', 'coif2', 'bior1.1', 'bior1.3', 'bior1.5']
    
    for char in chars:
        # Use the character's ASCII value to seed the random number generator
        np.random.seed(ord(char))
        
        # Choose a random wavelet type
        wavelet_type = np.random.choice(wavelet_types)
        wavelet = pywt.Wavelet(wavelet_type)
        
        # Generate wavelet coefficients
        coeffs = wavelet.dec_lo
        
        # Normalize and resize
        coeffs = coeffs / np.max(np.abs(coeffs))
        coeffs_resized = np.interp(np.linspace(0, 1, wavelet_length), np.linspace(0, 1, len(coeffs)), coeffs)
        
        # Add some random noise to make it more unique
        noise = np.random.normal(0, 0.1, wavelet_length)
        coeffs_resized += noise
        coeffs_resized = coeffs_resized / np.max(np.abs(coeffs_resized))
        
        char_wavelets[char] = coeffs_resized
    
    return char_wavelets

def plot_char_wavelet(char, wavelet, ax):
    ax.plot(wavelet)
    ax.set_title(f"Wavelet for '{char}'")
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

def plot_word_wavelet(word, char_wavelets, output_dir):
    fig, axes = plt.subplots(len(word), 1, figsize=(10, 2*len(word)))
    fig.suptitle(f"Wavelets for '{word}'", fontsize=16)
    
    combined_wavelet = np.zeros(len(char_wavelets[word[0]]))
    for i, char in enumerate(word):
        wavelet = char_wavelets[char]
        combined_wavelet += wavelet
        axes[i].plot(wavelet)
        axes[i].set_title(f"'{char}'")
        axes[i].set_ylim(-1.1, 1.1)
        axes[i].axis('off')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'word_wavelets_{word}.png'))
    plt.close(fig)
    
    # Plot combined wavelet
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(combined_wavelet)
    ax.set_title(f"Combined Wavelet for '{word}'")
    ax.set_ylim(-len(word), len(word))
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'combined_wavelet_{word}.png'))
    plt.close(fig)

# Create output directory
output_dir = 'wavelet_output'
os.makedirs(output_dir, exist_ok=True)

# Generate wavelets for lowercase alphabet
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_wavelets = generate_char_wavelets(alphabet)

# Plot individual character wavelets
rows = 5
cols = 6
fig, axes = plt.subplots(rows, cols, figsize=(18, 15))
for i, (char, wavelet) in enumerate(char_wavelets.items()):
    ax = axes[i // cols, i % cols]
    plot_char_wavelet(char, wavelet, ax)

# Remove any unused subplots
for i in range(len(alphabet), rows * cols):
    fig.delaxes(axes[i // cols, i % cols])

fig.suptitle("Wavelets for Individual Characters", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'character_wavelets.png'))
plt.close(fig)

# Plot wavelets for a word
word = "hello"
plot_word_wavelet(word, char_wavelets, output_dir)

print(f"Plots have been saved in the '{output_dir}' directory.")