"""Produit les sorties de référence de GPT-2 small avec PyTorch.

Le programme Rust `gpt2` charge les mêmes poids, refait le même calcul et
compare ses résultats à ceux enregistrés ici.

Usage : python gpt2_reference.py [dossier_gpt2]

Le dossier doit contenir les fichiers de `openai-community/gpt2` :
config.json, model.safetensors, vocab.json, merges.txt, tokenizer.json.
Il reçoit `reference.safetensors`, qui contient :

- `tokens`      [T]     les identifiants de tokens de la phrase d'amorce
- `logits`      [T, V]  les logits de GPT-2 à chaque position
- `greedy`      [N]     la suite la plus probable, token après token

Dépendances : torch, transformers, safetensors, numpy.
"""

import sys

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROMPT = "The history of science shows that careful observation"
GREEDY_TOKENS = 20

directory = sys.argv[1] if len(sys.argv) > 1 else "../data/gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(directory)
model = GPT2LMHeadModel.from_pretrained(directory).eval()

tokens = tokenizer(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(tokens).logits[0]
    # Génération gloutonne : toujours le token le plus probable.
    sequence = tokens
    for _ in range(GREEDY_TOKENS):
        next_token = model(sequence).logits[0, -1].argmax()
        sequence = torch.cat([sequence, next_token.view(1, 1)], dim=1)
greedy = sequence[0, tokens.shape[1]:]

# Les identifiants sont enregistrés en f32, le seul type que lit le programme
# Rust. Ils sont inférieurs à 2^24, donc représentés exactement.
save_file(
    {
        "tokens": tokens[0].numpy().astype(np.float32),
        "logits": logits.numpy().astype(np.float32),
        "greedy": greedy.numpy().astype(np.float32),
    },
    f"{directory}/reference.safetensors",
)

print(f"amorce  : {PROMPT!r}")
print(f"tokens  : {tokens[0].tolist()}")
print(f"logits  : {tuple(logits.shape)}, max {logits.max():.4f}, min {logits.min():.4f}")
print(f"greedy  : {greedy.tolist()}")
print(f"suite   : {tokenizer.decode(greedy)!r}")
