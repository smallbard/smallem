"""Débit d'entraînement du même GPT avec PyTorch, sur la même carte.

C'est le point de comparaison de l'étape 9 : même architecture (pré-norme,
requête, clé et valeur séparées, sortie liée aux embeddings), même optimiseur,
même écrêtage, mêmes tailles de lot que `cargo oxide run --bin bench`.

    python scripts/pytorch_reference.py

Dépendance : torch avec CUDA.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    # nom : (vocabulaire, contexte, couches, têtes, dimension, lot, pas mesurés)
    "petit": (257, 64, 4, 4, 128, 32, 100),
    "grand": (257, 256, 6, 6, 384, 16, 20),
    "xl": (8192, 512, 8, 8, 512, 16, 10),
}
WARMUP_STEPS = 5


class Attention(nn.Module):
    def __init__(self, dim, heads, fused):
        super().__init__()
        self.heads, self.fused = heads, fused
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, dim = x.shape
        split = lambda v: v.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        if self.fused:
            # L'attention toute faite de PyTorch (noyau fusionné).
            mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Les mêmes opérations que le modèle Rust, une par une.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            mixed = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
        return self.proj(mixed.transpose(1, 2).reshape(batch, length, dim))


class Block(nn.Module):
    def __init__(self, dim, heads, fused):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, fused)
        self.expand, self.contract = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.contract(F.gelu(self.expand(self.ln2(x)), approximate="tanh"))


class Gpt(nn.Module):
    def __init__(self, vocab, context, layers, heads, dim, fused):
        super().__init__()
        self.tokens, self.positions = nn.Embedding(vocab, dim), nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, fused) for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, tokens, targets):
        x = self.tokens(tokens) + self.positions(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        logits = self.final_norm(x) @ self.tokens.weight.T
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def bench(name, mode, fused):
    vocab, context, layers, heads, dim, batch, steps = CONFIGS[name]
    # « f32 strict » interdit à PyTorch d'arrondir les produits en TF32 sur
    # les Tensor Cores, ce qu'il fait sinon dès qu'on le lui permet.
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.manual_seed(0)
    model = Gpt(vocab, context, layers, heads, dim, fused).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(WARMUP_STEPS + steps):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize()
            start = time.perf_counter()
        tokens = torch.randint(vocab, (batch, context), device="cuda")
        targets = torch.randint(vocab, (batch, context), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "bf16"):
            loss = model(tokens, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    attention = "attention fusionnée" if fused else "attention détaillée"
    print(
        f"{name:<6} {mode:<5} {attention:<20} "
        f"{steps * batch * context / seconds:>9.0f} tokens/s {1000 * seconds / steps:>7.1f} ms/pas"
    )


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0))
    for name in CONFIGS:
        for mode in ("f32", "tf32", "bf16"):
            for fused in (False, True):
                bench(name, mode, fused)
