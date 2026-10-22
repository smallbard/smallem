# smallem

Un modèle de langage de type GPT écrit en Rust à partir de zéro, avec des kernels GPU
écrits à la main grâce à [CUDA Rust](https://github.com/NVIDIA/cuda-rust) (cuda-oxide).

La série d'articles qui l'explique pas à pas est sur [smallbard.com](https://smallbard.com/articles/categorie/ia).

## Deux paquets

| Paquet | Contenu | Se compile avec |
|---|---|---|
| `smallem/` (ici) | tout sauf le GPU : texte, backend CPU, autograd, couches, modèles, entraînement, génération | `cargo`, en Rust stable |
| [`../smallem-cuda/`](../smallem-cuda/README.md) | le backend GPU et ses kernels, les programmes qui s'en servent, les tests qui comparent GPU et CPU | `cargo oxide`, avec la nightly épinglée |

`smallem-cuda` dépend de `smallem`, jamais l'inverse. Ce paquet-ci n'a besoin ni d'une
carte, ni du CUDA Toolkit.

## Compiler et tester

```bash
cargo test                    # 481 tests, quelques secondes, sans GPU
cargo clippy --all-targets
```

## Les programmes

| Programme | Rôle |
|---|---|
| `prepare` | fichiers Parquet de FineWeb-Edu → fichiers de tokens |
| `bpe` | apprend une table de fusions BPE |
| `bigram` | modèle bigramme par comptage |
| `fit` | ajuste les lois d'échelle sur les résultats de `sweep`, et prédit |
| `cpubench` | ce que vaut le processeur, et ce qu'en tire le backend `Fast` |
| `remote` | les missions d'agent jouées par un grand modèle servi en local (API de chat d'OpenAI) |
| `ngrams` | compte les bigrammes et les trigrammes du corpus : couverture, collisions du hachage, modèles par comptage |

Chaque programme décrit ses options en tête de son fichier, dans `src/bin/`. Ceux qui
entraînent et génèrent sont dans `smallem-cuda`.

## Un entraînement de bout en bout

Les données sont attendues dans `../data/`.

```bash
# 1. Ici : apprendre le tokeniseur sur 100 Mo de texte, puis tokeniser le fichier entier.
cargo run --release --bin bpe
cargo run --release --bin prepare -- --tokenizer ../data/bpe/tokenizer.bpe

# 2. Dans ../smallem-cuda : entraîner. Ajouter --resume à la même commande pour reprendre.
cargo oxide run --bin train -- --config xl --arch modern --data ../data/bpe --batch 16 \
    --lr 6e-4 --steps 6000 --eval-every 250 --mixed

#    Variantes de l'étape 10, à ajouter à la commande : --attention mla, --experts
#    (avec --balance none|bias|loss), --mtp, --optimizer muon.
#    Variantes de l'étape 11 : --optimizer muon:short (et period=3, norm, boost=5),
#    --passes 2, --layers 16, --own-keys 4. Étape 12 : --context 4096.
#    Mémoire de n-grammes : --memory 262144 (lignes par table), --memory-storage host.

# 3. Générer : le texte s'affiche à mesure qu'il s'écrit.
cargo oxide run --bin generate -- --model ../data/bpe/gpt-xl.safetensors \
    --prompt "To teach children how to read, teachers" --temperature 0.8 --top-k 40
```

## Organisation

```text
src/
  corpus.rs, data.rs, tokenizer.rs   le texte : lecture, tokens, lots
  backend/                           les tenseurs
    mod.rs                           les primitives, et le trait Backend
    ops/                             une opération par fichier
    cpu.rs                           le backend CPU : la référence, simple et lente
    fast/                            le backend du processeur qui calcule : threads, produit
                                     par blocs, AVX-512, attention en tuiles, poids en octets
    profile.rs                       un backend qui chronomètre un autre
    float.rs, shape.rs               ce que le CPU et le GPU partagent
  autograd.rs                        la rétropropagation
  nn/                                les couches : paramètres, normalisations,
                                     attention classique et latente, feed-forward,
                                     mélange d'experts, prédiction multi-token, bloc
  nn/lora.rs                         les adaptateurs LoRA
  nn/memory.rs                       la mémoire de n-grammes : table dense, creuse ou en
                                     mémoire centrale, porte et convolution
  ngram.rs                           le hachage des n-grammes, et leurs comptes
  model.rs, mlp.rs                   les modèles
  optim.rs                           les optimiseurs : SGD, AdamW, Muon
  training/                          un pas d'entraînement, puis la boucle complète
  generation.rs, sampling.rs         la génération, seule ou par lots, et le tirage des tokens
  cache.rs                           le cache des clés et des valeurs, par pages
  quantize.rs                        poids int8 et int4, arrondi simple et GPTQ
  speculative.rs                     le décodage spéculatif
  chat.rs, tasks.rs                  les conversations, et les tâches vérifiables
  post_training.rs                   évaluation, fine-tuning, DPO, GRPO
  agent/                             un agent : la boucle, les outils, les missions,
                                     un grand modèle servi en local
  flops.rs, scaling.rs               le compte des opérations, les lois d'échelle
  io.rs, gpt2.rs                     les fichiers de poids
  cli.rs                             la lecture des options
  bin/                               prepare, bpe, bigram, fit, cpubench, ngrams, remote
```
