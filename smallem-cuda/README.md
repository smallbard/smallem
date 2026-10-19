# smallem-cuda

Le backend GPU de [smallem](../smallem/README.md) : les primitives et les opérations de
`smallem::backend`, exécutées par des kernels CUDA écrits en Rust grâce à
[CUDA Rust](https://github.com/NVIDIA/cuda-rust) (cuda-oxide). Avec lui, les programmes qui
entraînent et génèrent, et les tests qui comparent chaque kernel au backend CPU.

## Prérequis

- une carte NVIDIA Ampere ou plus récente, pilote 580 ou plus ;
- le CUDA Toolkit 13.0 ;
- Clang 21 (`libclang-21-dev`), pour `bindgen` ;
- `cargo-oxide`, installé avec la nightly épinglée dans `rust-toolchain.toml`.

```bash
cargo oxide doctor
```

## Compiler et tester

Tout se compile avec `cargo oxide`, pas avec `cargo` seul : c'est lui qui traduit les
kernels pour la carte.

```bash
cargo oxide build
cargo oxide test              # 143 tests, sur la carte
cargo clippy --all-targets    # l'analyse, elle, se passe de cargo oxide
```

Tous les tests sont des tests d'intégration, dans `tests/` : la bibliothèque n'a pas de
tests unitaires (`test = false` dans `Cargo.toml`). Elle n'est ainsi compilée qu'une fois,
et son module de kernels écrit une seule fois.

Le test `tests/memory.rs` remplit la mémoire de la carte : ne pas le lancer pendant un
entraînement.

## Les programmes

| Programme | Rôle |
|---|---|
| `train` | entraîne un GPT |
| `generate` | génère du texte avec un modèle enregistré ; avec `--chat`, répond à une question |
| `mlp` | modèle à contexte fixe |
| `gpt2` | compare GPT-2 small chargé dans ce code à PyTorch |
| `bench` | débit d'entraînement en tokens par seconde |
| `profile` | temps d'un pas par opération |
| `matmul` | débit des produits matriciels, y compris pour la génération et les poids quantifiés |
| `throughput` | débit de génération, seule ou par lots |
| `quantize` | quantifie un modèle en int8 ou int4, et mesure la perte |
| `speculate` | décodage spéculatif : brouillon ou prédiction multi-token |
| `serve` | simulateur de service : politiques statique, continue, par pages |
| `evaluate` | précision d'un modèle sur les tâches vérifiables |
| `sft`, `dpo`, `rl` | fine-tuning supervisé, DPO, GRPO |
| `sweep` | lance une grille d'entraînements et note les résultats |
| `mfu` | rendement en opérations de la carte, par taille de modèle |
| `noise` | bruit du gradient et taille de lot critique |
| `recall` | ce qu'un modèle doit à sa mémoire de n-grammes : la perte, mémoire coupée ou non, selon la fréquence du contexte |

Chaque programme décrit ses options en tête de son fichier, dans `src/bin/`. `train`,
`generate` et `throughput` tournent aussi sur le processeur (`--cpu`, avec le backend
`Fast` ; `--cpu reference` pour le backend de référence), `mlp` et `gpt2` sur le backend de
référence (`--cpu`).

## Organisation

```text
src/
  kernels.rs     le code qui tourne sur la carte
  cuda.rs        le backend : tenseurs sur la carte, lancement des kernels
  ops.rs         les opérations de smallem::backend::ops sur GPU
  profile.rs     un backend GPU qui chronomètre
  bin/           les programmes
tests/           GPU comparé au CPU
scripts/         références calculées avec PyTorch
```
