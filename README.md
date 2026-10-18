# Smallem

Un modèle de langage de type GPT écrit en Rust à partir de zéro : le tokenizer,
l'autograd, les couches, l'entraînement, la génération, et des kernels GPU écrits à la main
grâce à [CUDA Rust](https://github.com/NVIDIA/cuda-rust) (cuda-oxide). Aucune bibliothèque
d'apprentissage : les seules dépendances lisent Parquet et JSON, tirent des nombres
aléatoires, et parlent à la carte.

La série d'articles qui l'explique pas à pas est sur
[smallbard.com](https://smallbard.com/articles/categorie/ia).

## Contenu

- un tokenizer BPE, et la préparation du corpus FineWeb-Edu
- un autograd et des modèles GPT, classiques ou modernes : attention latente, mélange
  d'experts, prédiction multi-token
- les optimiseurs SGD, AdamW et Muon, la précision mixte, les lois d'échelle
- la génération : cache des clés et des valeurs par pages, lots, quantification int8 et
  int4, décodage spéculatif
- le post-entraînement : fine-tuning supervisé, LoRA, DPO, GRPO
- trois backends : un backend de référence sur le processeur, simple et lent, un backend
  processeur optimisé (AVX-512, produit par blocs, attention en tuiles) et un backend GPU.

## Deux paquets

| Paquet | Contenu | Se compile avec |
|---|---|---|
| [`smallem/`](smallem/README.md) | tout sauf le GPU : texte, backends du processeur, autograd, couches, modèles, entraînement, génération | `cargo`, en Rust stable |
| [`smallem-cuda/`](smallem-cuda/README.md) | le backend GPU et ses kernels, les programmes qui entraînent et génèrent, les tests qui comparent GPU et CPU | `cargo oxide`, avec la nightly épinglée |

`smallem-cuda` dépend de `smallem`, jamais l'inverse. Chaque paquet décrit dans son README
comment le compiler, le tester et lancer ses programmes ; le README de `smallem` donne un
entraînement de bout en bout.

Les données (corpus, tokens, modèles enregistrés) sont attendues dans `data/`, à côté des
deux paquets.

## Licence

[MIT](LICENSE).
