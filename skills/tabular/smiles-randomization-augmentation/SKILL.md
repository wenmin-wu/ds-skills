---
name: tabular-smiles-randomization-augmentation
description: >
  Augments molecular datasets by generating multiple randomized SMILES strings for the same molecule, exploiting SMILES non-uniqueness to multiply training samples.
---
# SMILES Randomization Augmentation

## Overview

A single molecule can be written as many valid SMILES strings depending on the atom traversal order — `CCO`, `OCC`, and `C(O)C` all represent ethanol. For sequence-based models (LSTM, Transformer) that process SMILES character-by-character, each randomized SMILES is a distinct training example with the same label. This effectively multiplies the dataset size by N without changing the chemistry. RDKit's `Chem.MolToSmiles(mol, doRandom=True)` generates these variants. Typical augmentation factors of 3-10x improve model generalization on small molecular datasets.

## Quick Start

```python
from rdkit import Chem
import numpy as np

def augment_smiles(smiles_list, labels, n_augments=3):
    """Generate randomized SMILES variants for data augmentation."""
    aug_smiles, aug_labels = [], []
    for smi, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        aug_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        aug_labels.append(label)
        for _ in range(n_augments):
            rand_smi = Chem.MolToSmiles(mol, doRandom=True)
            aug_smiles.append(rand_smi)
            aug_labels.append(label)
    return aug_smiles, np.array(aug_labels)

# 3x augmentation: 1000 molecules → 4000 training examples
train_smi, train_y = augment_smiles(df['SMILES'], df['target'], n_augments=3)
```

## Workflow

1. Parse each SMILES to an RDKit Mol object
2. Always include the canonical form as the base example
3. Generate N random SMILES variants with `doRandom=True`
4. Duplicate labels for each variant
5. Train sequence model on augmented dataset

## Key Decisions

- **n_augments**: 3-5 for moderate datasets; 10+ for very small (<500 molecules)
- **Canonical always included**: Ensures the standard representation is always in training
- **Train only**: Only augment training set — use canonical SMILES for validation/test
- **Best for**: Sequence models (LSTM, Transformer); fingerprint models see identical features regardless

## References

- [Extra Data with FS (Starting Point)](https://www.kaggle.com/code/alejandrolopezrincon/extra-data-with-fs-starting-point)
