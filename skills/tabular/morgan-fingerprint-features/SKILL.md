---
name: tabular-morgan-fingerprint-features
description: >
  Converts molecular SMILES strings to fixed-length Morgan fingerprint bit vectors using RDKit for use as tabular ML features.
---
# Morgan Fingerprint Features

## Overview

Morgan fingerprints (Extended-Connectivity Fingerprints / ECFP) encode the structural neighborhood of each atom in a molecule as a fixed-length binary vector. Each bit represents whether a particular circular substructure of radius R exists in the molecule. This converts variable-length SMILES strings into fixed-size feature vectors suitable for any tabular ML model (LGBM, XGBoost, random forest). Morgan FPs are the most widely used molecular representation in cheminformatics — they capture functional groups, ring systems, and local topology in a compact, hashable form.

## Quick Start

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# Vectorize a dataset
X = np.vstack([smiles_to_morgan(s) for s in df['SMILES']])
# X shape: (n_samples, 1024) — ready for LGBM/XGBoost
```

## Workflow

1. Parse SMILES string to RDKit Mol object
2. Generate Morgan fingerprint with chosen radius and bit count
3. Convert to numpy array
4. Stack into feature matrix for downstream ML

## Key Decisions

- **Radius**: 2 (ECFP4) is standard; 3 (ECFP6) captures larger substructures but is sparser
- **n_bits**: 1024 is default; 2048 reduces hash collisions for diverse datasets
- **Count vs binary**: `GetMorganFingerprint` returns counts; `AsBitVect` returns binary — binary is usually sufficient
- **Combining**: Stack with RDKit descriptors or graph features for richer representations

## References

- [NeurIPS 2025 Open Polymer Challenge Tutorial](https://www.kaggle.com/code/alexliu99/neurips-2025-open-polymer-challenge-tutorial)
