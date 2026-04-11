---
name: tabular-rdkit-molecular-descriptors
description: >
  Computes all numeric RDKit molecular descriptors from SMILES strings, filtering out NaN, constant, and infinite values to produce a clean feature matrix.
---
# RDKit Molecular Descriptors

## Overview

RDKit provides 200+ molecular descriptors — physicochemical properties (logP, molecular weight, TPSA), topological indices (Wiener, Balaban), and fragment counts (aromatic rings, H-bond donors/acceptors). Computing all descriptors from SMILES gives a rich feature set for tabular ML without domain-specific feature engineering. The key is robust filtering: some descriptors return NaN or infinity for certain molecules, and some are constant across datasets. After cleanup, this typically yields 150-180 usable numeric features.

## Quick Start

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

def compute_descriptors(smiles):
    """Compute all RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(Descriptors.descList)
    return [func(mol) for name, func in Descriptors.descList]

desc_names = [name for name, _ in Descriptors.descList]
desc_matrix = [compute_descriptors(s) for s in df['SMILES']]
features = pd.DataFrame(desc_matrix, columns=desc_names)

# Clean up
features = features.replace([np.inf, -np.inf], np.nan)
features = features.dropna(axis=1, thresh=int(0.9 * len(features)))  # drop >10% NaN cols
features = features.loc[:, features.nunique() > 1]  # drop constant cols
features = features.fillna(features.median())
```

## Workflow

1. Parse each SMILES to RDKit Mol object
2. Compute all descriptors from `Descriptors.descList`
3. Replace inf values with NaN
4. Drop columns with >10% missing or zero variance
5. Impute remaining NaNs with column median
6. Use as features for LGBM, XGBoost, etc.

## Key Decisions

- **Descriptor subset**: Use all 200+ by default; prune with feature importance after first model
- **AUTOCORR2D**: These 192 autocorrelation descriptors can dominate — cap at 10-20 or drop entirely
- **Combining**: Stack with Morgan fingerprints for complementary representations
- **Scaling**: Tree models don't need scaling; for neural nets or SVR, StandardScaler first

## References

- [NeurIPS Baseline + External Data](https://www.kaggle.com/code/dmitryuarov/neurips-baseline-external-data)
