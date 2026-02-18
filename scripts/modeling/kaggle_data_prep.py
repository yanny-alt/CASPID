#!/usr/bin/env python3
"""
kaggle_data_prep.py

Exports clean, processed data for Kaggle conditioning layer training.
Creates a single file with all necessary data.

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING"
KAGGLE_DIR = MODELING_DIR / "kaggle_export"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("EXPORTING DATA FOR KAGGLE")
print("="*70)

print("\n1. Loading data...")

data_file = MODELING_DIR / "full_modeling_dataset.csv"
df = pd.read_csv(data_file)
print(f"Loaded: {df.shape}")

set_b_file = MODELING_DIR / "04_consensus" / "set_b_expanded_structural.txt"
with open(set_b_file, 'r') as f:
    structural_features = [line.strip() for line in f if line.strip()]

trans_file = MODELING_DIR / "transcriptomic_features.txt"
with open(trans_file, 'r') as f:
    transcriptomic_features = [line.strip() for line in f if line.strip()]

print(f"Structural features: {len(structural_features)}")
print(f"Transcriptomic features: {len(transcriptomic_features)}")

print("\n2. Preparing features...")

y = df['LN_IC50'].values

X_struct_raw = df[structural_features].values
X_trans_raw = df[transcriptomic_features].values

imputer_struct = SimpleImputer(strategy='median')
X_struct = imputer_struct.fit_transform(X_struct_raw)

imputer_trans = SimpleImputer(strategy='median')
X_trans = imputer_trans.fit_transform(X_trans_raw)

print(f"Structural: {X_struct.shape}")
print(f"Transcriptomic: {X_trans.shape}")
print(f"Target: {y.shape}")

print("\n3. Creating stratification bins...")

discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

print(f"Stratification bins: {np.unique(y_binned, return_counts=True)}")

print("\n4. Creating export dataframe...")

struct_df = pd.DataFrame(X_struct, columns=[f'struct_{i}' for i in range(X_struct.shape[1])])
trans_df = pd.DataFrame(X_trans, columns=[f'trans_{i}' for i in range(X_trans.shape[1])])
target_df = pd.DataFrame({'target': y, 'strat_bin': y_binned})

export_df = pd.concat([struct_df, trans_df, target_df], axis=1)

print(f"Export dataframe: {export_df.shape}")

print("\n5. Saving files...")

export_file = KAGGLE_DIR / "caspid_conditioning_data.csv"
export_df.to_csv(export_file, index=False)
print(f"✓ Saved: {export_file}")
print(f"  Size: {export_file.stat().st_size / 1024 / 1024:.1f} MB")

feature_info = {
    'n_structural': len(structural_features),
    'n_transcriptomic': len(transcriptomic_features),
    'n_samples': len(y),
    'structural_names': structural_features,
    'transcriptomic_names': transcriptomic_features
}

import json
info_file = KAGGLE_DIR / "feature_info.json"
with open(info_file, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✓ Saved: {info_file}")

baseline_info = {
    'baseline_concatenation_r2': 0.8013,
    'structure_only_r2': 0.3905,
    'transcriptomics_only_r2': 0.2054,
    'target_improvement': 0.05,
    'success_threshold': 0.85
}

baseline_file = KAGGLE_DIR / "baseline_performance.json"
with open(baseline_file, 'w') as f:
    json.dump(baseline_info, f, indent=2)
print(f"✓ Saved: {baseline_file}")

print(f"\n{'='*70}")
print("EXPORT COMPLETE")
print(f"{'='*70}")

print(f"\nUPLOAD TO KAGGLE:")
print(f"  1. {export_file.name}")
print(f"  2. {info_file.name}")
print(f"  3. {baseline_file.name}")

print(f"\nTarget: R² > 0.8513 (baseline + 0.05)")
print("="*70)
