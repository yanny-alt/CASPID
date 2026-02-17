#!/usr/bin/env python3
"""
02_feature_selection_mi.py

Mutual Information feature selection on STRUCTURAL (docking) features.
Captures non-linear relationships between features and IC50.

CRITICAL: Only selects from 92 docking features, NOT transcriptomic features.

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING" / "braf"
MI_DIR = MODELING_DIR / "02_mi"
MI_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = MI_DIR / f"mi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("MUTUAL INFORMATION FEATURE SELECTION - STRUCTURAL FEATURES ONLY")
print("="*70)

# ============================================
# STEP 1: LOAD DATA
# ============================================

log("\n1. Loading merged dataset...")

data_file = MODELING_DIR / "full_modeling_dataset.csv"

if not data_file.exists():
    log(f"✗ ERROR: Dataset not found: {data_file}")
    log("  Please run 00_merge_docking_transcriptomics.py first")
    exit(1)

df = pd.read_csv(data_file)
log(f"✓ Loaded dataset: {df.shape}")

# Load feature lists
docking_features_file = MODELING_DIR / "docking_features.txt"

if not docking_features_file.exists():
    log(f"✗ ERROR: Docking features list not found")
    exit(1)

with open(docking_features_file, 'r') as f:
    docking_features = [line.strip() for line in f if line.strip()]

log(f"✓ Loaded {len(docking_features)} docking feature names")

# ============================================
# STEP 2: PREPARE DATA
# ============================================

log("\n2. Preparing data for MI calculation...")

# Extract target
y = df['LN_IC50'].values
log(f"✓ Target shape: {y.shape}")
log(f"  Mean: {y.mean():.3f}")
log(f"  Std: {y.std():.3f}")

# Extract ONLY docking features
missing_features = [f for f in docking_features if f not in df.columns]
if missing_features:
    log(f"✗ ERROR: Missing features in dataset:")
    for feat in missing_features[:10]:
        log(f"  - {feat}")
    exit(1)

X = df[docking_features].values
log(f"✓ Feature matrix shape: {X.shape}")
log(f"  Samples: {X.shape[0]}")
log(f"  Features: {X.shape[1]}")

# Check for missing values
missing_mask = np.isnan(X)
missing_per_feature = missing_mask.sum(axis=0)

log(f"\n✓ Missing value check:")
log(f"  Features with missing: {(missing_per_feature > 0).sum()}")
log(f"  Max missing per feature: {missing_per_feature.max()} ({missing_per_feature.max()/X.shape[0]*100:.1f}%)")

# Impute missing values with median
if missing_mask.any():
    log(f"\n⚠️  Imputing missing values with median...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    log(f"✓ Imputation complete")

# Handle infinite values
inf_mask = np.isinf(X)
if inf_mask.any():
    log(f"\n⚠️  WARNING: {inf_mask.sum()} infinite values detected")
    log(f"  Replacing with feature max/min...")
    X = np.nan_to_num(X, posinf=np.nanmax(X[~inf_mask]), neginf=np.nanmin(X[~inf_mask]))

# ============================================
# STEP 3: CALCULATE MUTUAL INFORMATION
# ============================================

log("\n3. Calculating Mutual Information scores...")
log("  (This may take 5-10 minutes)")

start_time = datetime.now()

try:
    mi_scores = mutual_info_regression(
        X, 
        y,
        n_neighbors=3,  # Captures local structure
        random_state=42
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\n✓ MI calculation completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
except Exception as e:
    log(f"\n✗ ERROR during MI calculation:")
    log(f"  {str(e)}")
    import traceback
    log(traceback.format_exc())
    exit(1)

# ============================================
# STEP 4: ANALYZE RESULTS
# ============================================

log("\n4. Analyzing MI scores...")

# Create results dataframe
# Ensure arrays match length
mi_results = pd.DataFrame({
    'feature': docking_features[:len(mi_scores)],
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Statistics
log(f"\n✓ MI score statistics:")
log(f"  Mean: {mi_scores.mean():.6f}")
log(f"  Median: {np.median(mi_scores):.6f}")
log(f"  Max: {mi_scores.max():.6f}")
log(f"  Min: {mi_scores.min():.6f}")
log(f"  Std: {mi_scores.std():.6f}")

# Count non-zero scores
non_zero = (mi_scores > 0).sum()
log(f"\n✓ Non-zero MI scores: {non_zero}/{len(mi_scores)}")

# Check for very low scores
threshold_low = 0.001
low_scores = (mi_scores < threshold_low).sum()
log(f"  Features with MI < {threshold_low}: {low_scores}")

# ============================================
# STEP 5: SELECT TOP FEATURES
# ============================================

log("\n5. Selecting top 50 features by MI score...")

# Select top 50
n_select = 50
if len(mi_results) < n_select:
    n_select = len(mi_results)
    log(f"⚠️  Only {n_select} features available (less than 50)")

top_features = mi_results.head(n_select)

log(f"\n✓ Selected top {n_select} features")
log(f"  Top feature MI: {top_features.iloc[0]['mi_score']:.6f}")
log(f"  50th feature MI: {top_features.iloc[-1]['mi_score']:.6f}")

log(f"\n✓ Top 20 features by MI score:")
for i, row in top_features.head(20).iterrows():
    log(f"  {row.name+1}. {row['feature']}: {row['mi_score']:.6f}")

if len(top_features) > 20:
    log(f"  ... and {len(top_features) - 20} more")

# ============================================
# STEP 6: COMPARE WITH BORUTA (if available)
# ============================================

log("\n6. Comparing with Boruta results...")

boruta_file = MODELING_DIR / "01_boruta" / "boruta_confirmed_features.txt"

if boruta_file.exists():
    with open(boruta_file, 'r') as f:
        boruta_features = set(line.strip() for line in f if line.strip())
    
    log(f"✓ Loaded {len(boruta_features)} Boruta-confirmed features")
    
    # Check overlap
    mi_selected = set(top_features['feature'].values)
    overlap = boruta_features & mi_selected
    
    log(f"\n✓ Overlap analysis:")
    log(f"  Boruta selected: {len(boruta_features)}")
    log(f"  MI selected: {len(mi_selected)}")
    log(f"  Overlap: {len(overlap)}")
    
    if len(overlap) > 0:
        log(f"\n✓ Features selected by BOTH methods:")
        for feat in sorted(overlap):
            mi_score = mi_results[mi_results['feature'] == feat]['mi_score'].values[0]
            log(f"  - {feat} (MI: {mi_score:.6f})")
    else:
        log(f"\n⚠️  NO overlap between Boruta and MI!")
        log(f"  This suggests different selection criteria")
else:
    log("⚠️  Boruta results not found, skipping comparison")

# ============================================
# STEP 7: SAVE RESULTS
# ============================================

log("\n7. Saving results...")

# Save top features list
selected_file = MI_DIR / "mi_selected_features.txt"
with open(selected_file, 'w') as f:
    f.write('\n'.join(top_features['feature'].values))
log(f"✓ Saved selected features: {selected_file}")

# Save all MI scores
all_scores_file = MI_DIR / "mi_all_scores.csv"
mi_results.to_csv(all_scores_file, index=False)
log(f"✓ Saved all MI scores: {all_scores_file}")

# Save top features with scores
top_scores_file = MI_DIR / "mi_top_features.csv"
top_features.to_csv(top_scores_file, index=False)
log(f"✓ Saved top features: {top_scores_file}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': X.shape[0],
    'n_features_total': X.shape[1],
    'n_selected': len(top_features),
    'mi_mean': float(mi_scores.mean()),
    'mi_max': float(mi_scores.max()),
    'mi_min': float(mi_scores.min()),
    'runtime_seconds': elapsed,
    'n_neighbors': 3
}

import json
summary_file = MI_DIR / "mi_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("MUTUAL INFORMATION FEATURE SELECTION COMPLETE")
log("="*70)

log(f"\n📊 RESULTS SUMMARY:")
log(f"   Input features: {X.shape[1]}")
log(f"   Selected features (top 50): {len(top_features)}")
log(f"   MI score range: {mi_scores.min():.6f} to {mi_scores.max():.6f}")
log(f"   Runtime: {elapsed:.1f} seconds")

log(f"\n📁 OUTPUT FILES:")
log(f"   Selected features: {selected_file}")
log(f"   All scores: {all_scores_file}")
log(f"   Top features: {top_scores_file}")
log(f"   Summary: {summary_file}")

log(f"\n✅ Next step: Run 03_feature_selection_shap.py")
log("="*70)
