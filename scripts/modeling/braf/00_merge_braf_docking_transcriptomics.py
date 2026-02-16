#!/usr/bin/env python3
"""
00_merge_docking_transcriptomics.py

CRITICAL SCRIPT: Merges docking features with integrated dataset.
This creates the full modeling dataset for feature selection.

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES" / "06_combined"
PROCESSED_DIR = PROJECT_ROOT / "DATA" / "PROCESSED"
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING" / "braf"
MODELING_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = MODELING_DIR / f"00_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("MERGE DOCKING + TRANSCRIPTOMICS")
print("="*70)

# ============================================
# STEP 1: LOAD DOCKING FEATURES
# ============================================

log("\n1. Loading docking features...")

docking_file = FEATURES_DIR / "clean_docking_features.csv"

if not docking_file.exists():
    log(f"✗ ERROR: Docking file not found: {docking_file}")
    exit(1)

docking_df = pd.read_csv(docking_file)

log(f"✓ Loaded docking features")
log(f"  Shape: {docking_df.shape}")
log(f"  Columns: {len(docking_df.columns)}")
log(f"  Unique drugs: {docking_df['ligand'].nunique()}")

# Filter for BRAF protein only
log("\n  Filtering for BRAF protein...")
docking_df = docking_df[docking_df['protein'] == 'BRAF'].copy()
log(f"  BRAF docking features: {len(docking_df)} poses")

# Identify feature columns (exclude metadata)
metadata_cols_docking = ['job_id', 'protein', 'ligand']
docking_feature_cols = [c for c in docking_df.columns if c not in metadata_cols_docking]

log(f"  Docking features: {len(docking_feature_cols)}")

# Standardize drug names to lowercase for matching
docking_df['ligand_lower'] = docking_df['ligand'].str.lower().str.strip()

log(f"\nDocking drugs (lowercase):")
for drug in sorted(docking_df['ligand_lower'].unique()):
    log(f"  - {drug}")

# ============================================
# STEP 2: LOAD INTEGRATED DATASET
# ============================================

log("\n2. Loading integrated dataset...")

integrated_file = PROCESSED_DIR / "caspid_braf_integrated_dataset.csv"

if not integrated_file.exists():
    log(f"✗ ERROR: Integrated dataset not found: {integrated_file}")
    exit(1)

integrated_df = pd.read_csv(integrated_file)

log(f"✓ Loaded integrated dataset")
log(f"  Shape: {integrated_df.shape}")
log(f"  Rows (measurements): {len(integrated_df)}")
log(f"  Unique cell lines: {integrated_df['CELL_LINE_NAME'].nunique()}")
log(f"  Unique drugs: {integrated_df['DRUG_NAME'].nunique()}")

# Identify gene expression columns (they have numbers in parentheses)
gene_cols = [c for c in integrated_df.columns if '(' in c and ')' in c]

log(f"  Gene expression features: {len(gene_cols)}")

# Standardize drug names to lowercase for matching
integrated_df['drug_lower'] = integrated_df['DRUG_NAME'].str.lower().str.strip()

log(f"\nIntegrated drugs (sample):")
for drug in sorted(integrated_df['drug_lower'].unique())[:10]:
    log(f"  - {drug}")
log(f"  ... and {integrated_df['drug_lower'].nunique() - 10} more")

# ============================================
# STEP 3: CHECK DRUG OVERLAP
# ============================================

log("\n3. Checking drug name overlap...")

docking_drugs = set(docking_df['ligand_lower'].unique())
integrated_drugs = set(integrated_df['drug_lower'].unique())

overlap = docking_drugs & integrated_drugs
docking_only = docking_drugs - integrated_drugs
integrated_only = integrated_drugs - docking_drugs

log(f"\n✓ Drug overlap analysis:")
log(f"  Drugs in docking: {len(docking_drugs)}")
log(f"  Drugs in integrated: {len(integrated_drugs)}")
log(f"  Overlap: {len(overlap)}")

if len(overlap) == 0:
    log("\n✗ ERROR: No drug overlap! Cannot merge.")
    log("\nDocking drugs:")
    for d in sorted(docking_drugs):
        log(f"  - {d}")
    log("\nIntegrated drugs (sample):")
    for d in sorted(list(integrated_drugs)[:15]):
        log(f"  - {d}")
    exit(1)

log(f"\n✓ Overlapping drugs:")
for drug in sorted(overlap):
    log(f"  - {drug}")

if docking_only:
    log(f"\n⚠️  Drugs in docking but NOT in integrated ({len(docking_only)}):")
    for drug in sorted(docking_only):
        log(f"  - {drug}")
        
# Filter for HIGH confidence BRAF drugs only
log("\n  Filtering for HIGH confidence BRAF drugs...")
high_conf_braf_drugs = ['dabrafenib', 'sb590885']
integrated_df = integrated_df[integrated_df['drug_lower'].isin(high_conf_braf_drugs)].copy()
log(f"  BRAF samples (2 drugs): {len(integrated_df)}")       

# ============================================
# STEP 4: MERGE DATASETS
# ============================================

log("\n4. Merging datasets...")

# Merge on lowercase drug name
merged_df = integrated_df.merge(
    docking_df,
    left_on='drug_lower',
    right_on='ligand_lower',
    how='inner',
    suffixes=('', '_docking')
)

log(f"✓ Merge complete")
log(f"  Merged shape: {merged_df.shape}")
log(f"  Rows: {len(merged_df)}")

if len(merged_df) == 0:
    log("✗ ERROR: Merge resulted in 0 rows!")
    exit(1)

# Check merge quality
expected_min = len(integrated_df[integrated_df['drug_lower'].isin(overlap)])
log(f"  Expected rows: ~{expected_min}")
log(f"  Actual rows: {len(merged_df)}")

if len(merged_df) < expected_min * 0.9:
    log("⚠️  WARNING: Fewer rows than expected after merge")

# ============================================
# STEP 5: SELECT AND ORGANIZE COLUMNS
# ============================================

log("\n5. Organizing columns...")

# Target variable
target_col = 'LN_IC50'

# Metadata columns to keep
metadata_cols = [
    'COSMIC_ID',
    'CELL_LINE_NAME', 
    'DRUG_NAME',
    'protein',
    'job_id'
]

# All columns to include in final dataset
final_cols = metadata_cols + [target_col] + gene_cols + docking_feature_cols

# Verify all columns exist
missing_cols = [c for c in final_cols if c not in merged_df.columns]
if missing_cols:
    log(f"\n✗ ERROR: Missing columns after merge:")
    for col in missing_cols:
        log(f"  - {col}")
    exit(1)

# Select final columns
final_df = merged_df[final_cols].copy()

log(f"✓ Final dataset organized")
log(f"  Shape: {final_df.shape}")
log(f"  Metadata columns: {len(metadata_cols)}")
log(f"  Target: 1 column (LN_IC50)")
log(f"  Gene expression features: {len(gene_cols)}")
log(f"  Docking features: {len(docking_feature_cols)}")
log(f"  Total features: {len(gene_cols) + len(docking_feature_cols)}")

# ============================================
# STEP 6: QUALITY CHECKS
# ============================================

log("\n6. Quality control checks...")

# Check for missing values
missing_counts = final_df.isnull().sum()
cols_with_missing = missing_counts[missing_counts > 0]

if len(cols_with_missing) > 0:
    log(f"\n⚠️  Columns with missing values: {len(cols_with_missing)}")
    log(f"  Showing top 10:")
    for col, count in cols_with_missing.head(10).items():
        pct = count / len(final_df) * 100
        log(f"    {col}: {count} ({pct:.1f}%)")
else:
    log(f"✓ No missing values")

# Check target variable distribution
log(f"\n✓ Target variable (LN_IC50) statistics:")
log(f"  Mean: {final_df[target_col].mean():.3f}")
log(f"  Std: {final_df[target_col].std():.3f}")
log(f"  Min: {final_df[target_col].min():.3f}")
log(f"  Max: {final_df[target_col].max():.3f}")
log(f"  Missing: {final_df[target_col].isnull().sum()}")

# Check docking feature variation
log(f"\n✓ Docking feature variation check:")
docking_stds = final_df[docking_feature_cols].std()
zero_var = docking_stds[docking_stds == 0]

if len(zero_var) > 0:
    log(f"  ⚠️  Zero variance features: {len(zero_var)}")
else:
    log(f"  ✓ All features have variance")

# Per-drug statistics
log(f"\n✓ Per-drug statistics:")
per_drug = final_df.groupby('DRUG_NAME').size()
log(f"  Drugs: {len(per_drug)}")
log(f"  Samples per drug (mean): {per_drug.mean():.0f}")
log(f"  Samples per drug (min): {per_drug.min()}")
log(f"  Samples per drug (max): {per_drug.max()}")

# Per-cell line statistics
log(f"\n✓ Per-cell line statistics:")
per_cell = final_df.groupby('CELL_LINE_NAME').size()
log(f"  Cell lines: {len(per_cell)}")
log(f"  Samples per cell line (mean): {per_cell.mean():.1f}")

# ============================================
# STEP 7: SAVE RESULTS
# ============================================

log("\n7. Saving merged dataset...")

output_file = MODELING_DIR / "full_modeling_dataset.csv"
final_df.to_csv(output_file, index=False)

log(f"✓ Saved: {output_file}")
log(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

# Save feature lists for reference
log("\n8. Saving feature lists...")

# Transcriptomic features
trans_features_file = MODELING_DIR / "transcriptomic_features.txt"
with open(trans_features_file, 'w') as f:
    f.write('\n'.join(gene_cols))
log(f"✓ Saved transcriptomic features: {trans_features_file}")

# Docking features
docking_features_file = MODELING_DIR / "docking_features.txt"
with open(docking_features_file, 'w') as f:
    f.write('\n'.join(docking_feature_cols))
log(f"✓ Saved docking features: {docking_features_file}")

# Summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': len(final_df),
    'n_drugs': final_df['DRUG_NAME'].nunique(),
    'n_cell_lines': final_df['CELL_LINE_NAME'].nunique(),
    'n_transcriptomic_features': len(gene_cols),
    'n_docking_features': len(docking_feature_cols),
    'n_total_features': len(gene_cols) + len(docking_feature_cols),
    'target_mean': float(final_df[target_col].mean()),
    'target_std': float(final_df[target_col].std())
}

import json
summary_file = MODELING_DIR / "merge_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("MERGE COMPLETE - READY FOR FEATURE SELECTION")
log("="*70)

log(f"\n📊 FINAL DATASET SUMMARY:")
log(f"   Samples: {len(final_df):,}")
log(f"   Drugs: {final_df['DRUG_NAME'].nunique()}")
log(f"   Cell lines: {final_df['CELL_LINE_NAME'].nunique()}")
log(f"   Transcriptomic features: {len(gene_cols)}")
log(f"   Docking features: {len(docking_feature_cols)}")
log(f"   Total features: {len(gene_cols) + len(docking_feature_cols)}")

log(f"\n📁 OUTPUT FILES:")
log(f"   Main dataset: {output_file}")
log(f"   Transcriptomic list: {trans_features_file}")
log(f"   Docking list: {docking_features_file}")
log(f"   Summary: {summary_file}")

log(f"\n✅ Next step: Run 01_feature_selection_boruta.py")
log("="*70)
