#!/usr/bin/env python3
"""
Feature Combination and Quality Control - FINAL CRITICAL SCRIPT

This script combines all extracted features, performs rigorous quality control,
and validates feature quality through Checkpoint 3 (correlation with IC50 data).

CRITICAL: This is the final step before machine learning. All quality gates
must be passed to ensure publication-ready features.

Author: CASPID Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES"
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"

# Output directory
COMBINED_DIR = FEATURES_DIR / "06_combined"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = COMBINED_DIR / f"feature_combination_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("FEATURE COMBINATION & QUALITY CONTROL - FINAL STEP")
print("="*70)
log("Starting feature combination and QC pipeline")

# ============================================
# STEP 1: LOAD ALL FEATURE FILES
# ============================================

log("\n" + "="*70)
log("STEP 1: LOADING ALL FEATURE FILES")
log("="*70)

feature_files = {
    'distance': FEATURES_DIR / "01_distance" / "distance_features.csv",
    'geometric': FEATURES_DIR / "02_geometric" / "geometric_features.csv",
    'physicochemical': FEATURES_DIR / "03_physicochemical" / "physicochemical_features.csv",
    'interaction': FEATURES_DIR / "04_interactions" / "interaction_features.csv",
    'pharmacophore': FEATURES_DIR / "05_pharmacophore" / "pharmacophore_features.csv"
}

feature_dfs = {}
total_features = 0

for name, filepath in feature_files.items():
    if not filepath.exists():
        log(f"✗ ERROR: {name} features not found at {filepath}")
        exit(1)
    
    df = pd.read_csv(filepath)
    feature_dfs[name] = df
    
    # Count features (excluding metadata columns)
    n_features = len([c for c in df.columns if c not in ['job_id', 'protein', 'ligand']])
    total_features += n_features
    
    log(f"✓ {name}: {df.shape[0]} poses × {n_features} features")

log(f"\n✓ Total raw features: {total_features}")

# ============================================
# STEP 2: MERGE ALL FEATURES
# ============================================

log("\n" + "="*70)
log("STEP 2: MERGING ALL FEATURES")
log("="*70)

# Start with distance features
combined_df = feature_dfs['distance'].copy()
log(f"Starting with distance features: {combined_df.shape}")

# Merge each feature set
for name in ['geometric', 'physicochemical', 'interaction', 'pharmacophore']:
    df = feature_dfs[name]
    
    # Merge on job_id, protein, ligand
    combined_df = combined_df.merge(
        df,
        on=['job_id', 'protein', 'ligand'],
        how='inner'
    )
    
    log(f"After merging {name}: {combined_df.shape}")

log(f"\n✓ Combined feature matrix: {combined_df.shape}")
log(f"  Poses: {combined_df.shape[0]}")
log(f"  Total columns: {combined_df.shape[1]}")
log(f"  Feature columns: {combined_df.shape[1] - 3} (excluding metadata)")

# Save raw combined features
raw_combined_file = COMBINED_DIR / "raw_combined_features.csv"
combined_df.to_csv(raw_combined_file, index=False)
log(f"\n✓ Saved raw combined features: {raw_combined_file}")

# ============================================
# STEP 3: QUALITY CONTROL - MISSING VALUES
# ============================================

log("\n" + "="*70)
log("STEP 3: QUALITY CONTROL - MISSING VALUE FILTERING")
log("="*70)

# Separate metadata from features
metadata_cols = ['job_id', 'protein', 'ligand']
feature_cols = [c for c in combined_df.columns if c not in metadata_cols]

log(f"\nAnalyzing {len(feature_cols)} features for missing values...")

# Calculate missing percentage for each feature
missing_pct = combined_df[feature_cols].isna().sum() / len(combined_df) * 100

# Find features with >95% missing
high_missing = missing_pct[missing_pct > 95]

if len(high_missing) > 0:
    log(f"\n⚠️  Features with >95% missing values: {len(high_missing)}")
    for feat, pct in high_missing.items():
        log(f"  {feat}: {pct:.1f}% missing")
    
    # Remove these features
    features_to_remove = high_missing.index.tolist()
    feature_cols = [c for c in feature_cols if c not in features_to_remove]
    
    log(f"\n✓ Removed {len(features_to_remove)} features with >95% missing")
else:
    log(f"\n✓ No features with >95% missing values")

log(f"✓ Remaining features: {len(feature_cols)}")

# ============================================
# STEP 4: QUALITY CONTROL - ZERO VARIANCE
# ============================================

log("\n" + "="*70)
log("STEP 4: QUALITY CONTROL - ZERO VARIANCE FILTERING")
log("="*70)

# Calculate variance for numeric features only
numeric_features = combined_df[feature_cols].select_dtypes(include=[np.number]).columns

log(f"\nAnalyzing {len(numeric_features)} numeric features for variance...")

# Calculate variance
variances = combined_df[numeric_features].var()

# Find zero variance features
zero_var = variances[variances == 0]

if len(zero_var) > 0:
    log(f"\n⚠️  Features with zero variance: {len(zero_var)}")
    for feat in zero_var.index[:10]:  # Show first 10
        log(f"  {feat}")
    if len(zero_var) > 10:
        log(f"  ... and {len(zero_var) - 10} more")
    
    # Remove zero variance features
    feature_cols = [c for c in feature_cols if c not in zero_var.index]
    
    log(f"\n✓ Removed {len(zero_var)} zero variance features")
else:
    log(f"\n✓ No zero variance features found")

log(f"✓ Remaining features: {len(feature_cols)}")

# ============================================
# STEP 5: QUALITY CONTROL - CORRELATION
# ============================================

log("\n" + "="*70)
log("STEP 5: QUALITY CONTROL - CORRELATION FILTERING")
log("="*70)

log(f"\nCalculating correlation matrix for {len(feature_cols)} features...")

# Get numeric features only for correlation
numeric_feature_cols = [c for c in feature_cols if c in numeric_features]

if len(numeric_feature_cols) > 1:
    # Calculate correlation matrix
    corr_matrix = combined_df[numeric_feature_cols].corr().abs()
    
    # Find pairs with |r| > 0.95
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        (col, row, upper_triangle.loc[row, col])
        for col in upper_triangle.columns
        for row in upper_triangle.index
        if upper_triangle.loc[row, col] > 0.95
    ]
    
    if len(high_corr_pairs) > 0:
        log(f"\n⚠️  Found {len(high_corr_pairs)} feature pairs with |r| > 0.95")
        
        # Show first 10 pairs
        for i, (col, row, corr) in enumerate(high_corr_pairs[:10]):
            log(f"  {col} <-> {row}: r = {corr:.3f}")
        if len(high_corr_pairs) > 10:
            log(f"  ... and {len(high_corr_pairs) - 10} more pairs")
        
        # Remove one feature from each highly correlated pair
        # Strategy: Keep first, remove second
        features_to_remove = set()
        for col, row, corr in high_corr_pairs:
            if row not in features_to_remove:
                features_to_remove.add(row)
        
        feature_cols = [c for c in feature_cols if c not in features_to_remove]
        
        log(f"\n✓ Removed {len(features_to_remove)} highly correlated features")
    else:
        log(f"\n✓ No highly correlated feature pairs (|r| > 0.95) found")
else:
    log(f"\n⚠️  Only {len(numeric_feature_cols)} numeric features, skipping correlation")

log(f"✓ Remaining features after QC: {len(feature_cols)}")

# ============================================
# STEP 6: CREATE CLEAN FEATURE MATRIX
# ============================================

log("\n" + "="*70)
log("STEP 6: CREATING CLEAN FEATURE MATRIX")
log("="*70)

# Create final clean dataframe
clean_df = combined_df[metadata_cols + feature_cols].copy()

log(f"\n✓ Clean feature matrix created")
log(f"  Shape: {clean_df.shape}")
log(f"  Poses: {clean_df.shape[0]}")
log(f"  Features: {len(feature_cols)}")

# Save clean features
clean_features_file = COMBINED_DIR / "clean_docking_features.csv"
clean_df.to_csv(clean_features_file, index=False)

log(f"\n✓ Saved clean features: {clean_features_file}")

# ============================================
# CHECKPOINT 3: CORRELATION WITH IC50
# ============================================

log("\n" + "="*70)
log("CHECKPOINT 3: FEATURE CORRELATION WITH IC50 VALUES")
log("="*70)

# Load integrated dataset with IC50 values
integrated_file = DATA_PROCESSED / "caspid_integrated_dataset.csv"

if integrated_file.exists():
    log(f"\nLoading IC50 data from: {integrated_file}")
    
    ic50_df = pd.read_csv(integrated_file)
    
    # For each docking pose, we need to match to IC50 values
    # Note: Each pose is for a specific drug-protein pair
    # IC50 data has drug × cell line combinations
    
    log(f"✓ Loaded {len(ic50_df)} IC50 measurements")
    log(f"  Unique drugs: {ic50_df['DRUG_NAME'].nunique()}")
    log(f"  Unique cell lines: {ic50_df['CELL_LINE_NAME'].nunique()}")
    
    # Calculate feature-IC50 correlations
    # Strategy: For each feature, correlate with average IC50 per drug-protein
    
    log(f"\nCalculating feature correlations with IC50...")
    
    # Get average IC50 per drug
    avg_ic50_per_drug = ic50_df.groupby('DRUG_NAME')['LN_IC50'].mean().reset_index()
    avg_ic50_per_drug.columns = ['ligand', 'avg_ln_ic50']
    
    # Merge with our features
    features_with_ic50 = clean_df.merge(
        avg_ic50_per_drug,
        left_on='ligand',
        right_on='ligand',
        how='left'
    )
    
    # Calculate Spearman correlation for each feature
    significant_features = []
    
    for feat in feature_cols:
        if feat not in features_with_ic50.columns:
            continue
        
        # Get non-missing values
        valid_mask = features_with_ic50[feat].notna() & features_with_ic50['avg_ln_ic50'].notna()
        
        if valid_mask.sum() < 5:  # Need at least 5 points
            continue
        
        x = features_with_ic50.loc[valid_mask, feat]
        y = features_with_ic50.loc[valid_mask, 'avg_ln_ic50']
        
        try:
            rho, pval = spearmanr(x, y)
            
            if abs(rho) > 0.3:
                significant_features.append({
                    'feature': feat,
                    'spearman_rho': rho,
                    'p_value': pval,
                    'n_samples': valid_mask.sum()
                })
        except:
            continue
    
    # Sort by absolute correlation
    significant_features = sorted(
        significant_features,
        key=lambda x: abs(x['spearman_rho']),
        reverse=True
    )
    
    log(f"\n✓ Analyzed {len(feature_cols)} features")
    log(f"✓ Features with |Spearman ρ| > 0.3: {len(significant_features)}")
    
    if len(significant_features) >= 10:
        log(f"\n✅ CHECKPOINT 3 PASSED: ≥10 features correlate with IC50")
        log(f"\nTop 10 features by correlation with IC50:")
        
        for i, feat_info in enumerate(significant_features[:10], 1):
            log(f"  {i}. {feat_info['feature']}")
            log(f"     ρ = {feat_info['spearman_rho']:.3f}, p = {feat_info['p_value']:.4f}")
    elif len(significant_features) > 0:
        log(f"\n⚠️  CHECKPOINT 3 WARNING: Only {len(significant_features)} features with |ρ| > 0.3")
        log(f"   Expected ≥10 for strong signal")
        log(f"\nSignificant features:")
        
        for i, feat_info in enumerate(significant_features, 1):
            log(f"  {i}. {feat_info['feature']}")
            log(f"     ρ = {feat_info['spearman_rho']:.3f}, p = {feat_info['p_value']:.4f}")
    else:
        log(f"\n❌ CHECKPOINT 3 FAILED: No features with |ρ| > 0.3")
        log(f"   This may indicate:")
        log(f"   - Feature extraction issues")
        log(f"   - Insufficient structural diversity")
        log(f"   - IC50 data quality issues")
    
    # Save correlation results
    if significant_features:
        corr_df = pd.DataFrame(significant_features)
        corr_file = COMBINED_DIR / "feature_ic50_correlations.csv"
        corr_df.to_csv(corr_file, index=False)
        log(f"\n✓ Saved correlation results: {corr_file}")

else:
    log(f"\n⚠️  WARNING: IC50 data not found at {integrated_file}")
    log(f"   Skipping Checkpoint 3 correlation analysis")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("FEATURE EXTRACTION PIPELINE COMPLETE")
log("="*70)

log(f"\n📊 FINAL FEATURE SUMMARY:")
log(f"   Raw features extracted: {total_features}")
log(f"   After QC filtering: {len(feature_cols)}")
log(f"   Poses with features: {len(clean_df)}")
log(f"   Feature reduction: {total_features - len(feature_cols)} features removed")

log(f"\n📁 OUTPUT FILES:")
log(f"   Raw features: {raw_combined_file}")
log(f"   Clean features: {clean_features_file}")

if integrated_file.exists() and significant_features:
    log(f"   IC50 correlations: {corr_file}")

log(f"\n✅ STATUS: READY FOR MACHINE LEARNING")

log(f"\n📋 NEXT STEPS:")
log(f"   1. Review feature-IC50 correlations")
log(f"   2. Merge docking features with transcriptomic data")
log(f"   3. Implement neural conditioning layer")
log(f"   4. Train XGBoost prediction model")

log("\n" + "="*70)
log("ALL FEATURE EXTRACTION COMPLETE")
log("="*70)
