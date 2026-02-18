#!/usr/bin/env python3
"""
01_feature_selection_boruta.py

Boruta feature selection on STRUCTURAL (docking) features.
Uses RandomForest with shadow features to identify truly important features.

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

# Check for boruta installation
try:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    print("="*70)
    print("ERROR: Required packages not installed")
    print("="*70)
    print("\nPlease install:")
    print("  pip install boruta --break-system-packages")
    print("  (scikit-learn should already be installed)")
    print(f"\nMissing: {e}")
    exit(1)

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING" / "braf"
BORUTA_DIR = MODELING_DIR / "01_boruta"
BORUTA_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = BORUTA_DIR / f"boruta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("BORUTA FEATURE SELECTION - STRUCTURAL FEATURES ONLY")
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
# STEP 2: PREPARE DATA FOR BORUTA
# ============================================

log("\n2. Preparing data for Boruta...")

# Extract target
y = df['LN_IC50'].values
log(f"✓ Target shape: {y.shape}")
log(f"  Mean: {y.mean():.3f}")
log(f"  Std: {y.std():.3f}")

# Extract ONLY docking features
# Verify all features exist
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

# Check for missing values in features
missing_mask = np.isnan(X)
missing_per_feature = missing_mask.sum(axis=0)
missing_per_sample = missing_mask.sum(axis=1)

log(f"\n✓ Missing value check:")
log(f"  Features with missing: {(missing_per_feature > 0).sum()}")
log(f"  Max missing per feature: {missing_per_feature.max()} ({missing_per_feature.max()/X.shape[0]*100:.1f}%)")
log(f"  Samples with missing: {(missing_per_sample > 0).sum()}")

# CRITICAL FIX: Filter BEFORE imputation
log(f"\n✓ Filtering features with >50% missing values...")
missing_pct_raw = missing_per_feature / X.shape[0]
valid_mask = missing_pct_raw < 0.5

log(f"  Total features: {len(docking_features)}")
log(f"  Features with >50% missing: {(~valid_mask).sum()}")
log(f"  Valid features to use: {valid_mask.sum()}")

# Filter features and data BEFORE imputation
docking_features = [docking_features[i] for i in range(len(docking_features)) if valid_mask[i]]
X = X[:, valid_mask]

log(f"\n✓ After filtering:")
log(f"  Features: {len(docking_features)}")
log(f"  Matrix shape: {X.shape}")

# Now impute the remaining valid features
if np.isnan(X).any():
    log(f"\n⚠️  Imputing missing values with median...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    log(f"✓ Imputation complete")

# Check for infinite values
inf_mask = np.isinf(X)
if inf_mask.any():
    log(f"\n⚠️  WARNING: {inf_mask.sum()} infinite values detected")
    log(f"  Replacing with feature max/min...")
    X = np.nan_to_num(X, posinf=np.nanmax(X[~inf_mask]), neginf=np.nanmin(X[~inf_mask]))

# ============================================
# STEP 3: CONFIGURE BORUTA
# ============================================

log("\n3. Configuring Boruta algorithm...")

# RandomForest base estimator
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=7,
    min_samples_split=10,
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)
log(f"✓ RandomForest configured")
log(f"  n_estimators: 100")
log(f"  max_depth: 7")
log(f"  min_samples_split: 10")

# Boruta selector
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    max_iter=100,
    alpha=0.01,  # p-value threshold (strict)
    random_state=42,
    verbose=2
)
log(f"✓ Boruta configured")
log(f"  max_iter: 100")
log(f"  alpha: 0.01 (p-value threshold)")

# ============================================
# STEP 4: RUN BORUTA
# ============================================

log("\n4. Running Boruta feature selection...")
log("  (This may take 10-30 minutes depending on data size)")
log("  Progress will be shown below:\n")

start_time = datetime.now()

try:
    boruta_selector.fit(X, y)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\n✓ Boruta completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
except Exception as e:
    log(f"\n✗ ERROR during Boruta execution:")
    log(f"  {str(e)}")
    import traceback
    log(traceback.format_exc())
    exit(1)

# ============================================
# STEP 5: EXTRACT RESULTS
# ============================================

log("\n5. Extracting Boruta results...")

# Get feature rankings
# ranking_: 1 = confirmed, 2+ = tentative or rejected
confirmed_mask = boruta_selector.support_
tentative_mask = boruta_selector.support_weak_
ranking = boruta_selector.ranking_

confirmed_features = [docking_features[i] for i, conf in enumerate(confirmed_mask) if conf]
tentative_features = [docking_features[i] for i, tent in enumerate(tentative_mask) if tent]

log(f"\n✓ Feature selection results:")
log(f"  Confirmed features: {len(confirmed_features)}")
log(f"  Tentative features: {len(tentative_features)}")
log(f"  Rejected features: {len(docking_features) - len(confirmed_features) - len(tentative_features)}")

# ============================================
# STEP 6: ANALYZE CONFIRMED FEATURES
# ============================================

log("\n6. Analyzing confirmed features...")

if len(confirmed_features) == 0:
    log("\n⚠️  WARNING: No features confirmed by Boruta!")
    log("  This may indicate:")
    log("  - Features don't predict IC50 well individually")
    log("  - Need to check tentative features")
    log("  - May need to relax alpha threshold")
else:
    log(f"\n✓ Confirmed features ({len(confirmed_features)}):")
    for i, feat in enumerate(confirmed_features[:20], 1):
        log(f"  {i}. {feat}")
    if len(confirmed_features) > 20:
        log(f"  ... and {len(confirmed_features) - 20} more")

if len(tentative_features) > 0:
    log(f"\n✓ Tentative features ({len(tentative_features)}):")
    for i, feat in enumerate(tentative_features[:10], 1):
        log(f"  {i}. {feat}")
    if len(tentative_features) > 10:
        log(f"  ... and {len(tentative_features) - 10} more")

# ============================================
# STEP 7: GET FEATURE IMPORTANCES
# ============================================

log("\n7. Calculating feature importances...")

# Train final RF on selected features to get importances
if len(confirmed_features) > 0:
    confirmed_indices = [i for i, conf in enumerate(confirmed_mask) if conf]
    X_confirmed = X[:, confirmed_indices]
    
    rf_final = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42
    )
    rf_final.fit(X_confirmed, y)
    
    importances = rf_final.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': confirmed_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    log(f"\n✓ Top 10 features by importance:")
    for i, row in importance_df.head(10).iterrows():
        log(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================

log("\n8. Saving results...")

# Save confirmed features
confirmed_file = BORUTA_DIR / "boruta_confirmed_features.txt"
with open(confirmed_file, 'w') as f:
    f.write('\n'.join(confirmed_features))
log(f"✓ Saved confirmed features: {confirmed_file}")

# Save tentative features
tentative_file = BORUTA_DIR / "boruta_tentative_features.txt"
with open(tentative_file, 'w') as f:
    f.write('\n'.join(tentative_features))
log(f"✓ Saved tentative features: {tentative_file}")

# Save all results (skip if arrays incompatible)
results_df = None
try:
    results_df = pd.DataFrame({
        'feature': docking_features,
        'confirmed': list(confirmed_mask),
        'tentative': list(tentative_mask),
        'ranking': list(ranking)
    })
    results_file = BORUTA_DIR / "boruta_all_results.csv"
    results_df.to_csv(results_file, index=False)
    log(f"✓ Saved all results: {results_file}")
except Exception as e:
    log(f"⚠️  Could not save full results: {e}")
    log(f"  (Not critical - confirmed/tentative lists already saved)")

# Save importance scores (if confirmed features exist)
if len(confirmed_features) > 0:
    importance_file = BORUTA_DIR / "boruta_feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)
    log(f"✓ Saved importance scores: {importance_file}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': X.shape[0],
    'n_features_total': X.shape[1],
    'n_confirmed': len(confirmed_features),
    'n_tentative': len(tentative_features),
    'n_rejected': X.shape[1] - len(confirmed_features) - len(tentative_features),
    'runtime_seconds': elapsed,
    'alpha': 0.01,
    'max_iter': 100
}

import json
summary_file = BORUTA_DIR / "boruta_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("BORUTA FEATURE SELECTION COMPLETE")
log("="*70)

log(f"\n📊 RESULTS SUMMARY:")
log(f"   Input features: {X.shape[1]}")
log(f"   Confirmed features: {len(confirmed_features)}")
log(f"   Tentative features: {len(tentative_features)}")
log(f"   Rejected features: {X.shape[1] - len(confirmed_features) - len(tentative_features)}")
log(f"   Runtime: {elapsed:.1f} seconds")

log(f"\n📁 OUTPUT FILES:")
log(f"   Confirmed features: {confirmed_file}")
log(f"   Tentative features: {tentative_file}")
# log(f"   All results: {results_file}")
if len(confirmed_features) > 0:
    log(f"   Importances: {importance_file}")
log(f"   Summary: {summary_file}")

log(f"\n✅ Next step: Run 02_feature_selection_mi.py")
log("="*70)
