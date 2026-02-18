#!/usr/bin/env python3
"""
03_feature_selection_shap.py

SHAP-based feature selection on STRUCTURAL (docking) features.
Uses XGBoost model + SHAP to find features that contribute to predictions.

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

# Check for required packages
try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError as e:
    print("="*70)
    print("ERROR: Required packages not installed")
    print("="*70)
    print("\nPlease install:")
    print("  pip install xgboost shap --break-system-packages")
    print(f"\nMissing: {e}")
    exit(1)

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING" / "braf"
SHAP_DIR = MODELING_DIR / "03_shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = SHAP_DIR / f"shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("SHAP FEATURE SELECTION - STRUCTURAL FEATURES ONLY")
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

log("\n2. Preparing data for SHAP analysis...")

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

# Split data for validation
log(f"\n✓ Splitting data for model training...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
log(f"  Train: {X_train.shape[0]} samples")
log(f"  Test: {X_test.shape[0]} samples")

# ============================================
# STEP 3: TRAIN XGBOOST MODEL
# ============================================

log("\n3. Training XGBoost model...")

# Configure XGBoost
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

log(f"✓ XGBoost parameters:")
for k, v in xgb_params.items():
    log(f"  {k}: {v}")

start_time = datetime.now()

try:
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    elapsed_train = (datetime.now() - start_time).total_seconds()
    log(f"\n✓ Model training completed in {elapsed_train:.1f} seconds")
    
except Exception as e:
    log(f"\n✗ ERROR during model training:")
    log(f"  {str(e)}")
    import traceback
    log(traceback.format_exc())
    exit(1)

# Evaluate model
log("\n✓ Model performance:")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

log(f"  Train R²: {train_r2:.4f}")
log(f"  Test R²: {test_r2:.4f}")
log(f"  Train RMSE: {train_rmse:.4f}")
log(f"  Test RMSE: {test_rmse:.4f}")

if test_r2 < 0.1:
    log(f"\n⚠️  WARNING: Very low R² ({test_r2:.4f})")
    log(f"  This suggests features have weak predictive power")

# ============================================
# STEP 4: CALCULATE SHAP VALUES
# ============================================

log("\n4. Calculating SHAP values...")
log("  (This may take 5-15 minutes depending on data size)")

start_time = datetime.now()

try:
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values on a sample for speed
    # Use up to 5000 samples if available
    sample_size = min(5000, X_train.shape[0])
    X_sample = X_train[:sample_size]
    
    log(f"  Computing SHAP on {sample_size} samples...")
    
    shap_values = explainer.shap_values(X_sample)
    
    elapsed_shap = (datetime.now() - start_time).total_seconds()
    log(f"\n✓ SHAP calculation completed in {elapsed_shap:.1f} seconds ({elapsed_shap/60:.1f} minutes)")
    
except Exception as e:
    log(f"\n✗ ERROR during SHAP calculation:")
    log(f"  {str(e)}")
    import traceback
    log(traceback.format_exc())
    exit(1)

# ============================================
# STEP 5: COMPUTE FEATURE IMPORTANCE
# ============================================

log("\n5. Computing SHAP-based feature importance...")

# Feature importance = mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create results dataframe
shap_results = pd.DataFrame({
    'feature': docking_features[:len(mean_abs_shap)],
    'shap_importance': mean_abs_shap
}).sort_values('shap_importance', ascending=False)

log(f"\n✓ SHAP importance statistics:")
log(f"  Mean: {mean_abs_shap.mean():.6f}")
log(f"  Median: {np.median(mean_abs_shap):.6f}")
log(f"  Max: {mean_abs_shap.max():.6f}")
log(f"  Min: {mean_abs_shap.min():.6f}")

# ============================================
# STEP 6: SELECT TOP FEATURES
# ============================================

log("\n6. Selecting top features (95th percentile)...")

# Select features above 95th percentile
threshold = np.percentile(mean_abs_shap, 95)
selected_mask = mean_abs_shap >= threshold

selected_features = shap_results[shap_results['shap_importance'] >= threshold]

log(f"\n✓ Selection threshold (95th percentile): {threshold:.6f}")
log(f"✓ Selected {len(selected_features)} features")

log(f"\n✓ Top 20 features by SHAP importance:")
for i, row in shap_results.head(20).iterrows():
    log(f"  {i+1}. {row['feature']}: {row['shap_importance']:.6f}")

if len(shap_results) > 20:
    log(f"  ... (showing top 20 of {len(shap_results)})")

# ============================================
# STEP 7: COMPARE WITH PREVIOUS METHODS
# ============================================

log("\n7. Comparing with Boruta and MI results...")

# Load Boruta results
boruta_file = MODELING_DIR / "01_boruta" / "boruta_confirmed_features.txt"
mi_file = MODELING_DIR / "02_mi" / "mi_selected_features.txt"

comparisons = {}

if boruta_file.exists():
    with open(boruta_file, 'r') as f:
        boruta_features = set(line.strip() for line in f if line.strip())
    comparisons['Boruta'] = boruta_features
    log(f"✓ Loaded {len(boruta_features)} Boruta features")
else:
    log("⚠️  Boruta results not found")

if mi_file.exists():
    with open(mi_file, 'r') as f:
        mi_features = set(line.strip() for line in f if line.strip())
    comparisons['MI'] = mi_features
    log(f"✓ Loaded {len(mi_features)} MI features")
else:
    log("⚠️  MI results not found")

# Check overlaps
shap_selected = set(selected_features['feature'].values)

if comparisons:
    log(f"\n✓ Overlap analysis:")
    log(f"  SHAP selected: {len(shap_selected)}")
    
    for method, features in comparisons.items():
        overlap = features & shap_selected
        log(f"  {method} selected: {len(features)}")
        log(f"  {method} ∩ SHAP: {len(overlap)}")
        
        if len(overlap) > 0 and len(overlap) <= 10:
            log(f"    Overlapping features:")
            for feat in sorted(overlap):
                shap_val = shap_results[shap_results['feature'] == feat]['shap_importance'].values[0]
                log(f"      - {feat} (SHAP: {shap_val:.6f})")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================

log("\n8. Saving results...")

# Save selected features
selected_file = SHAP_DIR / "shap_selected_features.txt"
with open(selected_file, 'w') as f:
    f.write('\n'.join(selected_features['feature'].values))
log(f"✓ Saved selected features: {selected_file}")

# Save all SHAP importance scores
all_scores_file = SHAP_DIR / "shap_all_scores.csv"
shap_results.to_csv(all_scores_file, index=False)
log(f"✓ Saved all SHAP scores: {all_scores_file}")

# Save top features
top_scores_file = SHAP_DIR / "shap_top_features.csv"
selected_features.to_csv(top_scores_file, index=False)
log(f"✓ Saved top features: {top_scores_file}")

# Save model performance
performance = {
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse)
}

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': X.shape[0],
    'n_features_total': X.shape[1],
    'n_selected': len(selected_features),
    'selection_threshold': float(threshold),
    'shap_mean': float(mean_abs_shap.mean()),
    'shap_max': float(mean_abs_shap.max()),
    'runtime_train_seconds': elapsed_train,
    'runtime_shap_seconds': elapsed_shap,
    'model_performance': performance
}

import json
summary_file = SHAP_DIR / "shap_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("SHAP FEATURE SELECTION COMPLETE")
log("="*70)

log(f"\n📊 RESULTS SUMMARY:")
log(f"   Input features: {X.shape[1]}")
log(f"   Selected features: {len(selected_features)}")
log(f"   Model test R²: {test_r2:.4f}")
log(f"   SHAP threshold (95th percentile): {threshold:.6f}")
log(f"   Runtime (train): {elapsed_train:.1f}s")
log(f"   Runtime (SHAP): {elapsed_shap:.1f}s")

log(f"\n📁 OUTPUT FILES:")
log(f"   Selected features: {selected_file}")
log(f"   All scores: {all_scores_file}")
log(f"   Top features: {top_scores_file}")
log(f"   Summary: {summary_file}")

log(f"\n✅ Next step: Run 04_consensus_features.py")
log("="*70)
