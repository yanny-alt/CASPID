"""
CASPID Hyperparameter Optimization - Kaggle Version
====================================================

Comprehensive hyperparameter optimization using nested cross-validation.

Optimizes XGBoost parameters for conditioned features.

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

import xgboost as xgb

print("="*70)
print("HYPERPARAMETER OPTIMIZATION - NESTED CV")
print("="*70)

print("\n1. Loading data...")

df = pd.read_csv('/kaggle/input/caspid-conditioning-layer-training-data/caspid_conditioning_data.csv')

struct_cols = [c for c in df.columns if c.startswith('struct_')]
trans_cols = [c for c in df.columns if c.startswith('trans_')]

X_struct = df[struct_cols].values
X_trans = df[trans_cols].values
y = df['target'].values
strat_bins = df['strat_bin'].values

print(f"Dataset: {df.shape}")
print(f"Structural features: {len(struct_cols)}")
print(f"Transcriptomic features: {len(trans_cols)}")

print("\n2. Loading conditioning weights...")

# Load weights from previous runs
try:
    weights_file = '/kaggle/input/caspid-full-model-results/conditioning_weights.npy'
    conditioning_weights = np.load(weights_file)
    print(f"Loaded conditioning weights: {conditioning_weights.shape}")
except:
    print("WARNING: Using uniform weights (no conditioning)")
    conditioning_weights = None

print("\n3. Defining hyperparameter grid...")

xgb_param_grid = {
    'n_estimators': [200],              # Just 1 value (was 3)
    'max_depth': [6, 8],                # Just 2 values (was 3)
    'learning_rate': [0.05, 0.1],       # Just 2 values (was 3)
    'subsample': [0.8],                 # Just 1 value (was 3)
    'colsample_bytree': [0.8],          # Just 1 value (was 3)
    'reg_alpha': [0.1],                 # Just 1 value (was 3)
    'reg_lambda': [1.0]                 # Just 1 value (was 3)
}

total_combinations = np.prod([len(v) for v in xgb_param_grid.values()])
print(f"Total combinations: {total_combinations}")

print("\n4. Setting up nested cross-validation...")

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=123)

print("Outer CV: 5-fold (evaluation)")
print("Inner CV: 3-fold (selection)")

print("\n5. Running nested CV...")

outer_results = []
best_params_per_fold = []

for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_struct, strat_bins)):
    
    print(f"\n{'='*70}")
    print(f"OUTER FOLD {outer_idx+1}/5")
    print(f"{'='*70}")
    
    X_struct_train = X_struct[outer_train_idx]
    X_trans_train = X_trans[outer_train_idx]
    y_train = y[outer_train_idx]
    
    X_struct_test = X_struct[outer_test_idx]
    X_trans_test = X_trans[outer_test_idx]
    y_test = y[outer_test_idx]
    
    scaler_s = StandardScaler()
    X_struct_train_sc = scaler_s.fit_transform(X_struct_train)
    X_struct_test_sc = scaler_s.transform(X_struct_test)
    
    scaler_t = StandardScaler()
    X_trans_train_sc = scaler_t.fit_transform(X_trans_train)
    X_trans_test_sc = scaler_t.transform(X_trans_test)
    
    if conditioning_weights is not None:
        w_train = conditioning_weights[outer_idx][outer_train_idx]
        w_test = conditioning_weights[outer_idx][outer_test_idx]
        X_struct_train_cond = X_struct_train_sc * w_train
        X_struct_test_cond = X_struct_test_sc * w_test
    else:
        X_struct_train_cond = X_struct_train_sc
        X_struct_test_cond = X_struct_test_sc
    
    X_train_combined = np.hstack([X_struct_train_cond, X_trans_train_sc])
    X_test_combined = np.hstack([X_struct_test_cond, X_trans_test_sc])
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Features: {X_train_combined.shape[1]}")
    
    print(f"Grid search ({total_combinations} combinations)...")
    
    xgb_base = xgb.XGBRegressor(
    random_state=42,
    tree_method='hist',
    device='cuda:0'  # Use device instead of gpu_id
    )

    grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    cv=inner_cv,
    scoring='r2',
    n_jobs=-1,  # Back to -1 is fine
    verbose=1
    )

    
    grid_search.fit(X_train_combined, y_train)
    
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Inner CV R²: {best_cv_score:.4f}")
    
    best_model = grid_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train_combined)
    train_r2 = r2_score(y_train, y_pred_train)
    
    y_pred_test = best_model.predict(X_test_combined)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nPerformance:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Gap: {train_r2 - test_r2:.4f}")
    
    outer_results.append({
        'fold': outer_idx,
        'best_params': best_params,
        'inner_cv_r2': best_cv_score,
        'train_r2': train_r2,
        'test_r2': test_r2
    })
    
    best_params_per_fold.append(best_params)

print(f"\n{'='*70}")
print("NESTED CV COMPLETE")
print(f"{'='*70}")

test_scores = [r['test_r2'] for r in outer_results]
mean_r2 = np.mean(test_scores)
std_r2 = np.std(test_scores)

print(f"\nTest R² per fold:")
for i, score in enumerate(test_scores):
    print(f"  Fold {i+1}: {score:.4f}")

print(f"\nMean: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"Range: [{min(test_scores):.4f}, {max(test_scores):.4f}]")

baseline_r2 = 0.8451
improvement = mean_r2 - baseline_r2

print(f"\nvs Baseline (R² = {baseline_r2:.4f}):")
print(f"  Optimized: {mean_r2:.4f}")
print(f"  Improvement: {improvement:+.4f}")

if improvement > 0.01:
    print("  ✓ Significant improvement")
elif improvement > 0:
    print("  ~ Marginal improvement")
else:
    print("  × No improvement")

print("\n6. Finding consensus parameters...")

param_freq = {}
for params in best_params_per_fold:
    for k, v in params.items():
        if k not in param_freq:
            param_freq[k] = {}
        if v not in param_freq[k]:
            param_freq[k][v] = 0
        param_freq[k][v] += 1

consensus = {}
print("\nConsensus parameters:")
for param, counts in param_freq.items():
    best_val = max(counts.items(), key=lambda x: x[1])
    consensus[param] = best_val[0]
    print(f"  {param}: {best_val[0]} ({best_val[1]}/5 folds)")

print("\n7. Training final model...")

scaler_s_final = StandardScaler()
X_struct_final = scaler_s_final.fit_transform(X_struct)

scaler_t_final = StandardScaler()
X_trans_final = scaler_t_final.fit_transform(X_trans)

if conditioning_weights is not None:
    mean_weights = np.mean(conditioning_weights, axis=0)
    X_struct_cond_final = X_struct_final * mean_weights
else:
    X_struct_cond_final = X_struct_final

X_final = np.hstack([X_struct_cond_final, X_trans_final])

final_model = xgb.XGBRegressor(
    **consensus,
    random_state=42,
    tree_method='hist',
    device='cuda:0'  # Use device instead of gpu_id
)

final_model.fit(X_final, y)

y_pred_final = final_model.predict(X_final)
final_r2 = r2_score(y, y_pred_final)

print(f"\nFinal model (full data):")
print(f"  R²: {final_r2:.4f}")

print("\n8. Saving results...")

results_df = pd.DataFrame(outer_results)
results_df.to_csv('hyperparameter_optimization_results.csv', index=False)
print("✓ Saved: hyperparameter_optimization_results.csv")

summary = {
    'baseline_r2': float(baseline_r2),
    'optimized_mean_r2': float(mean_r2),
    'optimized_std_r2': float(std_r2),
    'improvement': float(improvement),
    'consensus_params': {k: str(v) for k, v in consensus.items()},
    'outer_fold_results': outer_results,
    'final_model_r2': float(final_r2)
}

with open('optimization_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: optimization_summary.json")

# Save final model
with open('caspid_optimized_final.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'params': consensus,
        'cv_mean_r2': mean_r2,
        'cv_std_r2': std_r2,
        'full_r2': final_r2
    }, f)
print("✓ Saved: caspid_optimized_final.pkl")

print(f"\n{'='*70}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*70}")

print(f"\nOptimized CASPID:")
print(f"  CV R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"  Full R²: {final_r2:.4f}")
print(f"  Improvement: {improvement:+.4f}")

print(f"\nOptimal parameters:")
for k, v in consensus.items():
    print(f"  {k}: {v}")

print("="*70)
