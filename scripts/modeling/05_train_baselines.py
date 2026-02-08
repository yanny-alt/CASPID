#!/usr/bin/env python3
"""
05_train_baselines.py

Train baseline models for comparison with CASPID conditioning layer.
Establishes performance benchmarks that conditioning must beat.

Baselines:
1. Structure-only (Set B: 32 features)
2. Transcriptomics-only (95 features)
3. Simple concatenation (127 features, no conditioning)

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING"
BASELINES_DIR = MODELING_DIR / "05_baselines"
BASELINES_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = BASELINES_DIR / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("BASELINE MODELS - ESTABLISHING PERFORMANCE BENCHMARKS")
print("="*70)

log("\n1. Loading data and features...")

data_file = MODELING_DIR / "full_modeling_dataset.csv"
if not data_file.exists():
    log(f"ERROR: Dataset not found: {data_file}")
    exit(1)

df = pd.read_csv(data_file)
log(f"Loaded dataset: {df.shape}")

set_b_file = MODELING_DIR / "04_consensus" / "set_b_expanded_structural.txt"
with open(set_b_file, 'r') as f:
    structural_features = [line.strip() for line in f if line.strip()]

trans_file = MODELING_DIR / "transcriptomic_features.txt"
with open(trans_file, 'r') as f:
    transcriptomic_features = [line.strip() for line in f if line.strip()]

log(f"Structural features (Set B): {len(structural_features)}")
log(f"Transcriptomic features: {len(transcriptomic_features)}")

missing_struct = [f for f in structural_features if f not in df.columns]
missing_trans = [f for f in transcriptomic_features if f not in df.columns]

if missing_struct:
    log(f"ERROR: Missing structural features: {missing_struct[:5]}")
    exit(1)
if missing_trans:
    log(f"ERROR: Missing transcriptomic features: {missing_trans[:5]}")
    exit(1)

y = df['LN_IC50'].values
X_struct = df[structural_features].values
X_trans = df[transcriptomic_features].values
X_concat = np.hstack([X_struct, X_trans])

log(f"\nData shapes:")
log(f"  Target (y): {y.shape}")
log(f"  Structural (X_struct): {X_struct.shape}")
log(f"  Transcriptomic (X_trans): {X_trans.shape}")
log(f"  Concatenated (X_concat): {X_concat.shape}")

from sklearn.impute import SimpleImputer

imputer_struct = SimpleImputer(strategy='median')
X_struct = imputer_struct.fit_transform(X_struct)

imputer_trans = SimpleImputer(strategy='median')
X_trans = imputer_trans.fit_transform(X_trans)

X_concat = np.hstack([X_struct, X_trans])

log(f"Imputation complete")

log("\n2. Creating stratified folds...")

from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

n_folds = 5
n_repeats = 10
random_seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

log(f"Cross-validation: {n_folds}-fold, {n_repeats} repeats")
log(f"Total runs per model: {n_folds * n_repeats} = {n_folds * n_repeats}")

log("\n3. Defining XGBoost configuration...")

xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

log(f"XGBoost parameters:")
for k, v in xgb_params.items():
    log(f"  {k}: {v}")

def evaluate_model(y_true, y_pred):
    """Calculate all metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    spearman_r, _ = spearmanr(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman_r,
        'pearson': pearson_r
    }

def train_and_evaluate(X, y, model_name, seed):
    """Train model with cross-validation"""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_binned)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)
        
        fold_results.append({
            'fold': fold_idx,
            'seed': seed,
            'train': train_metrics,
            'test': test_metrics
        })
    
    return fold_results

log("\n4. Training Model 1: Structure-only...")

struct_results = []
for repeat_idx, seed in enumerate(random_seeds):
    log(f"  Repeat {repeat_idx+1}/{n_repeats} (seed={seed})...")
    results = train_and_evaluate(X_struct, y, "structure", seed)
    struct_results.extend(results)

log(f"Completed {len(struct_results)} runs")

log("\n5. Training Model 2: Transcriptomics-only...")

trans_results = []
for repeat_idx, seed in enumerate(random_seeds):
    log(f"  Repeat {repeat_idx+1}/{n_repeats} (seed={seed})...")
    results = train_and_evaluate(X_trans, y, "transcriptomics", seed)
    trans_results.extend(results)

log(f"Completed {len(trans_results)} runs")

log("\n6. Training Model 3: Simple concatenation...")

concat_results = []
for repeat_idx, seed in enumerate(random_seeds):
    log(f"  Repeat {repeat_idx+1}/{n_repeats} (seed={seed})...")
    results = train_and_evaluate(X_concat, y, "concatenation", seed)
    concat_results.extend(results)

log(f"Completed {len(concat_results)} runs")

log("\n7. Computing summary statistics...")

def summarize_results(results, model_name):
    """Calculate mean and std across all runs"""
    
    test_r2 = [r['test']['r2'] for r in results]
    test_rmse = [r['test']['rmse'] for r in results]
    test_mae = [r['test']['mae'] for r in results]
    test_spearman = [r['test']['spearman'] for r in results]
    
    train_r2 = [r['train']['r2'] for r in results]
    
    summary = {
        'model': model_name,
        'n_runs': len(results),
        'test_r2_mean': np.mean(test_r2),
        'test_r2_std': np.std(test_r2),
        'test_r2_min': np.min(test_r2),
        'test_r2_max': np.max(test_r2),
        'test_rmse_mean': np.mean(test_rmse),
        'test_rmse_std': np.std(test_rmse),
        'test_mae_mean': np.mean(test_mae),
        'test_mae_std': np.std(test_mae),
        'test_spearman_mean': np.mean(test_spearman),
        'test_spearman_std': np.std(test_spearman),
        'train_r2_mean': np.mean(train_r2),
        'train_r2_std': np.std(train_r2),
        'test_r2_values': test_r2
    }
    
    return summary

struct_summary = summarize_results(struct_results, "Structure-only")
trans_summary = summarize_results(trans_results, "Transcriptomics-only")
concat_summary = summarize_results(concat_results, "Concatenation")

log(f"\n{'='*70}")
log("BASELINE RESULTS")
log(f"{'='*70}")

for summary in [struct_summary, trans_summary, concat_summary]:
    log(f"\n{summary['model']}:")
    log(f"  Test R²: {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}")
    log(f"  Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}")
    log(f"  Test MAE: {summary['test_mae_mean']:.4f} ± {summary['test_mae_std']:.4f}")
    log(f"  Test Spearman: {summary['test_spearman_mean']:.4f} ± {summary['test_spearman_std']:.4f}")
    log(f"  Train R²: {summary['train_r2_mean']:.4f} ± {summary['train_r2_std']:.4f}")
    log(f"  Range: [{summary['test_r2_min']:.4f}, {summary['test_r2_max']:.4f}]")

log("\n8. Statistical testing...")

from scipy.stats import ttest_rel

def compare_models(results1, results2, name1, name2):
    """Paired t-test between two models"""
    
    r2_1 = [r['test']['r2'] for r in results1]
    r2_2 = [r['test']['r2'] for r in results2]
    
    t_stat, p_value = ttest_rel(r2_2, r2_1)
    
    mean_diff = np.mean(r2_2) - np.mean(r2_1)
    
    return {
        'comparison': f"{name2} vs {name1}",
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'significant': p_value < 0.05
    }

comparisons = [
    compare_models(struct_results, trans_results, "Structure", "Transcriptomics"),
    compare_models(struct_results, concat_results, "Structure", "Concatenation"),
    compare_models(trans_results, concat_results, "Transcriptomics", "Concatenation")
]

log(f"\nPaired t-tests (n={len(struct_results)} pairs):")
for comp in comparisons:
    sig_marker = "***" if comp['p_value'] < 0.001 else ("**" if comp['p_value'] < 0.01 else ("*" if comp['p_value'] < 0.05 else "ns"))
    log(f"\n  {comp['comparison']}:")
    log(f"    ΔR² = {comp['mean_difference']:+.4f}")
    log(f"    t = {comp['t_statistic']:.3f}")
    log(f"    p = {comp['p_value']:.4f} {sig_marker}")

log("\n9. Saving results...")

results_df = pd.DataFrame({
    'Model': ['Structure-only', 'Transcriptomics-only', 'Concatenation'],
    'Test_R2_Mean': [struct_summary['test_r2_mean'], trans_summary['test_r2_mean'], concat_summary['test_r2_mean']],
    'Test_R2_Std': [struct_summary['test_r2_std'], trans_summary['test_r2_std'], concat_summary['test_r2_std']],
    'Test_RMSE_Mean': [struct_summary['test_rmse_mean'], trans_summary['test_rmse_mean'], concat_summary['test_rmse_mean']],
    'Test_MAE_Mean': [struct_summary['test_mae_mean'], trans_summary['test_mae_mean'], concat_summary['test_mae_mean']],
    'Test_Spearman_Mean': [struct_summary['test_spearman_mean'], trans_summary['test_spearman_mean'], concat_summary['test_spearman_mean']],
    'Train_R2_Mean': [struct_summary['train_r2_mean'], trans_summary['train_r2_mean'], concat_summary['train_r2_mean']],
    'N_Runs': [struct_summary['n_runs'], trans_summary['n_runs'], concat_summary['n_runs']]
})

results_file = BASELINES_DIR / "baseline_summary.csv"
results_df.to_csv(results_file, index=False)
log(f"Saved summary: {results_file}")

detailed_results = {
    'structure_only': struct_results,
    'transcriptomics_only': trans_results,
    'concatenation': concat_results
}

import pickle
detailed_file = BASELINES_DIR / "baseline_detailed_results.pkl"
with open(detailed_file, 'wb') as f:
    pickle.dump(detailed_results, f)
log(f"Saved detailed results: {detailed_file}")

summary_dict = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': len(y),
    'n_structural_features': len(structural_features),
    'n_transcriptomic_features': len(transcriptomic_features),
    'cv_folds': n_folds,
    'cv_repeats': n_repeats,
    'total_runs_per_model': n_folds * n_repeats,
    'xgb_params': xgb_params,
    'models': {
        'structure_only': struct_summary,
        'transcriptomics_only': trans_summary,
        'concatenation': concat_summary
    },
    'statistical_tests': [
    {k: (v if not isinstance(v, (np.bool_, bool)) else bool(v)) 
     for k, v in comp.items()} 
    for comp in comparisons
]
}

for key in ['test_r2_values']:
    if key in summary_dict['models']['structure_only']:
        del summary_dict['models']['structure_only'][key]
    if key in summary_dict['models']['transcriptomics_only']:
        del summary_dict['models']['transcriptomics_only'][key]
    if key in summary_dict['models']['concatenation']:
        del summary_dict['models']['concatenation'][key]

summary_file = BASELINES_DIR / "baseline_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary_dict, f, indent=2)
log(f"Saved summary JSON: {summary_file}")

log(f"\n{'='*70}")
log("BASELINE TRAINING COMPLETE")
log(f"{'='*70}")

log(f"\nBEST BASELINE: {concat_summary['model']}")
log(f"  R² = {concat_summary['test_r2_mean']:.4f} ± {concat_summary['test_r2_std']:.4f}")

log(f"\nCONDITIONING LAYER MUST BEAT:")
log(f"  Concatenation R² > {concat_summary['test_r2_mean']:.4f}")
log(f"  Target improvement: ΔR² ≥ 0.05 (p < 0.01)")

log(f"\nFiles saved:")
log(f"  {results_file}")
log(f"  {detailed_file}")
log(f"  {summary_file}")

log(f"\nNext: Run 06_conditioning_architectures.py")
log("="*70)
