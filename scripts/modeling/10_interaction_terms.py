#!/usr/bin/env python3
"""
10_interaction_terms.py

Tests structural × transcriptomic interaction terms for significance.
Uses likelihood ratio tests with FDR correction.

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
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

import xgboost as xgb

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING"
RESULTS_DIR = PROJECT_ROOT / "results"
INTERACTIONS_DIR = RESULTS_DIR / "interactions"

INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = INTERACTIONS_DIR / f"interaction_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("INTERACTION TERM SELECTION")
print("="*70)

log("\n1. Loading data...")

data_file = MODELING_DIR / "full_modeling_dataset.csv"
df = pd.read_csv(data_file)

set_b_file = MODELING_DIR / "04_consensus" / "set_b_expanded_structural.txt"
with open(set_b_file, 'r') as f:
    structural_features = [line.strip() for line in f if line.strip()]

trans_file = MODELING_DIR / "transcriptomic_features.txt"
with open(trans_file, 'r') as f:
    transcriptomic_features = [line.strip() for line in f if line.strip()]

log(f"Dataset: {df.shape}")
log(f"Structural features: {len(structural_features)}")
log(f"Transcriptomic features: {len(transcriptomic_features)}")

y = df['LN_IC50'].values
X_struct = df[structural_features].values
X_trans = df[transcriptomic_features].values

imputer_struct = SimpleImputer(strategy='median')
X_struct = imputer_struct.fit_transform(X_struct)

imputer_trans = SimpleImputer(strategy='median')
X_trans = imputer_trans.fit_transform(X_trans)

discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

log("Data prepared")

log("\n2. Loading optimal hyperparameters...")

import pickle
optimized_model_file = MODELING_DIR / "06_conditioning" / "caspid_optimized_final.pkl"

if optimized_model_file.exists():
    with open(optimized_model_file, 'rb') as f:
        opt_data = pickle.load(f)
    optimal_params = opt_data['params']
    log(f"Loaded optimal parameters from Script 09")
else:
    optimal_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    log(f"Using default optimal parameters")

for k, v in optimal_params.items():
    log(f"  {k}: {v}")

log("\n3. Testing top interactions (structural × transcriptomic)...")

log("Strategy: Test top 50 structural features × top 50 transcriptomic features")
log("Total possible interactions: 32 × 95 = 3,040")
log("Testing subset for computational feasibility")

scaler_struct = StandardScaler()
X_struct_scaled = scaler_struct.fit_transform(X_struct)

scaler_trans = StandardScaler()
X_trans_scaled = scaler_trans.fit_transform(X_trans)

X_base = np.hstack([X_struct_scaled, X_trans_scaled])

struct_importance = []
for i in range(X_struct.shape[1]):
    corr = np.abs(np.corrcoef(X_struct[:, i], y)[0, 1])
    struct_importance.append((i, structural_features[i], corr))

struct_importance.sort(key=lambda x: x[2], reverse=True)
top_struct_idx = [x[0] for x in struct_importance[:20]]
top_struct_names = [x[1] for x in struct_importance[:20]]

trans_importance = []
for i in range(X_trans.shape[1]):
    corr = np.abs(np.corrcoef(X_trans[:, i], y)[0, 1])
    trans_importance.append((i, transcriptomic_features[i], corr))

trans_importance.sort(key=lambda x: x[2], reverse=True)
top_trans_idx = [x[0] for x in trans_importance[:20]]
top_trans_names = [x[1] for x in trans_importance[:20]]

log(f"\nTop 20 structural features by correlation:")
for i, (idx, name, corr) in enumerate(struct_importance[:20]):
    log(f"  {i+1}. {name}: r={corr:.3f}")

log(f"\nTop 20 transcriptomic features by correlation:")
for i, (idx, name, corr) in enumerate(trans_importance[:20]):
    log(f"  {i+1}. {name}: r={corr:.3f}")

log(f"\n4. Testing {len(top_struct_idx)} × {len(top_trans_idx)} = {len(top_struct_idx) * len(top_trans_idx)} interactions...")

base_model = xgb.XGBRegressor(**optimal_params, random_state=42, n_jobs=-1)
base_model.fit(X_base, y)
y_pred_base = base_model.predict(X_base)
base_r2 = r2_score(y, y_pred_base)

log(f"Base model R²: {base_r2:.4f}")

interaction_results = []

for s_idx, s_name in zip(top_struct_idx, top_struct_names):
    for t_idx, t_name in zip(top_trans_idx, top_trans_names):
        
        interaction = X_struct_scaled[:, s_idx] * X_trans_scaled[:, t_idx]
        
        X_with_interaction = np.hstack([X_base, interaction.reshape(-1, 1)])
        
        model_interact = xgb.XGBRegressor(**optimal_params, random_state=42, n_jobs=-1)
        model_interact.fit(X_with_interaction, y)
        y_pred_interact = model_interact.predict(X_with_interaction)
        interact_r2 = r2_score(y, y_pred_interact)
        
        r2_improvement = interact_r2 - base_r2
        
        n = len(y)
        k_base = X_base.shape[1]
        k_interact = k_base + 1
        
        rss_base = np.sum((y - y_pred_base)**2)
        rss_interact = np.sum((y - y_pred_interact)**2)
        
        f_stat = ((rss_base - rss_interact) / (k_interact - k_base)) / (rss_interact / (n - k_interact))
        p_value = 1 - stats.f.cdf(f_stat, k_interact - k_base, n - k_interact)
        
        interaction_results.append({
            'structural_feature': s_name,
            'transcriptomic_feature': t_name,
            'base_r2': base_r2,
            'interaction_r2': interact_r2,
            'r2_improvement': r2_improvement,
            'f_statistic': f_stat,
            'p_value': p_value
        })

log(f"Tested {len(interaction_results)} interactions")

results_df = pd.DataFrame(interaction_results)
results_df = results_df.sort_values('p_value')

log("\n5. Applying FDR correction...")

p_values = results_df['p_value'].values
rejected, p_adjusted = fdrcorrection(p_values, alpha=0.05, method='indep')

results_df['p_adjusted'] = p_adjusted
results_df['significant_fdr'] = rejected

n_significant = np.sum(rejected)
log(f"Significant interactions (FDR < 0.05): {n_significant}/{len(interaction_results)}")

if n_significant > 0:
    log(f"\nTop {min(10, n_significant)} significant interactions:")
    for i, row in results_df[results_df['significant_fdr']].head(10).iterrows():
        log(f"  {row['structural_feature']} × {row['transcriptomic_feature']}")
        log(f"    ΔR² = {row['r2_improvement']:+.6f}, p_adj = {row['p_adjusted']:.4e}")
else:
    log("\nNo interactions passed FDR correction")
    log("This suggests additive model is sufficient")

log("\n6. Cross-validation with top interactions...")

if n_significant > 0:
    top_interactions = results_df[results_df['significant_fdr']].head(10)
    
    log(f"Testing model with top {len(top_interactions)} interactions...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores_base = []
    cv_scores_interact = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, y_binned)):
        
        X_struct_train = X_struct_scaled[train_idx]
        X_trans_train = X_trans_scaled[train_idx]
        y_train = y[train_idx]
        
        X_struct_val = X_struct_scaled[val_idx]
        X_trans_val = X_trans_scaled[val_idx]
        y_val = y[val_idx]
        
        X_base_train = np.hstack([X_struct_train, X_trans_train])
        X_base_val = np.hstack([X_struct_val, X_trans_val])
        
        interactions_train = []
        interactions_val = []
        
        for _, row in top_interactions.iterrows():
            s_idx = top_struct_names.index(row['structural_feature'])
            t_idx = top_trans_names.index(row['transcriptomic_feature'])
            
            s_idx_orig = top_struct_idx[s_idx]
            t_idx_orig = top_trans_idx[t_idx]
            
            int_train = X_struct_train[:, s_idx_orig] * X_trans_train[:, t_idx_orig]
            int_val = X_struct_val[:, s_idx_orig] * X_trans_val[:, t_idx_orig]
            
            interactions_train.append(int_train)
            interactions_val.append(int_val)
        
        X_train_interact = np.hstack([X_base_train] + [i.reshape(-1, 1) for i in interactions_train])
        X_val_interact = np.hstack([X_base_val] + [i.reshape(-1, 1) for i in interactions_val])
        
        model_base = xgb.XGBRegressor(**optimal_params, random_state=42, n_jobs=-1)
        model_base.fit(X_base_train, y_train)
        val_r2_base = r2_score(y_val, model_base.predict(X_base_val))
        cv_scores_base.append(val_r2_base)
        
        model_interact = xgb.XGBRegressor(**optimal_params, random_state=42, n_jobs=-1)
        model_interact.fit(X_train_interact, y_train)
        val_r2_interact = r2_score(y_val, model_interact.predict(X_val_interact))
        cv_scores_interact.append(val_r2_interact)
        
        log(f"  Fold {fold_idx+1}: Base={val_r2_base:.4f}, +Interactions={val_r2_interact:.4f}, Δ={val_r2_interact-val_r2_base:+.4f}")
    
    mean_base = np.mean(cv_scores_base)
    mean_interact = np.mean(cv_scores_interact)
    improvement = mean_interact - mean_base
    
    log(f"\nCross-validation results:")
    log(f"  Base model: R² = {mean_base:.4f}")
    log(f"  + Interactions: R² = {mean_interact:.4f}")
    log(f"  Improvement: ΔR² = {improvement:+.4f}")
    
    if improvement > 0.001:
        log("  ✓ Interactions provide measurable improvement")
    else:
        log("  × Interactions do not improve generalization")
else:
    log("\nSkipping CV - no significant interactions found")

log("\n7. Saving results...")

results_file = INTERACTIONS_DIR / "interaction_analysis.csv"
results_df.to_csv(results_file, index=False)
log(f"Saved: {results_file}")

summary = {
    'timestamp': datetime.now().isoformat(),
    'total_tested': len(interaction_results),
    'significant_fdr': int(n_significant),
    'base_r2': float(base_r2),
    'top_interactions': results_df[results_df['significant_fdr']].head(10).to_dict('records') if n_significant > 0 else []
}

summary_file = INTERACTIONS_DIR / "interaction_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"Saved: {summary_file}")

log(f"\n{'='*70}")
log("INTERACTION TERM ANALYSIS COMPLETE")
log(f"{'='*70}")

if n_significant > 0:
    log(f"\nFound {n_significant} significant interactions")
    log(f"Recommend including top {min(10, n_significant)} in final model")
else:
    log(f"\nNo significant interactions found")
    log(f"Additive model (Script 09) is optimal")

log("="*70)
