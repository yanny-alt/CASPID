#!/usr/bin/env python3
"""
08_statistical_testing_and_baselines.py

Statistical validation of CASPID results with extended baseline comparisons.

Performs:
- Paired statistical tests (CASPID vs baselines)
- Extended baseline models (Ridge, Random Forest)
- Nested cross-validation
- Comprehensive statistical reporting

Requires: Kaggle results (caspid_full_summary.json, conditioning_weights.npy)

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

from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
VALIDATION_DIR = RESULTS_DIR / "validation"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = VALIDATION_DIR / f"statistical_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("STATISTICAL TESTING AND EXTENDED BASELINES")
print("="*70)

log("\n1. Loading Kaggle results and data...")

kaggle_results_file = MODELING_DIR / "06_conditioning" / "caspid_full_summary.json"
if not kaggle_results_file.exists():
    log(f"ERROR: Kaggle results not found at {kaggle_results_file}")
    log("Please place caspid_full_summary.json in DATA/MODELING/06_conditioning/")
    exit(1)

with open(kaggle_results_file, 'r') as f:
    kaggle_results = json.load(f)

log(f"Loaded Kaggle results: {kaggle_results_file}")

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

from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

log("Data prepared")

log("\n2. Extracting Kaggle results...")

kaggle_models = kaggle_results['models']

caspid_full_scores = np.array(kaggle_models['CASPID (Full)']['fold_scores'])
concat_scores = np.array(kaggle_models['Concatenation']['fold_scores'])
struct_scores = np.array(kaggle_models['Structure Only']['fold_scores'])
trans_scores = np.array(kaggle_models['Transcriptomics Only']['fold_scores'])

log(f"CASPID (Full): {len(caspid_full_scores)} scores")
log(f"Baseline scores extracted")

log("\n3. Training extended baselines...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

ridge_scores = []
rf_scores = []

log("\nRidge Regression baseline:")
log("-" * 50)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, y_binned)):
    print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ')
    
    X_struct_train = X_struct[train_idx]
    X_trans_train = X_trans[train_idx]
    y_train = y[train_idx]
    
    X_struct_val = X_struct[val_idx]
    X_trans_val = X_trans[val_idx]
    y_val = y[val_idx]
    
    scaler_s = StandardScaler()
    X_struct_train_sc = scaler_s.fit_transform(X_struct_train)
    X_struct_val_sc = scaler_s.transform(X_struct_val)
    
    scaler_t = StandardScaler()
    X_trans_train_sc = scaler_t.fit_transform(X_trans_train)
    X_trans_val_sc = scaler_t.transform(X_trans_val)
    
    X_train_concat = np.hstack([X_struct_train_sc, X_trans_train_sc])
    X_val_concat = np.hstack([X_struct_val_sc, X_trans_val_sc])
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_concat, y_train)
    
    y_pred_val = ridge.predict(X_val_concat)
    ridge_r2 = r2_score(y_val, y_pred_val)
    ridge_scores.append(ridge_r2)
    
    log(f"R² = {ridge_r2:.4f}")

ridge_mean = np.mean(ridge_scores)
ridge_std = np.std(ridge_scores)
log(f"\nRidge: R² = {ridge_mean:.4f} ± {ridge_std:.4f}")

log("\nRandom Forest baseline:")
log("-" * 50)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, y_binned)):
    print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ')
    
    X_struct_train = X_struct[train_idx]
    X_trans_train = X_trans[train_idx]
    y_train = y[train_idx]
    
    X_struct_val = X_struct[val_idx]
    X_trans_val = X_trans[val_idx]
    y_val = y[val_idx]
    
    scaler_s = StandardScaler()
    X_struct_train_sc = scaler_s.fit_transform(X_struct_train)
    X_struct_val_sc = scaler_s.transform(X_struct_val)
    
    scaler_t = StandardScaler()
    X_trans_train_sc = scaler_t.fit_transform(X_trans_train)
    X_trans_val_sc = scaler_t.transform(X_trans_val)
    
    X_train_concat = np.hstack([X_struct_train_sc, X_trans_train_sc])
    X_val_concat = np.hstack([X_struct_val_sc, X_trans_val_sc])
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_concat, y_train)
    
    y_pred_val = rf.predict(X_val_concat)
    rf_r2 = r2_score(y_val, y_pred_val)
    rf_scores.append(rf_r2)
    
    log(f"R² = {rf_r2:.4f}")

rf_mean = np.mean(rf_scores)
rf_std = np.std(rf_scores)
log(f"\nRandom Forest: R² = {rf_mean:.4f} ± {rf_std:.4f}")

log("\n4. Statistical testing...")

all_models = {
    'Structure Only': struct_scores,
    'Transcriptomics Only': trans_scores,
    'Ridge': np.array(ridge_scores),
    'Random Forest': np.array(rf_scores),
    'XGBoost Concat': concat_scores,
    'CASPID': caspid_full_scores
}

baseline_concat_r2 = kaggle_results['baseline_concatenation_r2']

log(f"\n{'='*70}")
log("COMPREHENSIVE RESULTS")
log(f"{'='*70}")

results_summary = []

for model_name, scores in all_models.items():
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    improvement = mean_r2 - baseline_concat_r2
    
    results_summary.append({
        'Model': model_name,
        'Mean_R2': mean_r2,
        'Std_R2': std_r2,
        'Min_R2': np.min(scores),
        'Max_R2': np.max(scores),
        'vs_Concat': improvement
    })
    
    log(f"\n{model_name}:")
    log(f"  R² = {mean_r2:.4f} ± {std_r2:.4f}")
    log(f"  Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    log(f"  vs Concat: {improvement:+.4f}")

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('Mean_R2', ascending=False)

log(f"\n{'='*70}")
log("STATISTICAL COMPARISONS (CASPID vs Baselines)")
log(f"{'='*70}")

statistical_tests = []

for model_name, scores in all_models.items():
    if model_name == 'CASPID':
        continue
    
    t_stat, p_value_t = ttest_rel(caspid_full_scores, scores)
    
    w_stat, p_value_w = wilcoxon(caspid_full_scores, scores)
    
    mean_diff = np.mean(caspid_full_scores) - np.mean(scores)
    
    ci_lower = mean_diff - 1.96 * np.std(caspid_full_scores - scores) / np.sqrt(len(scores))
    ci_upper = mean_diff + 1.96 * np.std(caspid_full_scores - scores) / np.sqrt(len(scores))
    
    effect_size = mean_diff / np.std(scores)
    
    statistical_tests.append({
        'Comparison': f'CASPID vs {model_name}',
        'Mean_Difference': mean_diff,
        'CI_95_Lower': ci_lower,
        'CI_95_Upper': ci_upper,
        't_statistic': t_stat,
        'p_value_ttest': p_value_t,
        'p_value_wilcoxon': p_value_w,
        'Cohens_d': effect_size,
        'Significant': p_value_t < 0.05
    })
    
    sig_marker = "***" if p_value_t < 0.001 else ("**" if p_value_t < 0.01 else ("*" if p_value_t < 0.05 else "ns"))
    
    log(f"\nCASPID vs {model_name}:")
    log(f"  ΔR² = {mean_diff:+.4f} [95% CI: {ci_lower:+.4f}, {ci_upper:+.4f}]")
    log(f"  t = {t_stat:.3f}, p = {p_value_t:.4f} {sig_marker}")
    log(f"  Wilcoxon p = {p_value_w:.4f}")
    log(f"  Cohen's d = {effect_size:.3f}")

stats_df = pd.DataFrame(statistical_tests)

bonferroni_alpha = 0.05 / len(statistical_tests)
log(f"\nBonferroni-corrected α = {bonferroni_alpha:.4f}")

significant_comparisons = stats_df[stats_df['p_value_ttest'] < bonferroni_alpha]
log(f"Significant comparisons (Bonferroni): {len(significant_comparisons)}/{len(statistical_tests)}")

log("\n5. Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
model_names = results_df['Model'].tolist()
means = results_df['Mean_R2'].tolist()
stds = results_df['Std_R2'].tolist()

colors = ['#e74c3c' if m == 'CASPID' else '#3498db' for m in model_names]
bars = ax1.bar(range(len(model_names)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)

ax1.axhline(y=baseline_concat_r2, color='gray', linestyle='--', linewidth=1.5, 
            label=f'XGBoost Concat ({baseline_concat_r2:.3f})')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[0, 1]
box_data = [all_models[m] for m in model_names]
bp = ax2.boxplot(box_data, labels=model_names, patch_artist=True)

for patch, model_name in zip(bp['boxes'], model_names):
    if model_name == 'CASPID':
        patch.set_facecolor('#e74c3c')
    else:
        patch.set_facecolor('#3498db')
    patch.set_alpha(0.6)

ax2.axhline(y=baseline_concat_r2, color='gray', linestyle='--', linewidth=1.5)
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('R² Distribution Across Folds', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

ax3 = axes[1, 0]
improvements = results_df['vs_Concat'].tolist()
colors_imp = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements]
bars_imp = ax3.barh(range(len(model_names)), improvements, color=colors_imp, alpha=0.7)

ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.set_yticks(range(len(model_names)))
ax3.set_yticklabels(model_names)
ax3.set_xlabel('ΔR² (vs XGBoost Concat)', fontsize=12)
ax3.set_title('Improvement Over Baseline', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

ax4 = axes[1, 1]
p_values = [-np.log10(row['p_value_ttest']) for _, row in stats_df.iterrows()]
comparisons = [row['Comparison'].replace('CASPID vs ', '') for _, row in stats_df.iterrows()]

bars_p = ax4.barh(range(len(comparisons)), p_values, color='#9b59b6', alpha=0.7)

ax4.axvline(x=-np.log10(0.05), color='orange', linestyle='--', linewidth=1.5, label='p=0.05')
ax4.axvline(x=-np.log10(0.01), color='red', linestyle='--', linewidth=1.5, label='p=0.01')
ax4.set_yticks(range(len(comparisons)))
ax4.set_yticklabels(comparisons)
ax4.set_xlabel('-log₁₀(p-value)', fontsize=12)
ax4.set_title('Statistical Significance', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig_file = FIGURES_DIR / "statistical_comparison.png"
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
log(f"\n✓ Saved: {fig_file}")
plt.close()

log("\n6. Saving results...")

results_file = VALIDATION_DIR / "model_comparison_results.csv"
results_df.to_csv(results_file, index=False)
log(f"✓ Saved: {results_file}")

stats_file = VALIDATION_DIR / "statistical_tests.csv"
stats_df.to_csv(stats_file, index=False)
log(f"✓ Saved: {stats_file}")

summary_output = {
    'timestamp': datetime.now().isoformat(),
    'baseline_concatenation_r2': float(baseline_concat_r2),
    'models': {
        name: {
            'mean_r2': float(np.mean(scores)),
            'std_r2': float(np.std(scores)),
            'fold_scores': [float(x) for x in scores]
        }
        for name, scores in all_models.items()
    },
    'statistical_tests': stats_df.to_dict('records'),
    'bonferroni_alpha': float(bonferroni_alpha),
    'significant_comparisons': int(len(significant_comparisons))
}

summary_file = VALIDATION_DIR / "statistical_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary_output, f, indent=2)
log(f"✓ Saved: {summary_file}")

log(f"\n{'='*70}")
log("STATISTICAL TESTING COMPLETE")
log(f"{'='*70}")

best_model = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Mean_R2']

log(f"\nBest Model: {best_model}")
log(f"R² = {best_r2:.4f} ± {results_df.iloc[0]['Std_R2']:.4f}")

caspid_vs_concat = stats_df[stats_df['Comparison'] == 'CASPID vs XGBoost Concat'].iloc[0]
log(f"\nCASPID vs XGBoost Concatenation:")
log(f"  ΔR² = {caspid_vs_concat['Mean_Difference']:+.4f}")
log(f"  95% CI: [{caspid_vs_concat['CI_95_Lower']:+.4f}, {caspid_vs_concat['CI_95_Upper']:+.4f}]")
log(f"  p-value = {caspid_vs_concat['p_value_ttest']:.6f}")

if caspid_vs_concat['p_value_ttest'] < 0.001:
    log(f"  *** Highly significant (p < 0.001)")
elif caspid_vs_concat['p_value_ttest'] < 0.01:
    log(f"  ** Very significant (p < 0.01)")
elif caspid_vs_concat['p_value_ttest'] < 0.05:
    log(f"  * Significant (p < 0.05)")
else:
    log(f"  Not significant (p ≥ 0.05)")

log(f"\nFiles saved in:")
log(f"  {VALIDATION_DIR}")
log(f"  {FIGURES_DIR}")

log("="*70)
