#!/usr/bin/env python3
"""
11_final_egfr_model.py

Trains final optimized CASPID model for EGFR inhibitors.

Combines:
- Optimal hyperparameters (from Script 09)
- Selected interactions (from Script 10, if significant)
- Full dataset training

Saves model for BRAF validation and generates predictions for interpretation.

Author: CASPID Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr, pearsonr

import xgboost as xgb

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
INTERACTIONS_DIR = RESULTS_DIR / "interactions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = MODELS_DIR / f"final_egfr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("FINAL EGFR MODEL TRAINING")
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

cell_lines = df['CCLE_Name'].values if 'CCLE_Name' in df.columns else None
drug_names = df['DRUG_NAME'].values if 'DRUG_NAME' in df.columns else None

imputer_struct = SimpleImputer(strategy='median')
X_struct = imputer_struct.fit_transform(X_struct)

imputer_trans = SimpleImputer(strategy='median')
X_trans = imputer_trans.fit_transform(X_trans)

log("\n2. Loading optimal configuration...")

opt_file = MODELING_DIR / "06_conditioning" / "optimization_summary.json"
with open(opt_file, 'rb') as f:
    opt_summary = json.load(f)

optimal_params = {
    'n_estimators': int(opt_summary['consensus_params']['n_estimators']),
    'max_depth': int(opt_summary['consensus_params']['max_depth']),
    'learning_rate': float(opt_summary['consensus_params']['learning_rate']),
    'subsample': float(opt_summary['consensus_params']['subsample']),
    'colsample_bytree': float(opt_summary['consensus_params']['colsample_bytree']),
    'reg_alpha': float(opt_summary['consensus_params']['reg_alpha']),
    'reg_lambda': float(opt_summary['consensus_params']['reg_lambda'])
}

log("Optimal hyperparameters:")
for k, v in optimal_params.items():
    log(f"  {k}: {v}")

log("\n3. Checking for interaction terms...")

interaction_file = INTERACTIONS_DIR / "interaction_summary.json"
use_interactions = False
selected_interactions = []

if interaction_file.exists():
    with open(interaction_file, 'r') as f:
        interaction_summary = json.load(f)
    
    n_significant = interaction_summary.get('significant_fdr', 0)
    
    if n_significant > 0:
        use_interactions = True
        # Get top 10 interactions
        top_interactions = interaction_summary.get('top_interactions', [])[:10]
        
        for inter in top_interactions:
            selected_interactions.append({
                'struct_feature': inter['structural_feature'],
                'trans_feature': inter['transcriptomic_feature'],
                'struct_idx': structural_features.index(inter['structural_feature']),
                'trans_idx': transcriptomic_features.index(inter['transcriptomic_feature'])
            })
        
        log(f"Using {len(selected_interactions)} interaction terms")
        for inter in selected_interactions:
            log(f"  {inter['struct_feature']} × {inter['trans_feature']}")
    else:
        log("No interactions selected (main effects sufficient)")
else:
    log("No interaction analysis found - using main effects only")

log("\n4. Preparing features...")

scaler_struct = StandardScaler()
X_struct_scaled = scaler_struct.fit_transform(X_struct)

scaler_trans = StandardScaler()
X_trans_scaled = scaler_trans.fit_transform(X_trans)

X_features = np.hstack([X_struct_scaled, X_trans_scaled])

if use_interactions and selected_interactions:
    interaction_features = []
    
    for inter in selected_interactions:
        s_idx = inter['struct_idx']
        t_idx = inter['trans_idx']
        
        interaction_term = X_struct_scaled[:, s_idx] * X_trans_scaled[:, t_idx]
        interaction_features.append(interaction_term)
    
    X_features = np.hstack([X_features, np.column_stack(interaction_features)])
    log(f"Total features: {X_features.shape[1]} ({len(structural_features)} struct + {len(transcriptomic_features)} trans + {len(selected_interactions)} interactions)")
else:
    log(f"Total features: {X_features.shape[1]} ({len(structural_features)} struct + {len(transcriptomic_features)} trans)")

log("\n5. Training final EGFR model...")

final_model = xgb.XGBRegressor(**optimal_params, random_state=42)
final_model.fit(X_features, y, verbose=False)

log("Model training complete")

log("\n6. Evaluating model performance...")

y_pred = final_model.predict(X_features)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
pearson_r, pearson_p = pearsonr(y, y_pred)
spearman_r, spearman_p = spearmanr(y, y_pred)

log(f"\nFull dataset performance:")
log(f"  R² = {r2:.4f}")
log(f"  RMSE = {rmse:.4f}")
log(f"  MAE = {mae:.4f}")
log(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.4e})")
log(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.4e})")

residuals = y - y_pred
log(f"\nResidual statistics:")
log(f"  Mean = {np.mean(residuals):.4f}")
log(f"  Std = {np.std(residuals):.4f}")
log(f"  Min = {np.min(residuals):.4f}")
log(f"  Max = {np.max(residuals):.4f}")

log("\n7. Feature importance analysis...")

importances = final_model.feature_importances_

struct_imp = importances[:len(structural_features)]
trans_imp = importances[len(structural_features):len(structural_features)+len(transcriptomic_features)]

if use_interactions and selected_interactions:
    interact_imp = importances[len(structural_features)+len(transcriptomic_features):]
else:
    interact_imp = np.array([])

log(f"\nTop 10 structural features:")
top_struct_idx = np.argsort(struct_imp)[-10:][::-1]
for i, idx in enumerate(top_struct_idx):
    log(f"  {i+1}. {structural_features[idx]}: {struct_imp[idx]:.4f}")

log(f"\nTop 10 transcriptomic features:")
top_trans_idx = np.argsort(trans_imp)[-10:][::-1]
for i, idx in enumerate(top_trans_idx):
    log(f"  {i+1}. {transcriptomic_features[idx]}: {trans_imp[idx]:.4f}")

if len(interact_imp) > 0:
    log(f"\nInteraction term importances:")
    for i, inter in enumerate(selected_interactions):
        log(f"  {i+1}. {inter['struct_feature']} × {inter['trans_feature']}: {interact_imp[i]:.4f}")

log("\n8. Saving final model...")

model_package = {
    'model': final_model,
    'optimal_params': optimal_params,
    'structural_features': structural_features,
    'transcriptomic_features': transcriptomic_features,
    'scaler_struct': scaler_struct,
    'scaler_trans': scaler_trans,
    'imputer_struct': imputer_struct,
    'imputer_trans': imputer_trans,
    'use_interactions': use_interactions,
    'selected_interactions': selected_interactions,
    'performance': {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r)
    },
    'cv_performance': {
        'mean_r2': opt_summary['optimized_mean_r2'],
        'std_r2': opt_summary['optimized_std_r2']
    },
    'training_date': datetime.now().isoformat(),
    'target': 'EGFR',
    'n_samples': len(y)
}

final_model_file = MODELS_DIR / "caspid_egfr_final.pkl"
with open(final_model_file, 'wb') as f:
    pickle.dump(model_package, f)

log(f"✓ Saved: {final_model_file}")

log("\n9. Generating predictions for interpretation...")

predictions_df = pd.DataFrame({
    'CCLE_Name': cell_lines if cell_lines is not None else range(len(y)),
    'DRUG_NAME': drug_names if drug_names is not None else ['Unknown'] * len(y),
    'actual_LN_IC50': y,
    'predicted_LN_IC50': y_pred,
    'residual': residuals,
    'abs_residual': np.abs(residuals)
})

predictions_df = predictions_df.sort_values('abs_residual', ascending=False)

predictions_file = MODELS_DIR / "egfr_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
log(f"✓ Saved: {predictions_file}")

log("\n10. Model summary...")

summary = {
    'timestamp': datetime.now().isoformat(),
    'target': 'EGFR',
    'n_samples': int(len(y)),
    'n_structural_features': len(structural_features),
    'n_transcriptomic_features': len(transcriptomic_features),
    'n_interactions': len(selected_interactions) if use_interactions else 0,
    'total_features': int(X_features.shape[1]),
    'optimal_params': optimal_params,
    'performance': {
        'full_dataset_r2': float(r2),
        'full_dataset_rmse': float(rmse),
        'full_dataset_mae': float(mae),
        'cv_mean_r2': float(opt_summary['optimized_mean_r2']),
        'cv_std_r2': float(opt_summary['optimized_std_r2']),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r)
    },
    'top_structural_features': [
        {'feature': structural_features[idx], 'importance': float(struct_imp[idx])}
        for idx in top_struct_idx
    ],
    'top_transcriptomic_features': [
        {'feature': transcriptomic_features[idx], 'importance': float(trans_imp[idx])}
        for idx in top_trans_idx
    ]
}

summary_file = MODELS_DIR / "egfr_final_model_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved: {summary_file}")

log(f"\n{'='*70}")
log("FINAL EGFR MODEL COMPLETE")
log(f"{'='*70}")

log(f"\nModel: CASPID (Optimized)")
log(f"Target: EGFR inhibitors")
log(f"Samples: {len(y)}")
log(f"Features: {X_features.shape[1]}")
log(f"Performance: R² = {r2:.4f} (CV: {opt_summary['optimized_mean_r2']:.4f} ± {opt_summary['optimized_std_r2']:.4f})")

log(f"\nFiles saved:")
log(f"  Model: {final_model_file}")
log(f"  Predictions: {predictions_file}")
log(f"  Summary: {summary_file}")

log(f"\n✓ Ready for BRAF validation (Phase 8)")
log("="*70)
