"""
CASPID Full Model Training
===========================

Trains complete CASPID framework using V3_BatchNorm conditioning architecture
(selected from architecture comparison) with XGBoost downstream predictor.

Compares against baselines:
- Structure only
- Transcriptomics only  
- Simple concatenation (no conditioning)

Input: caspid_conditioning_data.csv, baseline_performance.json
Output: Model performance metrics, trained models, predictions
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

import xgboost as xgb

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')))

# ============================================
# LOAD DATA
# ============================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('/kaggle/input/caspid-conditioning-layer-training-data/caspid_conditioning_data.csv')

struct_cols = [c for c in df.columns if c.startswith('struct_')]
trans_cols = [c for c in df.columns if c.startswith('trans_')]

X_struct = df[struct_cols].values
X_trans = df[trans_cols].values
y = df['target'].values
strat_bins = df['strat_bin'].values

n_samples = len(y)
n_struct = len(struct_cols)
n_trans = len(trans_cols)

print(f"Samples: {n_samples}")
print(f"Structural features: {n_struct}")
print(f"Transcriptomic features: {n_trans}")

with open('/kaggle/input/caspid-conditioning-layer-training-data/baseline_performance.json') as f:
    baseline_info = json.load(f)

baseline_concat_r2 = baseline_info['baseline_concatenation_r2']
baseline_struct_r2 = baseline_info['structure_only_r2']
baseline_trans_r2 = baseline_info['transcriptomics_only_r2']

print(f"\nBaseline R² scores:")
print(f"  Structure only: {baseline_struct_r2:.4f}")
print(f"  Transcriptomics only: {baseline_trans_r2:.4f}")
print(f"  Concatenation: {baseline_concat_r2:.4f}")

# ============================================
# DEFINE V3_BATCHNORM ARCHITECTURE
# ============================================

print("\n" + "="*70)
print("DEFINING CONDITIONING ARCHITECTURE")
print("="*70)

def build_v3_batchnorm(n_trans, n_struct):
    """
    V3_BatchNorm architecture (winner from architecture comparison).
    Uses batch normalization for stable training.
    """
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    # Conditioning network: transcriptomics -> weights
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='conditioning_weights')(h)
    
    # Scale to [0, 2] range
    weights_scaled = layers.Lambda(lambda x: x * 2.0, name='weight_scaling')(weights)
    
    # Apply conditioning to structural features
    conditioned = layers.Multiply(name='feature_conditioning')([struct_input, weights_scaled])
    
    # Downstream prediction network
    combined = layers.Concatenate(name='feature_concatenation')([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation='relu')(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(
        inputs=[trans_input, struct_input],
        outputs=output,
        name='V3_BatchNorm_Conditioning'
    )
    
    return model

print("Architecture: V3_BatchNorm (128 -> BN -> Dropout -> weights)")
print("Conditioning: Sigmoid weights scaled to [0, 2]")
print("Downstream: 256 -> BN -> Dropout -> 128 -> output")

# ============================================
# CONDITIONING WEIGHT EXTRACTION MODEL
# ============================================

def build_weight_extractor(n_trans, n_struct):
    """Separate model to extract learned conditioning weights."""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='weights')(h)
    weights_scaled = layers.Lambda(lambda x: x * 2.0)(weights)
    
    extractor = models.Model(inputs=trans_input, outputs=weights_scaled, name='WeightExtractor')
    
    return extractor

# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_conditioning_model(X_struct_train, X_trans_train, y_train,
                              X_struct_val, X_trans_val, y_val):
    """
    Train V3_BatchNorm conditioning model.
    Returns trained model and validation R2.
    """
    
    scaler_struct = StandardScaler()
    X_struct_train_scaled = scaler_struct.fit_transform(X_struct_train)
    X_struct_val_scaled = scaler_struct.transform(X_struct_val)
    
    scaler_trans = StandardScaler()
    X_trans_train_scaled = scaler_trans.fit_transform(X_trans_train)
    X_trans_val_scaled = scaler_trans.transform(X_trans_val)
    
    model = build_v3_batchnorm(n_trans, n_struct)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    history = model.fit(
        [X_trans_train_scaled, X_struct_train_scaled],
        y_train,
        validation_data=([X_trans_val_scaled, X_struct_val_scaled], y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred_val = model.predict(
        [X_trans_val_scaled, X_struct_val_scaled],
        verbose=0
    ).ravel()
    
    val_r2 = r2_score(y_val, y_pred_val)
    
    return model, val_r2, scaler_struct, scaler_trans

def extract_conditioned_features(model, X_trans, X_struct, scaler_trans, scaler_struct):
    """
    Extract conditioning weights and apply to structural features.
    Returns conditioned structural features for XGBoost training.
    """
    
    X_trans_scaled = scaler_trans.transform(X_trans)
    X_struct_scaled = scaler_struct.transform(X_struct)
    
    # Extract conditioning layer weights
    weight_model = models.Model(
        inputs=model.input[0],
        outputs=model.get_layer('weight_scaling').output
    )
    
    weights = weight_model.predict(X_trans_scaled, verbose=0)
    
    # Apply weights to structural features
    X_struct_conditioned = X_struct_scaled * weights
    
    return X_struct_conditioned, weights

def train_xgboost_on_conditioned(X_struct_conditioned, X_trans, y_train, y_val, 
                                  train_idx, val_idx):
    """
    Train XGBoost on conditioned structural features + transcriptomics.
    """
    
    X_train_combined = np.hstack([
        X_struct_conditioned[train_idx],
        X_trans[train_idx]
    ])
    
    X_val_combined = np.hstack([
        X_struct_conditioned[val_idx],
        X_trans[val_idx]
    ])
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_combined, y_train, verbose=False)
    
    y_pred_val = xgb_model.predict(X_val_combined)
    val_r2 = r2_score(y_val, y_pred_val)
    
    return xgb_model, val_r2

# ============================================
# CROSS-VALIDATION EVALUATION
# ============================================

print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION")
print("="*70)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {
    'caspid_nn_r2': [],
    'caspid_xgb_r2': [],
    'fold': []
}

fold_predictions = []
fold_weights = []

print("\nTraining CASPID (Conditioning + XGBoost):")
print("-" * 50)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, strat_bins)):
    print(f"Fold {fold_idx+1}/{n_folds}...", end=' ')
    
    X_struct_train = X_struct[train_idx]
    X_trans_train = X_trans[train_idx]
    y_train = y[train_idx]
    
    X_struct_val = X_struct[val_idx]
    X_trans_val = X_trans[val_idx]
    y_val = y[val_idx]
    
    # Train conditioning model (neural network)
    cond_model, nn_r2, scaler_s, scaler_t = train_conditioning_model(
        X_struct_train, X_trans_train, y_train,
        X_struct_val, X_trans_val, y_val
    )
    
    # Extract conditioned features for entire dataset
    X_struct_conditioned, weights = extract_conditioned_features(
        cond_model, X_trans, X_struct, scaler_t, scaler_s
    )
    
    # Train XGBoost on conditioned features
    xgb_model, xgb_r2 = train_xgboost_on_conditioned(
        X_struct_conditioned, X_trans, y_train, y_val,
        train_idx, val_idx
    )
    
    results['caspid_nn_r2'].append(nn_r2)
    results['caspid_xgb_r2'].append(xgb_r2)
    results['fold'].append(fold_idx)
    
    fold_weights.append(weights)
    
    print(f"NN R²={nn_r2:.4f}, XGB R²={xgb_r2:.4f}")

# ============================================
# BASELINE COMPARISONS
# ============================================

print("\n" + "="*70)
print("BASELINE MODEL TRAINING")
print("="*70)

baseline_results = {
    'struct_only_r2': [],
    'trans_only_r2': [],
    'concat_r2': []
}

print("\nTraining baselines for direct comparison:")
print("-" * 50)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, strat_bins)):
    print(f"Fold {fold_idx+1}/{n_folds}...", end=' ')
    
    X_struct_train = X_struct[train_idx]
    X_trans_train = X_trans[train_idx]
    y_train = y[train_idx]
    
    X_struct_val = X_struct[val_idx]
    X_trans_val = X_trans[val_idx]
    y_val = y[val_idx]
    
    # Standardize
    scaler_s = StandardScaler()
    X_struct_train_sc = scaler_s.fit_transform(X_struct_train)
    X_struct_val_sc = scaler_s.transform(X_struct_val)
    
    scaler_t = StandardScaler()
    X_trans_train_sc = scaler_t.fit_transform(X_trans_train)
    X_trans_val_sc = scaler_t.transform(X_trans_val)
    
    # Structure only
    model_s = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1)
    model_s.fit(X_struct_train_sc, y_train, verbose=False)
    struct_r2 = r2_score(y_val, model_s.predict(X_struct_val_sc))
    
    # Transcriptomics only
    model_t = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1)
    model_t.fit(X_trans_train_sc, y_train, verbose=False)
    trans_r2 = r2_score(y_val, model_t.predict(X_trans_val_sc))
    
    # Simple concatenation
    X_train_concat = np.hstack([X_struct_train_sc, X_trans_train_sc])
    X_val_concat = np.hstack([X_struct_val_sc, X_trans_val_sc])
    
    model_c = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1)
    model_c.fit(X_train_concat, y_train, verbose=False)
    concat_r2 = r2_score(y_val, model_c.predict(X_val_concat))
    
    baseline_results['struct_only_r2'].append(struct_r2)
    baseline_results['trans_only_r2'].append(trans_r2)
    baseline_results['concat_r2'].append(concat_r2)
    
    print(f"S={struct_r2:.4f}, T={trans_r2:.4f}, C={concat_r2:.4f}")

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

summary_stats = {
    'Structure Only': {
        'mean': np.mean(baseline_results['struct_only_r2']),
        'std': np.std(baseline_results['struct_only_r2']),
        'scores': baseline_results['struct_only_r2']
    },
    'Transcriptomics Only': {
        'mean': np.mean(baseline_results['trans_only_r2']),
        'std': np.std(baseline_results['trans_only_r2']),
        'scores': baseline_results['trans_only_r2']
    },
    'Concatenation': {
        'mean': np.mean(baseline_results['concat_r2']),
        'std': np.std(baseline_results['concat_r2']),
        'scores': baseline_results['concat_r2']
    },
    'CASPID (NN only)': {
        'mean': np.mean(results['caspid_nn_r2']),
        'std': np.std(results['caspid_nn_r2']),
        'scores': results['caspid_nn_r2']
    },
    'CASPID (Full)': {
        'mean': np.mean(results['caspid_xgb_r2']),
        'std': np.std(results['caspid_xgb_r2']),
        'scores': results['caspid_xgb_r2']
    }
}

print("\nModel Performance (Mean R² ± SD):")
print("-" * 50)
for model_name, stats in summary_stats.items():
    improvement = stats['mean'] - baseline_concat_r2
    print(f"{model_name:25s} R² = {stats['mean']:.4f} ± {stats['std']:.4f}  (Δ = {improvement:+.4f})")

best_model = max(summary_stats.keys(), key=lambda k: summary_stats[k]['mean'])
best_r2 = summary_stats[best_model]['mean']

print(f"\nBest model: {best_model}")
print(f"R² = {best_r2:.4f}")

if best_r2 > baseline_concat_r2 + 0.05:
    print("\n✅ SUCCESS: CASPID significantly outperforms concatenation")
elif best_r2 > baseline_concat_r2 + 0.02:
    print("\n✅ GOOD: CASPID shows meaningful improvement")
elif best_r2 > baseline_concat_r2:
    print("\n⚠️  MARGINAL: Small improvement observed")
else:
    print("\n❌ NEGATIVE: No improvement over concatenation")

# ============================================
# VISUALIZATION
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot with error bars
ax1 = axes[0]
model_names = list(summary_stats.keys())
means = [summary_stats[m]['mean'] for m in model_names]
stds = [summary_stats[m]['std'] for m in model_names]

bars = ax1.bar(range(len(model_names)), means, yerr=stds, capsize=5,
               color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])

ax1.axhline(y=baseline_concat_r2, color='gray', linestyle='--', 
            linewidth=1, label=f'Baseline Concat ({baseline_concat_r2:.3f})')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Comparison (5-Fold CV)')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Box plot
ax2 = axes[1]
box_data = [summary_stats[m]['scores'] for m in model_names]
bp = ax2.boxplot(box_data, labels=model_names, patch_artist=True)

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.axhline(y=baseline_concat_r2, color='gray', linestyle='--', 
            linewidth=1, label=f'Baseline')
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Distribution Across Folds')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('caspid_full_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: caspid_full_results.png")

# ============================================
# SAVE RESULTS
# ============================================

results_df = pd.DataFrame({
    'Model': model_names,
    'Mean_R2': means,
    'Std_R2': stds,
    'Fold_1': [summary_stats[m]['scores'][0] for m in model_names],
    'Fold_2': [summary_stats[m]['scores'][1] for m in model_names],
    'Fold_3': [summary_stats[m]['scores'][2] for m in model_names],
    'Fold_4': [summary_stats[m]['scores'][3] for m in model_names],
    'Fold_5': [summary_stats[m]['scores'][4] for m in model_names],
    'vs_Concat': [m - baseline_concat_r2 for m in means]
})

results_df.to_csv('caspid_full_model_results.csv', index=False)
print("✓ Saved: caspid_full_model_results.csv")

summary_output = {
    'baseline_concatenation_r2': float(baseline_concat_r2),
    'baseline_structure_r2': float(baseline_struct_r2),
    'baseline_transcriptomics_r2': float(baseline_trans_r2),
    'models': {
        name: {
            'mean_r2': float(stats['mean']),
            'std_r2': float(stats['std']),
            'improvement_vs_concat': float(stats['mean'] - baseline_concat_r2),
            'fold_scores': [float(x) for x in stats['scores']]
        }
        for name, stats in summary_stats.items()
    },
    'best_model': best_model,
    'best_r2': float(best_r2)
}

with open('caspid_full_summary.json', 'w') as f:
    json.dump(summary_output, f, indent=2)
print("✓ Saved: caspid_full_summary.json")

# Save weights for interpretation
weights_array = np.array(fold_weights)
np.save('conditioning_weights.npy', weights_array)
print("✓ Saved: conditioning_weights.npy")

print("\n" + "="*70)
print("CASPID FULL MODEL TRAINING COMPLETE")
print("="*70)
print(f"\nFinal Result: {best_model}")
print(f"R² = {best_r2:.4f} ± {summary_stats[best_model]['std']:.4f}")
print(f"Improvement over concatenation: {best_r2 - baseline_concat_r2:+.4f}")
print("="*70)
