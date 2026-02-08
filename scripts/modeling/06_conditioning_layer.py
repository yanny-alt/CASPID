"""
CASPID Conditioning Layer - Proper End-to-End Training
========================================================

This notebook implements the neural conditioning layer with:
- 5 different architectures
- End-to-end differentiable training
- GPU acceleration
- Proper evaluation against baselines

Upload: caspid_conditioning_data.csv, feature_info.json, baseline_performance.json
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================
# 1. LOAD DATA
# ============================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('/kaggle/input/caspid-conditioning-layer-training-data/caspid_conditioning_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

struct_cols = [c for c in df.columns if c.startswith('struct_')]
trans_cols = [c for c in df.columns if c.startswith('trans_')]

X_struct = df[struct_cols].values
X_trans = df[trans_cols].values
y = df['target'].values
strat_bins = df['strat_bin'].values

print(f"\nStructural features: {X_struct.shape}")
print(f"Transcriptomic features: {X_trans.shape}")
print(f"Target: {y.shape}")

with open('/kaggle/input/caspid-conditioning-layer-training-data/baseline_performance.json') as f:
    baseline_info = json.load(f)

baseline_r2 = baseline_info['baseline_concatenation_r2']
target_r2 = baseline_info['success_threshold']

print(f"\nBaseline R²: {baseline_r2:.4f}")
print(f"Target R²: {target_r2:.4f}")
print(f"Required improvement: {target_r2 - baseline_r2:.4f}")

# ============================================
# 2. DEFINE CONDITIONING ARCHITECTURES
# ============================================

print("\n" + "="*70)
print("DEFINING ARCHITECTURES")
print("="*70)

def build_conditioning_v1(n_trans, n_struct, name="V1_Original"):
    """Original: Dense -> Dropout -> Dense -> Sigmoid"""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    # Conditioning network
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.Dropout(0.3)(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='weights')(h)
    
    # Scale weights to [0, 2]
    weights_scaled = layers.Lambda(lambda x: x * 2.0)(weights)
    
    # Apply conditioning
    conditioned = layers.Multiply()([struct_input, weights_scaled])
    
    # Prediction network
    combined = layers.Concatenate()([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    h = layers.Dropout(0.2)(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(inputs=[trans_input, struct_input], outputs=output, name=name)
    return model

def build_conditioning_v2(n_trans, n_struct, name="V2_Deeper"):
    """Deeper: Dense -> Dense -> Dropout -> Dense -> Sigmoid"""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    h = layers.Dense(256, activation='relu')(trans_input)
    h = layers.Dense(128, activation='relu')(h)
    h = layers.Dropout(0.3)(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='weights')(h)
    
    weights_scaled = layers.Lambda(lambda x: x * 2.0)(weights)
    conditioned = layers.Multiply()([struct_input, weights_scaled])
    
    combined = layers.Concatenate()([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(inputs=[trans_input, struct_input], outputs=output, name=name)
    return model

def build_conditioning_v3(n_trans, n_struct, name="V3_BatchNorm"):
    """With BatchNorm: Dense -> BN -> Dropout -> Dense -> Sigmoid"""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='weights')(h)
    
    weights_scaled = layers.Lambda(lambda x: x * 2.0)(weights)
    conditioned = layers.Multiply()([struct_input, weights_scaled])
    
    combined = layers.Concatenate()([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation='relu')(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(inputs=[trans_input, struct_input], outputs=output, name=name)
    return model

def build_conditioning_v4(n_trans, n_struct, name="V4_Residual"):
    """Residual: weights = 0.5 + sigmoid (prevents collapse to zero)"""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(n_struct)(h)
    weights = layers.Activation('sigmoid', name='weights')(h)
    
    # Scale to [0.5, 1.5] instead of [0, 2]
    weights_scaled = layers.Lambda(lambda x: 0.5 + x)(weights)
    conditioned = layers.Multiply()([struct_input, weights_scaled])
    
    combined = layers.Concatenate()([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(inputs=[trans_input, struct_input], outputs=output, name=name)
    return model

def build_conditioning_v5(n_trans, n_struct, name="V5_Attention"):
    """Attention-based conditioning (FIXED)"""
    
    trans_input = layers.Input(shape=(n_trans,), name='trans_input')
    struct_input = layers.Input(shape=(n_struct,), name='struct_input')
    
    # Simple attention substitute
    h = layers.Dense(128, activation='relu')(trans_input)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(64, activation='relu')(h)
    weights = layers.Dense(n_struct, activation='sigmoid', name='weights')(h)
    
    weights_scaled = layers.Lambda(lambda x: x * 2.0)(weights)
    conditioned = layers.Multiply()([struct_input, weights_scaled])
    
    combined = layers.Concatenate()([conditioned, trans_input])
    h = layers.Dense(256, activation='relu')(combined)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    output = layers.Dense(1, name='output')(h)
    
    model = models.Model(inputs=[trans_input, struct_input], outputs=output, name=name)
    return model

architectures = {
    'V1_Original': build_conditioning_v1,
    'V2_Deeper': build_conditioning_v2,
    'V3_BatchNorm': build_conditioning_v3,
    'V4_Residual': build_conditioning_v4,
    'V5_Attention': build_conditioning_v5
}

print(f"Defined {len(architectures)} architectures")
for name in architectures.keys():
    print(f"  - {name}")

# ============================================
# 3. TRAINING FUNCTION
# ============================================

def train_conditioning_model(build_fn, X_struct_train, X_trans_train, y_train,
                             X_struct_val, X_trans_val, y_val,
                             epochs=100, batch_size=256, name="model"):
    """Train conditioning model with early stopping"""
    
    scaler_struct = StandardScaler()
    X_struct_train_scaled = scaler_struct.fit_transform(X_struct_train)
    X_struct_val_scaled = scaler_struct.transform(X_struct_val)
    
    scaler_trans = StandardScaler()
    X_trans_train_scaled = scaler_trans.fit_transform(X_trans_train)
    X_trans_val_scaled = scaler_trans.transform(X_trans_val)
    
    model = build_fn(X_trans.shape[1], X_struct.shape[1], name=name)
    
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
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred_val = model.predict([X_trans_val_scaled, X_struct_val_scaled], verbose=0).ravel()
    val_r2 = r2_score(y_val, y_pred_val)
    
    return model, val_r2, history, scaler_struct, scaler_trans

# ============================================
# 4. CROSS-VALIDATION EVALUATION
# ============================================

print("\n" + "="*70)
print("CROSS-VALIDATION EVALUATION")
print("="*70)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

architecture_results = {}

for arch_name, build_fn in architectures.items():
    print(f"\n{arch_name}:")
    print("-" * 50)
    
    fold_r2_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_struct, strat_bins)):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ')
        
        X_struct_train = X_struct[train_idx]
        X_trans_train = X_trans[train_idx]
        y_train = y[train_idx]
        
        X_struct_val = X_struct[val_idx]
        X_trans_val = X_trans[val_idx]
        y_val = y[val_idx]
        
        model, val_r2, history, scaler_s, scaler_t = train_conditioning_model(
            build_fn,
            X_struct_train, X_trans_train, y_train,
            X_struct_val, X_trans_val, y_val,
            epochs=100,
            batch_size=256,
            name=f"{arch_name}_fold{fold_idx}"
        )
        
        fold_r2_scores.append(val_r2)
        print(f"R² = {val_r2:.4f}")
    
    mean_r2 = np.mean(fold_r2_scores)
    std_r2 = np.std(fold_r2_scores)
    
    architecture_results[arch_name] = {
        'fold_scores': fold_r2_scores,
        'mean_r2': mean_r2,
        'std_r2': std_r2
    }
    
    improvement = mean_r2 - baseline_r2
    
    print(f"  Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  vs Baseline: ΔR² = {improvement:+.4f}")

# ============================================
# 5. RESULTS SUMMARY
# ============================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame([
    {
        'Architecture': name,
        'Mean_R2': results['mean_r2'],
        'Std_R2': results['std_r2'],
        'vs_Baseline': results['mean_r2'] - baseline_r2,
        'Fold_1': results['fold_scores'][0],
        'Fold_2': results['fold_scores'][1],
        'Fold_3': results['fold_scores'][2],
        'Fold_4': results['fold_scores'][3],
        'Fold_5': results['fold_scores'][4]
    }
    for name, results in architecture_results.items()
]).sort_values('Mean_R2', ascending=False)

print(results_df[['Architecture', 'Mean_R2', 'Std_R2', 'vs_Baseline']].to_string(index=False))

best_arch = results_df.iloc[0]['Architecture']
best_r2 = results_df.iloc[0]['Mean_R2']
best_improvement = results_df.iloc[0]['vs_Baseline']

print(f"\n{'='*70}")
print(f"BEST: {best_arch}")
print(f"  R² = {best_r2:.4f}")
print(f"  Baseline = {baseline_r2:.4f}")
print(f"  Improvement = {best_improvement:+.4f}")
print(f"{'='*70}")

if best_r2 > target_r2:
    print(f"\n✅ SUCCESS! Exceeded target R² = {target_r2:.4f}")
elif best_r2 > baseline_r2 + 0.02:
    print(f"\n✅ GOOD! Significant improvement over baseline")
elif best_r2 > baseline_r2:
    print(f"\n⚠️  MARGINAL: Small improvement")
else:
    print(f"\n❌ NEGATIVE: No improvement over baseline")

# ============================================
# 6. VISUALIZATION
# ============================================

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
results_df.plot(x='Architecture', y='Mean_R2', kind='bar', 
                yerr='Std_R2', ax=plt.gca(), legend=False)
plt.axhline(y=baseline_r2, color='r', linestyle='--', label=f'Baseline ({baseline_r2:.4f})')
plt.axhline(y=target_r2, color='g', linestyle='--', label=f'Target ({target_r2:.4f})')
plt.ylabel('R² Score')
plt.title('Architecture Comparison')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
fold_data = []
for name, results in architecture_results.items():
    for score in results['fold_scores']:
        fold_data.append({'Architecture': name, 'R2': score})
fold_df = pd.DataFrame(fold_data)
sns.boxplot(data=fold_df, x='Architecture', y='R2')
plt.axhline(y=baseline_r2, color='r', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.title('R² Distribution Across Folds')
plt.tight_layout()

plt.savefig('conditioning_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: conditioning_results.png")

# ============================================
# 7. SAVE RESULTS
# ============================================

results_df.to_csv('conditioning_architecture_results.csv', index=False)
print("✓ Saved results: conditioning_architecture_results.csv")

summary = {
    'baseline_r2': float(baseline_r2),
    'target_r2': float(target_r2),
    'best_architecture': best_arch,
    'best_r2': float(best_r2),
    'best_improvement': float(best_improvement),
    'all_results': {
        name: {
            'mean_r2': float(results['mean_r2']),
            'std_r2': float(results['std_r2']),
            'fold_scores': [float(x) for x in results['fold_scores']]
        }
        for name, results in architecture_results.items()
    }
}

with open('conditioning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved summary: conditioning_summary.json")

print("\n" + "="*70)
print("CONDITIONING LAYER EVALUATION COMPLETE")
print("="*70)
