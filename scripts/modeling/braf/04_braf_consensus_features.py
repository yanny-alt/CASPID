#!/usr/bin/env python3
"""
04_consensus_features.py

Combines results from Boruta, MI, and SHAP to create consensus feature sets.
Creates THREE feature sets for different use cases:
- Set A: Strict consensus (≥2 methods)
- Set B: Expanded structural (top features from all methods)
- Set C: Biologically-informed (Set B + key binding features)

CRITICAL: This determines which features go to the conditioning layer.

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

# For Venn diagram
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3, venn3_circles
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib/matplotlib-venn not installed - skipping Venn diagram")

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
MODELING_DIR = PROJECT_ROOT / "DATA" / "MODELING" / "braf"
CONSENSUS_DIR = MODELING_DIR / "04_consensus"
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = CONSENSUS_DIR / f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("CONSENSUS FEATURE SELECTION")
print("="*70)

# ============================================
# STEP 1: LOAD ALL SELECTION RESULTS
# ============================================

log("\n1. Loading feature selection results...")

# Load all docking features
docking_features_file = MODELING_DIR / "docking_features.txt"
with open(docking_features_file, 'r') as f:
    all_docking_features = [line.strip() for line in f if line.strip()]

log(f"✓ Total docking features: {len(all_docking_features)}")

# Load Boruta results
boruta_file = MODELING_DIR / "01_boruta" / "boruta_confirmed_features.txt"
if not boruta_file.exists():
    log(f"✗ ERROR: Boruta results not found at {boruta_file}")
    exit(1)

with open(boruta_file, 'r') as f:
    boruta_features = set(line.strip() for line in f if line.strip())

log(f"✓ Boruta confirmed: {len(boruta_features)} features")

# Load MI results
mi_file = MODELING_DIR / "02_mi" / "mi_selected_features.txt"
if not mi_file.exists():
    log(f"✗ ERROR: MI results not found at {mi_file}")
    exit(1)

with open(mi_file, 'r') as f:
    mi_features = set(line.strip() for line in f if line.strip())

log(f"✓ MI selected: {len(mi_features)} features")

# Load MI scores for ranking
mi_scores_file = MODELING_DIR / "02_mi" / "mi_all_scores.csv"
mi_scores_df = pd.read_csv(mi_scores_file)

# Load SHAP results
shap_file = MODELING_DIR / "03_shap" / "shap_selected_features.txt"
if not shap_file.exists():
    log(f"✗ ERROR: SHAP results not found at {shap_file}")
    exit(1)

with open(shap_file, 'r') as f:
    shap_features = set(line.strip() for line in f if line.strip())

log(f"✓ SHAP selected: {len(shap_features)} features")

# Load SHAP scores for ranking
shap_scores_file = MODELING_DIR / "03_shap" / "shap_all_scores.csv"
shap_scores_df = pd.read_csv(shap_scores_file)

# ============================================
# STEP 2: CALCULATE OVERLAPS
# ============================================

log("\n2. Calculating method overlaps...")

# All pairwise overlaps
boruta_mi = boruta_features & mi_features
boruta_shap = boruta_features & shap_features
mi_shap = mi_features & shap_features

# All three
all_three = boruta_features & mi_features & shap_features

log(f"\n✓ Pairwise overlaps:")
log(f"  Boruta ∩ MI: {len(boruta_mi)}")
log(f"  Boruta ∩ SHAP: {len(boruta_shap)}")
log(f"  MI ∩ SHAP: {len(mi_shap)}")
log(f"  All three methods: {len(all_three)}")

# ============================================
# STEP 3: CREATE SET A - STRICT CONSENSUS
# ============================================

log("\n3. Creating Set A: Strict Consensus (≥2 methods)...")

# Features selected by at least 2 methods
set_a = set()

# Count how many methods selected each feature
feature_counts = {}
for feat in all_docking_features:
    count = 0
    if feat in boruta_features:
        count += 1
    if feat in mi_features:
        count += 1
    if feat in shap_features:
        count += 1
    
    if count >= 2:
        set_a.add(feat)
    
    feature_counts[feat] = count

log(f"\n✓ Set A: {len(set_a)} features")
log(f"  Selected by all 3 methods: {len(all_three)}")
log(f"  Selected by exactly 2 methods: {len(set_a) - len(all_three)}")

# Show which features
if len(all_three) > 0:
    log(f"\n  Features selected by ALL 3 methods:")
    for feat in sorted(all_three):
        log(f"    - {feat}")

two_methods = set_a - all_three
if len(two_methods) > 0:
    log(f"\n  Features selected by exactly 2 methods:")
    for feat in sorted(two_methods):
        methods = []
        if feat in boruta_features:
            methods.append("Boruta")
        if feat in mi_features:
            methods.append("MI")
        if feat in shap_features:
            methods.append("SHAP")
        log(f"    - {feat} ({' + '.join(methods)})")

# ============================================
# STEP 4: CREATE SET B - EXPANDED STRUCTURAL
# ============================================

log("\n4. Creating Set B: Expanded Structural (top 30-40 features)...")

# Strategy: Take top 30 from MI + all Boruta + all SHAP, remove duplicates
set_b = set()

# Add all Boruta (strict, no false positives)
set_b.update(boruta_features)
log(f"  Added {len(boruta_features)} Boruta features")

# Add top 30 from MI
mi_top30 = mi_scores_df.head(30)['feature'].values
set_b.update(mi_top30)
log(f"  Added top 30 MI features")

# Add all SHAP
set_b.update(shap_features)
log(f"  Added {len(shap_features)} SHAP features")

log(f"\n✓ Set B: {len(set_b)} features (after removing duplicates)")

# ============================================
# STEP 5: IDENTIFY KEY BINDING FEATURES
# ============================================

log("\n5. Identifying key binding geometry features...")

# Define binding-related feature patterns
binding_patterns = {
    'hinge': ['hinge_MET769', 'hinge_GLN767', 'hinge_LEU768'],
    'gatekeeper': ['gatekeeper_THR790', 'gatekeeper_THR529'],
    'dfg': ['dfg_ASP831', 'dfg_PHE832', 'dfg_GLY833', 'dfg_ASP594', 'dfg_PHE595', 'dfg_GLY596'],
    'p_loop': ['p_loop_GLY695', 'p_loop_GLY697', 'p_loop_GLY696', 'p_loop_GLY463', 'p_loop_GLY464'],
    'c_helix': ['c_helix_GLU738', 'c_helix_LYS721', 'c_helix_GLU501', 'c_helix_LYS483']
}

# Find best binding features from SHAP scores
binding_features = {}

for region, patterns in binding_patterns.items():
    # Find all features matching these patterns
    region_features = []
    for feat in all_docking_features:
        if any(pattern in feat for pattern in patterns):
            # Get SHAP score
            shap_score = shap_scores_df[shap_scores_df['feature'] == feat]['shap_importance'].values
            if len(shap_score) > 0:
                region_features.append((feat, shap_score[0]))
    
    # Sort by SHAP score and take top 2-3
    region_features.sort(key=lambda x: x[1], reverse=True)
    binding_features[region] = [f[0] for f in region_features[:2]]  # Top 2 per region

# Flatten
key_binding_features = []
for region, features in binding_features.items():
    key_binding_features.extend(features)
    if features:
        log(f"  {region}: {len(features)} features")
        for feat in features:
            shap_score = shap_scores_df[shap_scores_df['feature'] == feat]['shap_importance'].values[0]
            log(f"    - {feat} (SHAP: {shap_score:.6f})")

log(f"\n✓ Identified {len(key_binding_features)} key binding features")

# ============================================
# STEP 6: CREATE SET C - BIOLOGICALLY INFORMED
# ============================================

log("\n6. Creating Set C: Biologically-Informed (Set B + binding)...")

# Start with Set B
set_c = set_b.copy()

# Add key binding features
set_c.update(key_binding_features)

log(f"\n✓ Set C: {len(set_c)} features")
log(f"  Base (Set B): {len(set_b)}")
log(f"  Added binding features: {len(set_c) - len(set_b)}")

# ============================================
# STEP 7: CREATE FEATURE RANKINGS
# ============================================

log("\n7. Creating comprehensive feature rankings...")

# Create ranking dataframe with scores from all methods
ranking_data = []

for feat in all_docking_features:
    # Get scores
    mi_score = mi_scores_df[mi_scores_df['feature'] == feat]['mi_score'].values
    mi_score = mi_score[0] if len(mi_score) > 0 else 0.0
    
    shap_score = shap_scores_df[shap_scores_df['feature'] == feat]['shap_importance'].values
    shap_score = shap_score[0] if len(shap_score) > 0 else 0.0
    
    # Selection status
    in_boruta = feat in boruta_features
    in_mi = feat in mi_features
    in_shap = feat in shap_features
    
    # Which sets
    in_set_a = feat in set_a
    in_set_b = feat in set_b
    in_set_c = feat in set_c
    
    ranking_data.append({
        'feature': feat,
        'num_methods': feature_counts[feat],
        'boruta': in_boruta,
        'mi': in_mi,
        'shap': in_shap,
        'mi_score': mi_score,
        'shap_importance': shap_score,
        'set_a_strict': in_set_a,
        'set_b_expanded': in_set_b,
        'set_c_biological': in_set_c
    })

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values(['num_methods', 'mi_score'], ascending=[False, False])

log(f"\n✓ Created comprehensive ranking table")

# ============================================
# STEP 8: CREATE VENN DIAGRAM
# ============================================

log("\n8. Creating Venn diagram...")

if HAS_MATPLOTLIB:
    try:
        plt.figure(figsize=(10, 8))
        
        # Create Venn diagram
        venn = venn3(
            [boruta_features, mi_features, shap_features],
            set_labels=(f'Boruta\n(n={len(boruta_features)})',
                        f'MI\n(n={len(mi_features)})',
                        f'SHAP\n(n={len(shap_features)})')
        )
        
        # Customize
        venn3_circles([boruta_features, mi_features, shap_features], linewidth=1.5)
        
        plt.title('Feature Selection Method Overlap', fontsize=14, fontweight='bold')
        
        # Save
        venn_file = CONSENSUS_DIR / "feature_selection_venn.png"
        plt.savefig(venn_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        log(f"✓ Saved Venn diagram: {venn_file}")
    except Exception as e:
        log(f"⚠️  Could not create Venn diagram: {e}")
else:
    log("⚠️  Skipping Venn diagram (matplotlib not installed)")

# ============================================
# STEP 9: SAVE ALL RESULTS
# ============================================

log("\n9. Saving consensus results...")

# Save Set A
set_a_file = CONSENSUS_DIR / "set_a_strict_consensus.txt"
with open(set_a_file, 'w') as f:
    f.write('\n'.join(sorted(set_a)))
log(f"✓ Saved Set A: {set_a_file}")

# Save Set B
set_b_file = CONSENSUS_DIR / "set_b_expanded_structural.txt"
with open(set_b_file, 'w') as f:
    f.write('\n'.join(sorted(set_b)))
log(f"✓ Saved Set B: {set_b_file}")

# Save Set C
set_c_file = CONSENSUS_DIR / "set_c_biological.txt"
with open(set_c_file, 'w') as f:
    f.write('\n'.join(sorted(set_c)))
log(f"✓ Saved Set C: {set_c_file}")

# Save ranking table
ranking_file = CONSENSUS_DIR / "feature_ranking_comprehensive.csv"
ranking_df.to_csv(ranking_file, index=False)
log(f"✓ Saved comprehensive ranking: {ranking_file}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_features': len(all_docking_features),
    'boruta_selected': len(boruta_features),
    'mi_selected': len(mi_features),
    'shap_selected': len(shap_features),
    'overlaps': {
        'boruta_mi': len(boruta_mi),
        'boruta_shap': len(boruta_shap),
        'mi_shap': len(mi_shap),
        'all_three': len(all_three)
    },
    'feature_sets': {
        'set_a_strict': {
            'n_features': len(set_a),
            'description': 'Features selected by ≥2 methods'
        },
        'set_b_expanded': {
            'n_features': len(set_b),
            'description': 'Top features from all methods combined'
        },
        'set_c_biological': {
            'n_features': len(set_c),
            'description': 'Set B + key binding geometry features'
        }
    },
    'key_binding_features': key_binding_features
}

summary_file = CONSENSUS_DIR / "consensus_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"✓ Saved summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("CONSENSUS FEATURE SELECTION COMPLETE")
log("="*70)

log(f"\n📊 METHOD RESULTS:")
log(f"   Boruta: {len(boruta_features)} features")
log(f"   MI: {len(mi_features)} features")
log(f"   SHAP: {len(shap_features)} features")

log(f"\n📊 OVERLAPS:")
log(f"   All 3 methods: {len(all_three)} features")
log(f"   Any 2 methods: {len(set_a)} features")

log(f"\n📊 FINAL FEATURE SETS:")
log(f"   Set A (Strict Consensus): {len(set_a)} features")
log(f"   Set B (Expanded Structural): {len(set_b)} features")
log(f"   Set C (Biologically-Informed): {len(set_c)} features")

log(f"\n📁 OUTPUT FILES:")
log(f"   Set A: {set_a_file}")
log(f"   Set B: {set_b_file}")
log(f"   Set C: {set_c_file}")
log(f"   Ranking: {ranking_file}")
log(f"   Summary: {summary_file}")

log(f"\n✅ RECOMMENDATION:")
log(f"   For conditioning layer: Start with Set B ({len(set_b)} features)")
log(f"   For ablation studies: Test all 3 sets")
log(f"   For paper: Report Set A as consensus ({len(set_a)} features)")

log("\n" + "="*70)
log("Ready for Phase 6: Neural Conditioning Layer")
log("="*70)
