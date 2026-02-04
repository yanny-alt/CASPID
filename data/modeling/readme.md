# Modeling Data

**Source:** Feature selection pipeline (Scripts 00-04)  
**Input:** 92 structural features + 95 transcriptomic features  
**Output:** 3 curated feature sets for machine learning

---

## Directory Structure

```
modeling/
├── full_modeling_dataset.csv              # Main dataset (23,248 × 187 features)
├── docking_features.txt                   # List of 92 structural feature names
├── transcriptomic_features.txt            # List of 95 gene expression features
├── merge_summary.json                     # Merge statistics
├── 01_boruta/
│   ├── boruta_confirmed_features.txt      # 10 strict features
│   ├── boruta_tentative_features.txt      # 0 features (none)
│   ├── boruta_all_results.csv             # Full ranking
│   ├── boruta_feature_importance.csv      # RF importance scores
│   └── boruta_summary.json
├── 02_mi/
│   ├── mi_selected_features.txt           # Top 50 features
│   ├── mi_all_scores.csv                  # All MI scores
│   ├── mi_top_features.csv                # Top 50 with scores
│   └── mi_summary.json
├── 03_shap/
│   ├── shap_selected_features.txt         # Top 5 features (95th percentile)
│   ├── shap_all_scores.csv                # All SHAP importance
│   ├── shap_top_features.csv              # Top 5 with scores
│   └── shap_summary.json
└── 04_consensus/
    ├── set_a_strict_consensus.txt         # 12 features (≥2 methods) ⭐
    ├── set_b_expanded_structural.txt      # 32 features (primary) ⭐⭐
    ├── set_c_biological.txt               # 40 features (+ binding) ⭐
    ├── feature_ranking_comprehensive.csv  # All features ranked
    ├── feature_selection_venn.png         # Overlap visualization
    └── consensus_summary.json
```

---

## Main Dataset

### full_modeling_dataset.csv
**Shape:** 23,248 rows × 193 columns

**Columns:**
- **Metadata (5):** COSMIC_ID, CELL_LINE_NAME, DRUG_NAME, protein, job_id
- **Target (1):** LN_IC50
- **Transcriptomic (95):** Gene expression features (EGFR (1956), ERBB2 (2064), ...)
- **Structural (92):** Docking features (hinge distances, molecular properties, ...)

**Sample breakdown:**
- 15 drugs
- 711 cell lines
- Drugs: 697-2,840 samples per drug (mean: 1,550)
- Cell lines: Average 32.7 samples per cell line

**Missing values:**
- 36 features have missing values (~54.6%)
- Expected: Protein-specific residues (EGFR features missing for BRAF poses)
- Handled: Median imputation in feature selection

---

## Feature Sets (Primary Outputs)

### Set A: Strict Consensus (12 features)
**File:** `04_consensus/set_a_strict_consensus.txt`

**Selection:** Features chosen by ≥2 of 3 methods (Boruta, MI, SHAP)

**Features:**
1. abs_total_charge (all 3 methods)
2. num_hydrophobic_pharmacophore (all 3 methods)
3. num_negative_ionizable (all 3 methods)
4. fraction_csp3 (Boruta + MI)
5. ligand_efficiency (MI + SHAP)
6. ligand_sasa (Boruta + MI)
7. ligand_surface_area (Boruta + MI)
8. ligand_volume (Boruta + MI)
9. lipophilic_efficiency (Boruta + MI)
10. num_heavy_atoms (MI + SHAP)
11. shape_complementarity (Boruta + MI)
12. tpsa (Boruta + MI)

**Use case:** High-confidence features for paper validation

---

### Set B: Expanded Structural (32 features) ⭐ PRIMARY
**File:** `04_consensus/set_b_expanded_structural.txt`

**Selection:** 
- All 10 Boruta confirmed (strict)
- Top 30 MI features (informative)
- All 5 SHAP features (model-important)
- Duplicates removed

**Composition:**
- Molecular properties: 18 features
- Size/shape descriptors: 8 features
- Binding geometry: 4 features
- Pharmacophore: 2 features

**Sample-to-feature ratio:** 23,248 / 32 = 1:726 (optimal for ML)

**Use case:** **Primary feature set for neural conditioning layer**

---

### Set C: Biologically-Informed (40 features)
**File:** `04_consensus/set_c_biological.txt`

**Selection:** Set B + 8 key binding geometry features

**Added binding features:**
- Hinge: hinge_MET769_min_dist, hinge_GLN767_min_dist
- DFG: dfg_PHE832_mean_dist, dfg_PHE832_min_dist
- P-loop: p_loop_GLY697_min_dist, p_loop_GLY695_min_dist
- C-helix: c_helix_LYS721_min_dist, c_helix_GLU738_min_dist

**Use case:** Test if binding-specific features improve predictions

---

## Feature Selection Results

### Method Performance

| Method | Selected | Threshold | Time |
|--------|----------|-----------|------|
| Boruta | 10 | α=0.01 | 3 min |
| MI | 50 | Top 50 | 10 sec |
| SHAP | 5 | 95th %-ile | 1 min |

### Overlaps

- All 3 methods: 3 features (very strong)
- Boruta ∩ MI: 10 (100% validation)
- Boruta ∩ SHAP: 3
- MI ∩ SHAP: 5 (100% validation)

---

## Feature Categories

### Selected Features by Type

**Molecular Properties (dominant):**
- Charges: abs_total_charge, total_charge, max_positive/negative_charge
- Lipophilicity: logp, fraction_csp3
- Polarity: tpsa, polar_fraction
- Size: num_heavy_atoms, num_rotatable_bonds

**Geometric Descriptors:**
- Size: ligand_volume, ligand_surface_area, ligand_sasa
- Shape: shape_complementarity, spatial_distribution_x

**Pharmacophore Features:**
- Hydrophobic: num_hydrophobic_pharmacophore
- Ionizable: num_negative_ionizable, num_positive_ionizable
- H-bond: num_hbd_pharmacophore, num_hba_pharmacophore

**Efficiency Metrics:**
- ligand_efficiency (LE = affinity / heavy atoms)
- lipophilic_efficiency (LLE = pIC50 - LogP)

**Binding Geometry (in Set C only):**
- Hinge interactions (MET769, GLN767)
- DFG motif (PHE832)
- P-loop (GLY695, GLY697)
- C-helix (LYS721, GLU738)

---

## Usage for Machine Learning

### Loading Feature Sets

```python
import pandas as pd

# Load main dataset
data = pd.read_csv('full_modeling_dataset.csv')

# Extract target
y = data['LN_IC50']

# Load Set B (recommended)
with open('04_consensus/set_b_expanded_structural.txt') as f:
    set_b_features = [line.strip() for line in f]

# Extract structural features
X_structural = data[set_b_features]

# Extract transcriptomic features
with open('transcriptomic_features.txt') as f:
    trans_features = [line.strip() for line in f]
X_transcriptomic = data[trans_features]

# Ready for conditioning layer!
```

---

## Validation Metrics

### Structural Features Alone (SHAP Model)
- **R² (test):** 0.38
- **RMSE (test):** 1.45
- Explains 38% of IC50 variance without transcriptomics

### Feature Quality
- No zero-variance features
- All features have biological interpretation
- Appropriate size for sample count
- Validated by 3 independent methods

---

## Key Insights

### 1. Drug Properties Predict Potency
Most selected features are molecular properties (LogP, TPSA, charge), not binding geometry. This suggests:
- Features capture which drugs are potent
- Not necessarily HOW they bind
- Justifies conditioning layer to add cellular context

### 2. Binding Features Underrepresented
Few hinge/gatekeeper/DFG features selected because:
- Constant per drug across cell lines
- Don't explain IC50 variation alone
- May need transcriptomic context to matter

### 3. Strong Method Agreement
- 100% of Boruta features validated by MI
- All 3 methods agree on core features
- Consensus is scientifically rigorous

---

## Integration with Transcriptomics

### Combined Dataset Statistics

| Feature Type | Count | Source |
|-------------|-------|--------|
| Transcriptomic | 95 | Gene expression (biology-guided + data-driven) |
| Structural (Set A) | 12 | Consensus selection |
| Structural (Set B) | 32 | Expanded selection |
| Structural (Set C) | 40 | With binding features |

### Expected Performance
- Transcriptomics only: R² ~ 0.40-0.45
- Structure only: R² ~ 0.38
- Concatenation: R² ~ 0.50-0.55
- **Conditioning (goal):** R² ~ 0.55-0.65

---

## Next Steps

### Phase 6: Neural Conditioning Layer

**Test all three feature sets:**
1. Set A (12): Conservative baseline
2. Set B (32): **Primary evaluation**
3. Set C (40): Biological validation

**Ablation studies:**
- Compare against transcriptomics-only
- Compare against simple concatenation
- Test if conditioning adds value

---

**Created:** February 2026  
**Last Updated:** February 4, 2026  
**Project:** CASPID
