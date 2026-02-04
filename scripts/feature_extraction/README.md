# Feature Extraction Scripts

**Purpose:** Extract structural features from high-confidence consensus docking poses for machine learning.

**Status:** ✅ Complete - 92 non-redundant features extracted from 24 poses

---

## Execution Order

```bash
# Extract features sequentially
python 01_extract_distance_features.py      # ~2 seconds
python 02_extract_geometric_features.py     # ~2 seconds
python 03_extract_physicochemical_features.py  # ~3 seconds
python 04_extract_interaction_features.py   # ~3 seconds
python 05_extract_pharmacophore_features.py # ~2 seconds
python 06_combine_features.py               # ~1 second

# Total runtime: ~15 seconds for all scripts
```

---

## Scripts Overview

### 01_extract_distance_features.py
**Extracts:** 114 distance-based features (reduced to 60 after QC)
- Per-residue distances to key kinase regions (hinge, gatekeeper, DFG, P-loop, C-helix, activation loop)
- Global distance metrics (centroid distance, min/max distances, radius of gyration)
- Binding pocket penetration depth

**Output:** `DATA/FEATURES/01_distance/distance_features.csv`

---

### 02_extract_geometric_features.py
**Extracts:** 18 geometric features
- Shape complementarity (volume, surface area, max dimension)
- Spatial extent (radius of gyration, asphericity, eccentricity)
- Molecular planarity
- PCA-based orientation descriptors
- Buried surface area upon binding

**Output:** `DATA/FEATURES/02_geometric/geometric_features.csv`

---

### 03_extract_physicochemical_features.py
**Extracts:** 26 physicochemical features
- Electrostatic energy (Coulombic interactions)
- van der Waals energy (Lennard-Jones potential)
- SASA (solvent accessible surface area)
- Charge distribution (total, max positive/negative)
- Molecular descriptors (MW, LogP, HBD, HBA, TPSA, rotatable bonds)
- Ligand efficiency (LE) and lipophilic efficiency (LLE)
- Binding affinities (Vina, SMINA, consensus)

**Output:** `DATA/FEATURES/03_physicochemical/physicochemical_features.csv`

---

### 04_extract_interaction_features.py
**Extracts:** 18 interaction features
- H-bonds (count, directionality, distances)
- Aromatic interactions (π-π stacking counts and distances)
- Hydrophobic contacts (count, distances, surface area)
- Salt bridges (charge-charge interactions)
- Halogen atoms (F, Cl, Br counts)
- Total interaction counts and density

**Output:** `DATA/FEATURES/04_interactions/interaction_features.csv`

---

### 05_extract_pharmacophore_features.py
**Extracts:** 14 pharmacophore features
- Pharmacophore point counts (HBD, HBA, aromatic, hydrophobic, ionizable)
- 3D geometry (span, radius, triangular area)
- Spatial distribution (X, Y, Z variance, volume ratio)
- Molecular compactness and atom density

**Output:** `DATA/FEATURES/05_pharmacophore/pharmacophore_features.csv`

---

### 06_combine_features.py
**Combines and filters all features with quality control:**

**QC Steps:**
1. Merge all 5 feature sets (193 raw features)
2. Remove features with >95% missing values (46 removed)
3. Remove zero-variance features (27 removed)
4. Remove highly correlated pairs |r| > 0.95 (28 removed)
5. Checkpoint 3: Test Spearman correlation with IC50 values

**Final Output:** 92 non-redundant, high-quality features

**Files:**
- `DATA/FEATURES/06_combined/raw_combined_features.csv` (193 features)
- `DATA/FEATURES/06_combined/clean_docking_features.csv` (92 features) ← Use this for ML

---

## Feature Summary

| Feature Type | Raw | After QC |
|-------------|-----|----------|
| Distance | 114 | ~60 |
| Geometric | 18 | 18 |
| Physicochemical | 26 | 26 |
| Interaction | 18 | 18 |
| Pharmacophore | 14 | 14 |
| **Total** | **193** | **92** |

---

## Quality Control Results

**Missing values:** 46 features removed (>95% missing due to protein-specific residues)  
**Zero variance:** 27 features removed (no variation across poses)  
**High correlation:** 28 features removed (|r| > 0.95 with other features)  

**Checkpoint 3:** No individual features showed |Spearman ρ| > 0.3 with averaged IC50 values, confirming that structural features alone are insufficient for cell-specific predictions. This motivated the neural conditioning approach to integrate cellular context.

---

## Methods Text (for Manuscript)

> Structural features were extracted from 24 high-confidence consensus docking poses (RMSD < 2.0 Å between AutoDock Vina and SMINA). Feature extraction was parallelized across CPU cores and included: (1) distance-based features measuring proximity to critical kinase residues; (2) geometric features quantifying shape complementarity and molecular conformation; (3) physicochemical properties including electrostatic and van der Waals energies; (4) interaction fingerprints capturing hydrogen bonds, aromatic contacts, and hydrophobic interactions; and (5) pharmacophore descriptors encoding 3D spatial arrangements of functional groups. Quality control removed features with >95% missing values, zero variance, or high inter-correlation (|r| > 0.95), yielding 92 non-redundant features for machine learning.

---

## Dependencies

```bash
conda install -c conda-forge rdkit pandas numpy scipy scikit-learn
```

**Python version:** 3.9+  
**Tested on:** macOS (Apple Silicon)

---

**Created:** February 2026  
**Project:** CASPID 
