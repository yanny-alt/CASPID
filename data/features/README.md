# Extracted Structural Features

**Source:** Consensus docking poses (AutoDock Vina + SMINA)  
**Input:** 24 high-confidence poses (RMSD < 2.0 Å)  
**Output:** 92 non-redundant structural features

---

## Directory Structure

```
features/
├── 01_distance/
│   └── distance_features.csv              # 114 features → 60 after QC
├── 02_geometric/
│   └── geometric_features.csv             # 18 features
├── 03_physicochemical/
│   └── physicochemical_features.csv       # 26 features
├── 04_interactions/
│   └── interaction_features.csv           # 18 features
├── 05_pharmacophore/
│   └── pharmacophore_features.csv         # 14 features
└── 06_combined/
    ├── raw_combined_features.csv          # 193 raw features
    └── clean_docking_features.csv         # 92 QC-filtered features ⭐
```

---

## Feature Categories

### 1. Distance Features (60 final)
- Minimum/mean distances to critical kinase residues
- Contacts within 4.0 Å cutoff
- Global spatial metrics (centroid distance, radius of gyration)

**Key residues monitored:**
- **Hinge region:** H-bond donors/acceptors for ATP mimicry
- **Gatekeeper:** T790 (EGFR), T529 (BRAF) - resistance mutation site
- **DFG motif:** Activation state indicator
- **P-loop:** Phosphate-binding region
- **C-helix:** Regulatory element
- **Activation loop:** Catalytic competence

---

### 2. Geometric Features (18)
- Molecular volume and surface area
- Radius of gyration, asphericity, eccentricity
- Planarity index
- PCA variance ratios and principal axes
- Shape complementarity with protein
- Buried surface area upon binding

---

### 3. Physicochemical Features (26)
- Electrostatic energy (Coulombic, ε=4)
- van der Waals energy (Lennard-Jones 6-12)
- SASA (solvent accessible surface area)
- Atom type counts (polar vs nonpolar)
- Charge distribution
- Molecular descriptors (MW, LogP, HBD, HBA, TPSA)
- Ligand efficiency (LE = affinity / heavy atoms)
- Lipophilic efficiency (LLE)
- Binding affinities from docking

---

### 4. Interaction Features (18)
- H-bonds: count, directionality (ligand vs protein donor), distances
- Aromatic interactions: π-π stacking counts and distances
- Hydrophobic contacts: C-C interactions <4.5 Å
- Salt bridges: charge-charge pairs <4.0 Å
- Halogen atoms: F, Cl, Br counts
- Total interactions and interaction density

---

### 5. Pharmacophore Features (14)
- Functional group counts (HBD, HBA, aromatic, hydrophobic, ionizable)
- 3D spatial metrics (span, radius, triangular area)
- Spatial distribution along X, Y, Z axes
- Molecular compactness
- Atom density (atoms per unit volume)

---

## Quality Control Summary

**Initial features:** 193  
**Removed (>95% missing):** 46  
**Removed (zero variance):** 27  
**Removed (high correlation |r|>0.95):** 28  
**Final features:** 92  

**Success rate:** 47.7% retention after rigorous QC

---

## File Format

All CSV files contain:
- **Metadata columns:** `job_id`, `protein`, `ligand`
- **Feature columns:** Numeric values (some NaN for protein-specific features)
- **Rows:** 24 poses (15 BRAF, 14 EGFR; afatinib-EGFR missing from test run)

---

## Usage for Machine Learning

**Primary file:** `06_combined/clean_docking_features.csv`

```python
import pandas as pd

# Load clean features
features = pd.read_csv('06_combined/clean_docking_features.csv')

# Separate metadata
metadata = features[['job_id', 'protein', 'ligand']]
X_docking = features.drop(['job_id', 'protein', 'ligand'], axis=1)

# X_docking shape: (24, 92)
# Ready to merge with transcriptomic features
```

---

## Integration with IC50 Data

These 92 docking features will be merged with:
- **95 gene expression features** (selected via Boruta + MI + SHAP)
- **14,796 drug-cell line IC50 measurements**

**Note:** Each docking feature vector is constant across all cell lines for a given drug-protein pair. The neural conditioning layer learns to reweight these features based on cellular transcriptomic state.

---

## Validation

**Consensus docking validation:**
- Vina-SMINA RMSD < 2.0 Å: 82.8% (24/29 poses)
- Affinity correlation: r = 0.966
- Mean RMSD: 0.65 Å

**Feature quality:**
- No excessive missing values (all <50% after protein-specific removal)
- No zero-variance features remain
- No redundant highly correlated pairs (|r| < 0.95)

---

**Created:** February 2026  
**Last Updated:** February 4, 2026  
**Project:** CASPID
