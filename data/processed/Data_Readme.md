# CASPID Processed Data

This directory contains the final integrated dataset and supporting files for the CASPID project.

## Main Dataset

### `caspid_integrated_dataset.csv`
**The primary dataset for analysis and modeling.**

- **Records**: 14,796 compound-cell line combinations
- **Cell Lines**: 711 unique cancer cell lines
- **Compounds**: 15 EGFR/BRAF inhibitors with verified SMILES structures
- **Features**:
  - Drug response data (LN_IC50)
  - Chemical structure (SMILES)
  - 95 transcriptomic features (gene expression, Z-score normalized)
    - 45 biology-guided genes (EGFR/BRAF pathway)
    - 50 data-driven genes (correlation-based selection)

**Columns**:
- `DRUG_NAME`: Compound name
- `CELL_LINE_NAME`: GDSC cell line name
- `DepMap_ModelID`: DepMap cell line identifier
- `LN_IC50`: Natural log-transformed IC50 (drug response)
- `SMILES`: Canonical SMILES structure
- `ChEMBL_ID`: ChEMBL identifier
- `[GENE_NAME (EntrezID)]`: Z-score normalized gene expression (95 genes)

---

## Supporting Files

### `dockable_compounds.csv`
List of 37 EGFR/BRAF-targeted compounds with SMILES structures retrieved from ChEMBL/PubChem.

**Note**: Only **15 of these 37 compounds** were found in GDSC2 drug response data and included in the final integrated dataset. The remaining 22 compounds had no screening data available in GDSC.

**Compounds in Final Dataset (15)**:
- **EGFR inhibitors (9)**: AZD3759, Afatinib, Erlotinib, Gefitinib, Lapatinib, Osimertinib, Sapitinib, Foretinib, Cediranib
- **BRAF inhibitors (3)**: Dabrafenib, PLX-4720, SB590885
- **Multi-kinase inhibitors (3)**: Axitinib, Motesanib, Sorafenib

**Compounds NOT in GDSC (22)**: See `unmatched_compounds.csv` for the list of compounds that were excluded due to lack of drug response data.

### `cell_line_mapping.csv`
Mapping between GDSC cell line names and DepMap ModelIDs.

- **Total Matches**: 965 cell lines
- **Matching Methods**:
  - COSMIC_ID (most reliable)
  - Exact name matching
  - Fuzzy matching (score ≥85)

### `selected_genes.txt`
List of 95 genes used as transcriptomic features.

**Gene Selection Strategy**:
1. **Biology-guided (45 genes)**: Curated based on EGFR/BRAF pathway biology
   - EGFR pathway: EGFR, ERBB2, KRAS, BRAF, PIK3CA, AKT1-3, MTOR, PTEN
   - Apoptosis: BCL2, BAX, CASP3/8/9
   - Cell cycle: TP53, RB1, CCND1-3, CDK4/6
   - DNA damage: BRCA1/2, ATM, CHEK1/2
   - Drug resistance: ABCB1, ABCG2, MET, AXL

2. **Data-driven (50 genes)**: Selected by correlation with drug response
   - Pearson correlation with LN_IC50
   - Excluded if highly correlated (r > 0.8) with biology panel

### `gene_name_mapping.csv`
Reference table mapping simple gene names to full DepMap column names.

Format: `GENE_NAME (EntrezID)`
- Example: `EGFR (1956)`

### `integration_summary.csv`
Summary statistics of the integrated dataset.

### `matching_summary.csv`
Cell line matching quality control statistics.

---

## Data Quality

### Quality Control Applied

1. **Drug Response**:
   - 0% missing IC50 values
   - 0.7% outliers (>3 SD) retained for robustness

2. **Data Coverage**:
   - Removed cell lines with <3 compounds (2 cells removed)
   - Removed compounds with <10 cell lines (0 compounds removed)

3. **Gene Expression**:
   - Z-score normalized per gene across all cell lines
   - Only default DepMap entries used (IsDefaultEntryForModel = Yes)

### Data Provenance

- **Drug Response**: GDSC2 (Genomics of Drug Sensitivity in Cancer)
- **Gene Expression**: DepMap 24Q4 (Cancer Dependency Map)
- **Cell Line Mapping**: COSMIC ID + fuzzy matching
- **Chemical Structures**: ChEMBL API

---

## Usage

### Load Main Dataset

```python
import pandas as pd

# Load integrated dataset
data = pd.read_csv('caspid_integrated_dataset.csv')

# Separate features
drug_features = data[['DRUG_NAME', 'SMILES']]
response = data['LN_IC50']

# Gene expression features
gene_cols = [c for c in data.columns if '(' in c]  # Columns with (EntrezID)
gene_expression = data[gene_cols]
```

### Load Gene List

```python
# Load selected genes
with open('selected_genes.txt', 'r') as f:
    genes = [line.strip() for line in f if not line.startswith('#')]
```

---

## Citation

If you use this dataset, please cite:



---

## Data Version

- **Created**: 2026
- **Last Updated**: 2026
- **Version**: 1.0
- **Integration Script**: `04_prepare_final_dataset_FINAL.py`

---

