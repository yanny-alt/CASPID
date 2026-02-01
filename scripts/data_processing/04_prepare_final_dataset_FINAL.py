#!/usr/bin/env python3
"""
FINAL INTEGRATION: Create complete CASPID dataset
Combines: Drug response + Cell line matching + Transcriptomics + Compounds

PRODUCTION VERSION - For publication
- 15 verified EGFR/BRAF inhibitors
- 711 cell lines
- 95 transcriptomic features (45 biology-guided + 50 data-driven)

Author: CASPID Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_RAW = PROJECT_ROOT / "DATA"
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"

print("="*70)
print("CASPID FINAL DATASET INTEGRATION")
print("="*70)
print("\nProduction version - 15 verified compounds")

# ============================================
# 1. LOAD ALL COMPONENTS
# ============================================

print("\n" + "="*70)
print("1. LOADING DATA COMPONENTS")
print("="*70)

# Drug response
gdsc_response = pd.read_excel(DATA_RAW / "GDSC" / "GDSC2_drug_response.xlsx")
print(f"✓ GDSC responses: {len(gdsc_response):,} records")

# Dockable compounds
dockable = pd.read_csv(DATA_PROCESSED / "dockable_compounds.csv")
print(f"✓ Dockable compounds: {len(dockable)} (with SMILES)")

# Cell line mapping
cell_mapping = pd.read_csv(DATA_PROCESSED / "cell_line_mapping.csv")
print(f"✓ Matched cell lines: {len(cell_mapping):,}")

# Gene expression
print("\n  Loading gene expression (large file)...")
expression = pd.read_csv(
    DATA_RAW / "DEPMAP" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)
print(f"✓ Gene expression: {expression.shape[0]:,} profiles × {expression.shape[1]-4:,} genes")

# ============================================
# 2. FILTER FOR VERIFIED COMPOUNDS
# ============================================

print("\n" + "="*70)
print("2. FILTERING FOR VERIFIED COMPOUNDS")
print("="*70)

# Use only compounds with exact name matches in GDSC
# These are the 15 verified compounds with complete data
dockable_drug_names = set(dockable['DRUG_NAME'])

filtered_response = gdsc_response[
    gdsc_response['DRUG_NAME'].isin(dockable_drug_names)
].copy()

print(f"\nFiltered from {len(gdsc_response):,} to {len(filtered_response):,} records")
print(f"Compounds: {filtered_response['DRUG_NAME'].nunique()}")
print(f"Cell lines: {filtered_response['CELL_LINE_NAME'].nunique()}")

# Show compound breakdown
print("\nCompound breakdown:")
for drug in sorted(filtered_response['DRUG_NAME'].unique()):
    count = len(filtered_response[filtered_response['DRUG_NAME'] == drug])
    target = dockable[dockable['DRUG_NAME'] == drug]['TARGET'].values[0]
    print(f"  • {drug}: {count} measurements ({target})")

# ============================================
# 3. INTEGRATE WITH CELL LINE MAPPING
# ============================================

print("\n" + "="*70)
print("3. INTEGRATING CELL LINE MAPPING")
print("="*70)

# Merge with cell mapping
integrated = filtered_response.merge(
    cell_mapping[['GDSC_CELL_LINE', 'DepMap_ModelID', 'Match_Method', 'Confidence']],
    left_on='CELL_LINE_NAME',
    right_on='GDSC_CELL_LINE',
    how='inner'
)

print(f"\nAfter cell line matching: {len(integrated):,} records")
print(f"Unique cell lines: {integrated['DepMap_ModelID'].nunique()}")
print(f"Unique compounds: {integrated['DRUG_NAME'].nunique()}")

# Add compound info (SMILES and ChEMBL ID)
integrated = integrated.merge(
    dockable[['DRUG_NAME', 'SMILES', 'ChEMBL_ID']],
    on='DRUG_NAME',
    how='left'
)

# ============================================
# 4. SELECT TRANSCRIPTOMIC FEATURES
# ============================================

print("\n" + "="*70)
print("4. SELECTING TRANSCRIPTOMIC FEATURES")
print("="*70)

# Biology-guided gene panel
# Curated based on EGFR/BRAF pathway biology
biology_genes = [
    # EGFR pathway
    'EGFR', 'ERBB2', 'ERBB3', 'KRAS', 'BRAF', 'PIK3CA', 
    'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PTEN',
    
    # Apoptosis
    'BCL2', 'BCL2L1', 'BAX', 'BAK1', 'BID', 'CASP3', 'CASP8', 'CASP9',
    
    # Cell cycle
    'CCND1', 'CCND2', 'CCND3', 'CDK4', 'CDK6', 'RB1', 'TP53', 
    'CDKN1A', 'CDKN1B', 'CDKN2A',
    
    # Drug resistance
    'ABCB1', 'ABCG2', 'MET', 'AXL',
    
    # DNA damage
    'ATM', 'ATR', 'CHEK1', 'CHEK2', 'BRCA1', 'BRCA2',
    
    # RTK signaling
    'SRC', 'JAK1', 'JAK2', 'STAT3', 'STAT5A', 'STAT5B'
]

print(f"\nBiology-guided panel: {len(biology_genes)} genes")

# Gene names in DepMap expression file have format: "GENE (EntrezID)"
# Create mapping function
def find_gene_column(gene_name, all_columns):
    """Find column name that starts with gene_name followed by ' ('"""
    for col in all_columns:
        if col.startswith(f"{gene_name} ("):
            return col
    return None

# Get all expression columns
expression_cols = expression.columns.tolist()

# Find matching columns for biology genes
available_biology_genes = []
gene_mapping = {}  # maps simple name to full column name

for gene in biology_genes:
    full_col = find_gene_column(gene, expression_cols)
    if full_col:
        available_biology_genes.append(full_col)
        gene_mapping[gene] = full_col

print(f"  Found in expression data: {len(available_biology_genes)}/{len(biology_genes)}")

if len(gene_mapping) < len(biology_genes):
    missing = set(biology_genes) - set(gene_mapping.keys())
    print(f"  Missing: {', '.join(sorted(missing))}")

print(f"\n  Example matches:")
for simple, full in list(gene_mapping.items())[:5]:
    print(f"    {simple} → {full}")

# Prepare expression data
# Filter for default entries only
expression_default = expression[expression['IsDefaultEntryForModel'] == 'Yes'].copy()
expression_default = expression_default.set_index('ModelID')

# Get gene columns only
gene_cols = [c for c in expression_default.columns if c not in 
             ['Unnamed: 0', 'SequencingID', 'IsDefaultEntryForModel', 
              'ModelConditionID', 'IsDefaultEntryForMC']]

expression_matrix = expression_default[gene_cols]
print(f"\nExpression matrix: {expression_matrix.shape}")

# ============================================
# 5. DATA-DRIVEN GENE SELECTION
# ============================================

print("\n" + "="*70)
print("5. DATA-DRIVEN GENE SELECTION")
print("="*70)

# Get cell lines that have both expression and drug data
model_ids_with_data = set(integrated['DepMap_ModelID'])
model_ids_with_expression = set(expression_matrix.index)
common_models = model_ids_with_data & model_ids_with_expression

print(f"Cell lines with both expression and drug data: {len(common_models)}")

# Sample for computational efficiency
sample_size = min(500, len(common_models))
sample_models = list(common_models)[:sample_size]

# Get average drug response per cell line
avg_response = integrated.groupby('DepMap_ModelID')['LN_IC50'].mean()

# Calculate correlations
print(f"\nCalculating gene-response correlations for {len(gene_cols):,} genes...")
print("(This may take a few minutes...)")

correlations = []

for gene in gene_cols:
    if gene in available_biology_genes:
        continue  # Already selected
    
    # Get expression values
    expr_values = expression_matrix.loc[sample_models, gene]
    resp_values = avg_response.loc[sample_models]
    
    # Calculate correlation
    if expr_values.notna().sum() > 10:
        try:
            corr, pval = stats.pearsonr(
                expr_values.dropna().values,
                resp_values.loc[expr_values.dropna().index].values
            )
            
            correlations.append({
                'Gene': gene,
                'Correlation': corr,
                'AbsCorr': abs(corr),
                'PValue': pval
            })
        except:
            pass  # Skip genes with constant values

corr_df = pd.DataFrame(correlations).sort_values('AbsCorr', ascending=False)
print(f"Calculated correlations for {len(corr_df):,} genes")

# Select top 50 data-driven genes
# Exclude genes highly correlated with biology panel (r > 0.8)
print("\nSelecting top correlated genes...")

top_correlated = []
for _, row in corr_df.iterrows():
    if len(top_correlated) >= 50:
        break
    
    gene = row['Gene']
    
    # Check correlation with biology genes
    highly_correlated = False
    for bio_gene in available_biology_genes:
        if bio_gene in expression_matrix.columns and gene in expression_matrix.columns:
            try:
                gene_corr = expression_matrix[[bio_gene, gene]].corr().iloc[0, 1]
                if abs(gene_corr) > 0.8:
                    highly_correlated = True
                    break
            except:
                pass
    
    if not highly_correlated:
        top_correlated.append(gene)

print(f"Data-driven genes selected: {len(top_correlated)}")

# Combine gene panels
final_genes = available_biology_genes + top_correlated
print(f"\n✓ TOTAL GENES SELECTED: {len(final_genes)}")
print(f"  - Biology-guided: {len(available_biology_genes)}")
print(f"  - Data-driven: {len(top_correlated)}")

# ============================================
# 6. CREATE FINAL DATASET
# ============================================

print("\n" + "="*70)
print("6. CREATING FINAL INTEGRATED DATASET")
print("="*70)

# Extract expression for selected genes
selected_expression = expression_matrix[final_genes].copy()

# Z-score normalization (per gene across all cell lines)
print("\nApplying Z-score normalization to gene expression...")
selected_expression_z = selected_expression.apply(
    lambda x: (x - x.mean()) / x.std(), axis=0
)

# Merge expression with drug response
final_dataset = integrated.merge(
    selected_expression_z,
    left_on='DepMap_ModelID',
    right_index=True,
    how='inner'
)

print(f"\n✓ Final dataset created")
print(f"  Rows (compound-cell combinations): {len(final_dataset):,}")
print(f"  Unique cell lines: {final_dataset['DepMap_ModelID'].nunique()}")
print(f"  Unique compounds: {final_dataset['DRUG_NAME'].nunique()}")
print(f"  Features: Drug response + SMILES + {len(final_genes)} gene expression")

# ============================================
# 7. QUALITY CONTROL
# ============================================

print("\n" + "="*70)
print("7. QUALITY CONTROL")
print("="*70)

# Check for missing values
missing_ic50 = final_dataset['LN_IC50'].isna().sum()
print(f"\nMissing IC50 values: {missing_ic50} ({missing_ic50/len(final_dataset)*100:.1f}%)")

# Check for outliers
ic50_mean = final_dataset['LN_IC50'].mean()
ic50_std = final_dataset['LN_IC50'].std()
outliers = final_dataset[abs(final_dataset['LN_IC50'] - ic50_mean) > 3 * ic50_std]
print(f"IC50 outliers (>3 SD): {len(outliers)} ({len(outliers)/len(final_dataset)*100:.1f}%)")

# Check data coverage
compounds_per_cell = final_dataset.groupby('DepMap_ModelID')['DRUG_NAME'].nunique()
cells_per_compound = final_dataset.groupby('DRUG_NAME')['DepMap_ModelID'].nunique()

low_compound_cells = (compounds_per_cell < 3).sum()
low_cell_compounds = (cells_per_compound < 10).sum()

print(f"\nData coverage:")
print(f"  Cell lines with <3 compounds: {low_compound_cells}")
print(f"  Compounds with <10 cell lines: {low_cell_compounds}")

# Apply QC filters
print("\nApplying QC filters...")
original_size = len(final_dataset)

# Remove cells with <3 compounds
cells_to_keep = compounds_per_cell[compounds_per_cell >= 3].index
final_dataset = final_dataset[final_dataset['DepMap_ModelID'].isin(cells_to_keep)]

# Remove compounds with <10 cells
compounds_to_keep = cells_per_compound[cells_per_compound >= 10].index
final_dataset = final_dataset[final_dataset['DRUG_NAME'].isin(compounds_to_keep)]

print(f"  Removed {original_size - len(final_dataset):,} records")
print(f"  Final size: {len(final_dataset):,} records")

# ============================================
# 8. SAVE RESULTS
# ============================================

print("\n" + "="*70)
print("8. SAVING FINAL DATASET")
print("="*70)

# Save main dataset
output_file = DATA_PROCESSED / "caspid_integrated_dataset.csv"
final_dataset.to_csv(output_file, index=False)
print(f"\n✓ Main dataset: {output_file}")

# Save gene list
gene_list_file = DATA_PROCESSED / "selected_genes.txt"
with open(gene_list_file, 'w') as f:
    f.write("# Full gene names (with Entrez IDs)\n")
    f.write('\n'.join(final_genes))
    f.write("\n\n# Simple gene names\n")
    simple_names = [g.split(' (')[0] for g in final_genes]
    f.write('\n'.join(simple_names))
print(f"✓ Selected genes: {gene_list_file}")

# Save gene mapping
gene_map_file = DATA_PROCESSED / "gene_name_mapping.csv"
mapping_df = pd.DataFrame([
    {
        'Simple_Name': g.split(' (')[0], 
        'Full_Name': g, 
        'Entrez_ID': g.split('(')[1].rstrip(')')
    }
    for g in final_genes
])
mapping_df.to_csv(gene_map_file, index=False)
print(f"✓ Gene mapping: {gene_map_file}")

# Save summary statistics
summary = {
    'Total_Records': len(final_dataset),
    'Unique_Cells': final_dataset['DepMap_ModelID'].nunique(),
    'Unique_Compounds': final_dataset['DRUG_NAME'].nunique(),
    'Total_Genes': len(final_genes),
    'Biology_Genes': len(available_biology_genes),
    'DataDriven_Genes': len(top_correlated),
    'EGFR_Records': len(final_dataset[final_dataset['DRUG_NAME'].isin(
        dockable[dockable['TARGET'].str.contains('EGFR', case=False, na=False)]['DRUG_NAME']
    )]),
    'BRAF_Records': len(final_dataset[final_dataset['DRUG_NAME'].isin(
        dockable[dockable['TARGET'].str.contains('BRAF', case=False, na=False)]['DRUG_NAME']
    )])
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(DATA_PROCESSED / "integration_summary.csv", index=False)
print(f"✓ Summary: {DATA_PROCESSED / 'integration_summary.csv'}")

# ============================================
# 9. FINAL SUMMARY
# ============================================

print("\n" + "="*70)
print("INTEGRATION COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\n📊 DATASET STATISTICS:")
print(f"  ✓ Total data points: {len(final_dataset):,}")
print(f"  ✓ Cell lines: {final_dataset['DepMap_ModelID'].nunique()}")
print(f"  ✓ Compounds: {final_dataset['DRUG_NAME'].nunique()}")

# Count by target
egfr_drugs = dockable[dockable['TARGET'].str.contains('EGFR', case=False, na=False)]['DRUG_NAME']
braf_drugs = dockable[dockable['TARGET'].str.contains('BRAF', case=False, na=False)]['DRUG_NAME']

print(f"    - EGFR inhibitors: {final_dataset['DRUG_NAME'].isin(egfr_drugs).sum():,} records")
print(f"    - BRAF inhibitors: {final_dataset['DRUG_NAME'].isin(braf_drugs).sum():,} records")
print(f"  ✓ Transcriptomic features: {len(final_genes)}")
print(f"    - Biology-guided: {len(available_biology_genes)}")
print(f"    - Data-driven: {len(top_correlated)}")

print(f"\n✓ Key biology genes included:")
for simple in list(gene_mapping.keys())[:10]:
    print(f"  • {simple}")
if len(gene_mapping) > 10:
    print(f"  ... and {len(gene_mapping)-10} more")

print(f"\n✅ READY FOR MOLECULAR DOCKING!")
print(f"\nNext steps:")
print(f"  1. Protein preparation (ChimeraX)")
print(f"  2. Ligand preparation (Meeko)")
print(f"  3. Consensus docking (Vina + SMINA + LeDock)")
print(f"  4. Feature extraction and ML modeling")

print("\n" + "="*70)
