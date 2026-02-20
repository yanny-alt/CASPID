#!/usr/bin/env python3
"""
CASPID Data Exploration
Critical first step: Understand what we have before integration
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set paths - UPDATED FOR YOUR DIRECTORY
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_RAW = PROJECT_ROOT / "DATA"

print("="*70)
print("CASPID DATA EXPLORATION")
print("="*70)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_RAW}")

# Check directories exist
if not DATA_RAW.exists():
    print(f"\n❌ ERROR: Data directory not found at {DATA_RAW}")
    exit(1)

print(f"\n✓ Data directory found")
print(f"  Subdirectories: {[d.name for d in DATA_RAW.iterdir() if d.is_dir()]}")

# ============================================
# 1. EXPLORE GDSC FILES
# ============================================

print("\n" + "="*70)
print("1. GDSC DRUG RESPONSE DATA")
print("="*70)

# Find the GDSC file (flexible naming)
gdsc_dir = DATA_RAW / "GDSC"
gdsc_files = list(gdsc_dir.glob("*.xlsx"))

if not gdsc_files:
    print(f"❌ No .xlsx files found in {gdsc_dir}")
    print(f"   Files present: {list(gdsc_dir.iterdir())}")
    exit(1)

print(f"\nFound files in GDSC directory:")
for f in gdsc_dir.iterdir():
    print(f"  - {f.name}")

# Load drug response (first .xlsx file)
gdsc_response_file = [f for f in gdsc_files if 'drug' in f.name.lower() or 'response' in f.name.lower()][0]
print(f"\nLoading: {gdsc_response_file.name}")

gdsc_response = pd.read_excel(gdsc_response_file)

print(f"\nShape: {gdsc_response.shape}")
print(f"Columns: {list(gdsc_response.columns)}")
print(f"\nFirst 3 rows:")
print(gdsc_response.head(3))

print(f"\nUnique cell lines: {gdsc_response['CELL_LINE_NAME'].nunique()}")
print(f"Unique drugs: {gdsc_response['DRUG_NAME'].nunique()}")
print(f"Total data points: {len(gdsc_response)}")

# Check for missing IC50 values
missing_ic50 = gdsc_response['LN_IC50'].isna().sum()
print(f"\nMissing LN_IC50 values: {missing_ic50} ({missing_ic50/len(gdsc_response)*100:.1f}%)")

# ============================================
# 2. EXPLORE COMPOUND ANNOTATIONS
# ============================================

print("\n" + "="*70)
print("2. COMPOUND ANNOTATIONS")
print("="*70)

# Find compound file
compound_files = list(gdsc_dir.glob("*compound*.csv")) + list(gdsc_dir.glob("*annotation*.csv"))

if not compound_files:
    print(f"⚠️  No compound annotation file found in {gdsc_dir}")
    compounds = None
else:
    compound_file = compound_files[0]
    print(f"\nLoading: {compound_file.name}")
    compounds = pd.read_csv(compound_file)
    
    print(f"\nShape: {compounds.shape}")
    print(f"Columns: {list(compounds.columns)}")
    print(f"\nFirst 3 rows:")
    print(compounds.head(3))
    
    # Check for SMILES column
    if 'SMILES' in compounds.columns:
        print("\n✓ SMILES column found!")
        smiles_count = compounds['SMILES'].notna().sum()
        print(f"  Compounds with SMILES: {smiles_count}/{len(compounds)}")
    else:
        print("\n⚠️  WARNING: No SMILES column found!")
        print("   Available columns:", list(compounds.columns))
        print("   We'll need to fetch SMILES from ChEMBL/PubChem")
    
    # Check EGFR and BRAF targeting drugs
    # Find the correct column name for targets
    target_col = None
    for col in ['TARGET', 'PUTATIVE_TARGET', 'TARGETS', 'target']:
        if col in compounds.columns:
            target_col = col
            break
    
    if target_col:
        egfr_drugs = compounds[
            compounds[target_col].str.contains('EGFR', na=False, case=False)
        ]
        print(f"\nEGFR-targeting drugs: {len(egfr_drugs)}")
        print("Examples:")
        print(egfr_drugs[['DRUG_NAME', target_col]].head())
        
        braf_drugs = compounds[
            compounds[target_col].str.contains('BRAF', na=False, case=False)
        ]
        print(f"\nBRAF-targeting drugs: {len(braf_drugs)}")
        print("Examples:")
        print(braf_drugs[['DRUG_NAME', target_col]].head())
        
        # Check MEK drugs
        mek_drugs = compounds[
            compounds[target_col].str.contains('MEK', na=False, case=False)
        ]
        print(f"\nMEK-targeting drugs: {len(mek_drugs)}")
        print("Examples:")
        print(mek_drugs[['DRUG_NAME', target_col]].head())
    else:
        print(f"\n⚠️  Could not find target column")

# ============================================
# 3. EXPLORE DEPMAP MODEL FILE
# ============================================

print("\n" + "="*70)
print("3. DEPMAP MODEL METADATA")
print("="*70)

depmap_dir = DATA_RAW / "DEPMAP"
model_file = depmap_dir / "Model.csv"

if not model_file.exists():
    print(f"❌ Model.csv not found in {depmap_dir}")
    print(f"   Files present: {list(depmap_dir.iterdir())}")
    exit(1)

print(f"\nLoading: {model_file.name}")
depmap_model = pd.read_csv(model_file)

print(f"\nShape: {depmap_model.shape}")
print(f"Columns (first 10): {list(depmap_model.columns[:10])}")
print(f"\nKey columns for matching:")
print(f"  - ModelID: {depmap_model['ModelID'].nunique()} unique")
print(f"  - StrippedCellLineName: {depmap_model['StrippedCellLineName'].nunique()} unique")
print(f"  - CellLineName: {depmap_model['CellLineName'].nunique()} unique")

print(f"\nFirst 3 rows (key columns):")
print(depmap_model[['ModelID', 'CellLineName', 'StrippedCellLineName']].head(3))

# ============================================
# 4. EXPLORE OMICS PROFILES
# ============================================

print("\n" + "="*70)
print("4. DEPMAP OMICS PROFILES")
print("="*70)

profiles_file = depmap_dir / "OmicsProfiles.csv"

if not profiles_file.exists():
    print(f"⚠️  OmicsProfiles.csv not found in {depmap_dir}")
else:
    print(f"\nLoading: {profiles_file.name}")
    omics_profiles = pd.read_csv(profiles_file)
    
    print(f"\nShape: {omics_profiles.shape}")
    print(f"Columns: {list(omics_profiles.columns)}")
    
    # Filter for RNA-seq profiles
    rna_profiles = omics_profiles[omics_profiles['DataType'] == 'rna']
    print(f"\nRNA-seq profiles: {len(rna_profiles)}")
    print(f"  Default entries: {rna_profiles['IsDefaultEntryForModel'].sum()}")
    
    print(f"\nFirst 3 RNA profiles:")
    print(rna_profiles[['ModelID', 'ProfileID', 'StrippedCellLineName']].head(3))

# ============================================
# 5. PREVIEW EXPRESSION DATA (LARGE FILE)
# ============================================

print("\n" + "="*70)
print("5. GENE EXPRESSION DATA (Large File)")
print("="*70)

# Find expression file (flexible naming)
expression_files = list(depmap_dir.glob("*Expression*.csv")) + list(depmap_dir.glob("*expression*.csv"))

if not expression_files:
    print(f"⚠️  No expression file found in {depmap_dir}")
else:
    expression_file = expression_files[0]
    print(f"\nFound: {expression_file.name}")
    print(f"File size: {expression_file.stat().st_size / (1024**2):.1f} MB")
    
    # Read just first few rows to check structure
    print("\nReading first 5 rows (this may take a moment)...")
    expression_preview = pd.read_csv(expression_file, nrows=5)
    
    print(f"\nShape (preview): {expression_preview.shape}")
    print(f"Number of genes: {len(expression_preview.columns) - 3}")  # Minus ProfileID, IsDefault, ModelID
    
    print(f"\nFirst 5 columns:")
    print(list(expression_preview.columns[:5]))
    
    print(f"\nFirst 2 rows (first 5 columns):")
    print(expression_preview.iloc[:2, :5])

# ============================================
# 6. CELL LINE MATCHING PREVIEW
# ============================================

print("\n" + "="*70)
print("6. CELL LINE MATCHING PREVIEW")
print("="*70)

# Get unique cell line names from GDSC
gdsc_cells = set(gdsc_response['CELL_LINE_NAME'].unique())
print(f"\nUnique GDSC cell lines: {len(gdsc_cells)}")
print(f"Examples: {list(gdsc_cells)[:5]}")

# Get unique cell line names from DepMap
depmap_cells = set(depmap_model['StrippedCellLineName'].unique())
print(f"\nUnique DepMap cell lines: {len(depmap_cells)}")
print(f"Examples: {list(depmap_cells)[:5]}")

# Simple exact matching test
exact_matches = gdsc_cells & depmap_cells
print(f"\nExact matches (intersection): {len(exact_matches)}")
print(f"Match rate: {len(exact_matches)/len(gdsc_cells)*100:.1f}%")

if len(exact_matches) < 100:
    print("\n⚠️  WARNING: Low exact match rate!")
    print("   Will need fuzzy matching or name standardization")
    
    # Show some examples of non-matching names
    print("\n   GDSC names (first 10 not in DepMap):")
    non_matching = list(gdsc_cells - depmap_cells)[:10]
    for name in non_matching:
        print(f"     - {name}")
        
        # Try to find similar DepMap names
        from difflib import get_close_matches
        matches = get_close_matches(name, depmap_cells, n=1, cutoff=0.6)
        if matches:
            print(f"       (possibly matches: {matches[0]})")

# ============================================
# 7. DATA QUALITY SUMMARY
# ============================================

print("\n" + "="*70)
print("7. DATA QUALITY SUMMARY")
print("="*70)

issues = []

# Check 1: SMILES availability
if compounds is not None:
    if 'SMILES' not in compounds.columns:
        issues.append("❌ SMILES column missing - need to fetch from ChEMBL")
    else:
        smiles_available = compounds['SMILES'].notna().sum() / len(compounds)
        if smiles_available < 0.8:
            issues.append(f"⚠️  Only {smiles_available*100:.0f}% compounds have SMILES")

# Check 2: Cell line matching
if len(exact_matches) < 600:
    issues.append(f"⚠️  Only {len(exact_matches)} exact cell line matches (need ≥600)")

# Check 3: EGFR/BRAF/MEK drugs
if compounds is not None and target_col:
    if len(egfr_drugs) < 10:
        issues.append(f"⚠️  Only {len(egfr_drugs)} EGFR drugs found (expected 15-25)")
    if len(braf_drugs) < 5:
        issues.append(f"⚠️  Only {len(braf_drugs)} BRAF drugs found (expected 10-20)")
    if len(mek_drugs) < 5:  # ADD THIS LINE
        issues.append(f"⚠️  Only {len(mek_drugs)} MEK drugs found (expected 5-10)")

# Check 4: Missing data
if missing_ic50 > len(gdsc_response) * 0.1:
    issues.append(f"⚠️  High missing IC50 rate: {missing_ic50/len(gdsc_response)*100:.1f}%")

if len(issues) == 0:
    print("\n✅ No major data quality issues detected!")
else:
    print("\n⚠️  Issues detected:")
    for issue in issues:
        print(f"   {issue}")

# ============================================
# 8. NEXT STEPS
# ============================================

print("\n" + "="*70)
print("8. RECOMMENDED NEXT STEPS")
print("="*70)

print("\n1. Get compound SMILES:")
print("   Run: python scripts/data_processing/02_get_compound_smiles.py")

print("\n2. Cell line matching:")
print("   Run: python scripts/data_processing/03_integrate_gdsc_depmap.py")

print("\n3. Final integration:")
print("   Run: python scripts/data_processing/04_prepare_final_dataset.py")

print("\n" + "="*70)
print("EXPLORATION COMPLETE")
print("="*70)
