#!/usr/bin/env python3
"""
CRITICAL: Match GDSC cell lines with DepMap cell lines
Goal: ≥600 matched cell lines for robust analysis

Strategy:
1. Exact matching on stripped names
2. Fuzzy matching with multiple strategies
3. Manual verification of ambiguous matches
4. Quality control checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz, process
import re

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_RAW = PROJECT_ROOT / "DATA"
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"

print("="*70)
print("CRITICAL: GDSC ↔ DepMap CELL LINE MATCHING")
print("="*70)
print("\nGoal: Match ≥600 cell lines between datasets")

# ============================================
# LOAD DATA
# ============================================

print("\n" + "="*70)
print("1. LOADING DATA")
print("="*70)

# Load GDSC drug response
gdsc_response = pd.read_excel(DATA_RAW / "GDSC" / "GDSC2_drug_response.xlsx")
print(f"✓ GDSC responses loaded: {len(gdsc_response)} data points")

# Load DepMap model metadata
depmap_model = pd.read_csv(DATA_RAW / "DEPMAP" / "Model.csv")
print(f"✓ DepMap models loaded: {len(depmap_model)} cell lines")

# Get unique cell lines
gdsc_cells = gdsc_response[['CELL_LINE_NAME', 'COSMIC_ID']].drop_duplicates()
print(f"\n✓ Unique GDSC cell lines: {len(gdsc_cells)}")

depmap_cells = depmap_model[['ModelID', 'CellLineName', 'StrippedCellLineName', 'COSMICID']].copy()
print(f"✓ Unique DepMap cell lines: {len(depmap_cells)}")

# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_cell_line_name(name):
    """
    Standardize cell line name for matching
    """
    if pd.isna(name):
        return ""
    
    # Convert to string and uppercase
    name = str(name).upper().strip()
    
    # Remove common prefixes/suffixes
    name = re.sub(r'^CELL LINE[:\s]*', '', name)
    name = re.sub(r'\s*CELL.*$', '', name)
    
    # Remove special characters but keep hyphens
    name = re.sub(r'[^A-Z0-9\-]', '', name)
    
    # Remove tissue suffixes (e.g., "_LUNG", "_BREAST")
    name = re.sub(r'_[A-Z]+$', '', name)
    
    return name

def match_by_cosmic_id(gdsc_df, depmap_df):
    """
    Match using COSMIC ID (most reliable)
    """
    matches = []
    
    for idx, gdsc_row in gdsc_df.iterrows():
        cosmic_id = gdsc_row['COSMIC_ID']
        
        if pd.notna(cosmic_id):
            # Find in DepMap
            depmap_match = depmap_df[depmap_df['COSMICID'] == cosmic_id]
            
            if len(depmap_match) == 1:
                matches.append({
                    'GDSC_CELL_LINE': gdsc_row['CELL_LINE_NAME'],
                    'DepMap_ModelID': depmap_match.iloc[0]['ModelID'],
                    'DepMap_CellLineName': depmap_match.iloc[0]['CellLineName'],
                    'DepMap_StrippedName': depmap_match.iloc[0]['StrippedCellLineName'],
                    'COSMIC_ID': cosmic_id,
                    'Match_Method': 'COSMIC_ID',
                    'Confidence': 'HIGH'
                })
    
    return pd.DataFrame(matches)

def match_by_exact_name(gdsc_df, depmap_df, already_matched):
    """
    Exact matching on cleaned names
    """
    matches = []
    
    # Get unmatched GDSC cells
    matched_gdsc = set(already_matched['GDSC_CELL_LINE'])
    unmatched_gdsc = gdsc_df[~gdsc_df['CELL_LINE_NAME'].isin(matched_gdsc)]
    
    # Clean names
    gdsc_cleaned = {row['CELL_LINE_NAME']: clean_cell_line_name(row['CELL_LINE_NAME']) 
                    for _, row in unmatched_gdsc.iterrows()}
    
    depmap_cleaned = {row['ModelID']: {
        'cleaned': clean_cell_line_name(row['StrippedCellLineName']),
        'original': row['CellLineName'],
        'stripped': row['StrippedCellLineName']
    } for _, row in depmap_df.iterrows()}
    
    # Match
    for gdsc_name, gdsc_clean in gdsc_cleaned.items():
        for model_id, depmap_info in depmap_cleaned.items():
            if gdsc_clean == depmap_info['cleaned'] and gdsc_clean != "":
                matches.append({
                    'GDSC_CELL_LINE': gdsc_name,
                    'DepMap_ModelID': model_id,
                    'DepMap_CellLineName': depmap_info['original'],
                    'DepMap_StrippedName': depmap_info['stripped'],
                    'COSMIC_ID': None,
                    'Match_Method': 'EXACT_NAME',
                    'Confidence': 'HIGH'
                })
                break
    
    return pd.DataFrame(matches)

def match_by_fuzzy(gdsc_df, depmap_df, already_matched, threshold=85):
    """
    Fuzzy matching with conservative threshold
    """
    matches = []
    
    # Get unmatched cells
    matched_gdsc = set(already_matched['GDSC_CELL_LINE'])
    unmatched_gdsc = gdsc_df[~gdsc_df['CELL_LINE_NAME'].isin(matched_gdsc)]
    
    # Create DepMap lookup (cleaned names)
    depmap_lookup = {}
    for _, row in depmap_df.iterrows():
        clean_name = clean_cell_line_name(row['StrippedCellLineName'])
        if clean_name:
            depmap_lookup[clean_name] = {
                'ModelID': row['ModelID'],
                'CellLineName': row['CellLineName'],
                'StrippedName': row['StrippedCellLineName']
            }
    
    depmap_names = list(depmap_lookup.keys())
    
    # Fuzzy match each GDSC cell
    for _, gdsc_row in unmatched_gdsc.iterrows():
        gdsc_name = gdsc_row['CELL_LINE_NAME']
        gdsc_clean = clean_cell_line_name(gdsc_name)
        
        if not gdsc_clean:
            continue
        
        # Find best match
        best_match = process.extractOne(gdsc_clean, depmap_names, scorer=fuzz.ratio)
        
        if best_match and best_match[1] >= threshold:
            depmap_name = best_match[0]
            score = best_match[1]
            
            depmap_info = depmap_lookup[depmap_name]
            
            # Determine confidence
            if score >= 95:
                confidence = 'HIGH'
            elif score >= 85:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            matches.append({
                'GDSC_CELL_LINE': gdsc_name,
                'DepMap_ModelID': depmap_info['ModelID'],
                'DepMap_CellLineName': depmap_info['CellLineName'],
                'DepMap_StrippedName': depmap_info['StrippedName'],
                'COSMIC_ID': None,
                'Match_Method': f'FUZZY_{score}',
                'Confidence': confidence,
                'Fuzzy_Score': score
            })
    
    return pd.DataFrame(matches)

# ============================================
# MATCHING PIPELINE
# ============================================

print("\n" + "="*70)
print("2. MATCHING PIPELINE")
print("="*70)

all_matches = []

# Step 1: COSMIC ID matching
print("\nStep 1: Matching by COSMIC ID...")
cosmic_matches = match_by_cosmic_id(gdsc_cells, depmap_cells)
print(f"  ✓ COSMIC ID matches: {len(cosmic_matches)}")
all_matches.append(cosmic_matches)

# Step 2: Exact name matching
print("\nStep 2: Matching by exact name...")
exact_matches = match_by_exact_name(gdsc_cells, depmap_cells, cosmic_matches)
print(f"  ✓ Exact name matches: {len(exact_matches)}")
all_matches.append(exact_matches)

# Step 3: Fuzzy matching (conservative)
print("\nStep 3: Fuzzy matching (threshold ≥85)...")
combined_matches = pd.concat([cosmic_matches, exact_matches], ignore_index=True)
fuzzy_matches = match_by_fuzzy(gdsc_cells, depmap_cells, combined_matches, threshold=85)
print(f"  ✓ Fuzzy matches: {len(fuzzy_matches)}")
all_matches.append(fuzzy_matches)

# Combine all matches
final_matches = pd.concat(all_matches, ignore_index=True)

# Remove duplicates (keep highest confidence)
final_matches = final_matches.sort_values('Confidence').drop_duplicates('GDSC_CELL_LINE', keep='first')

print("\n" + "="*70)
print("3. MATCHING SUMMARY")
print("="*70)

total_matched = len(final_matches)
print(f"\n✓ TOTAL MATCHED: {total_matched} cell lines")
print(f"  Match rate: {total_matched/len(gdsc_cells)*100:.1f}%")

# Breakdown by method
print("\nBy matching method:")
for method in final_matches['Match_Method'].unique():
    count = (final_matches['Match_Method'] == method).sum()
    print(f"  - {method}: {count}")

# Breakdown by confidence
print("\nBy confidence:")
for conf in ['HIGH', 'MEDIUM', 'LOW']:
    count = (final_matches['Confidence'] == conf).sum()
    print(f"  - {conf}: {count}")

# ============================================
# QUALITY CONTROL
# ============================================

print("\n" + "="*70)
print("4. QUALITY CONTROL")
print("="*70)

# Check for duplicate DepMap matches
duplicate_depmap = final_matches['DepMap_ModelID'].duplicated().sum()
if duplicate_depmap > 0:
    print(f"\n⚠️  WARNING: {duplicate_depmap} duplicate DepMap IDs (same cell matched multiple times)")
    print("  These will be reviewed and removed")
    
    # Remove duplicates (keep first occurrence)
    final_matches = final_matches.drop_duplicates('DepMap_ModelID', keep='first')
    print(f"  ✓ After deduplication: {len(final_matches)} matches")

# Show sample of fuzzy matches for verification
print("\n" + "="*70)
print("5. SAMPLE FUZZY MATCHES (for verification)")
print("="*70)

fuzzy_sample = final_matches[final_matches['Match_Method'].str.contains('FUZZY')].head(10)
if len(fuzzy_sample) > 0:
    print("\nTop 10 fuzzy matches:")
    for _, row in fuzzy_sample.iterrows():
        print(f"\n  GDSC: {row['GDSC_CELL_LINE']}")
        print(f"  DepMap: {row['DepMap_StrippedName']}")
        print(f"  Score: {row.get('Fuzzy_Score', 'N/A')} | Confidence: {row['Confidence']}")

# ============================================
# CHECKPOINT: DID WE REACH GOAL?
# ============================================

print("\n" + "="*70)
print("6. CHECKPOINT: GOAL ASSESSMENT")
print("="*70)

GOAL = 600

if total_matched >= GOAL:
    print(f"\n✅ SUCCESS! Matched {total_matched} ≥ {GOAL} cell lines")
    print("   → Proceeding to save results")
    status = "PASS"
else:
    print(f"\n⚠️  BELOW GOAL: Matched {total_matched} < {GOAL}")
    print(f"   Shortfall: {GOAL - total_matched} cell lines")
    print("\n   Options:")
    print("   1. Lower fuzzy threshold to 80 (may introduce errors)")
    print("   2. Proceed with current matches (still substantial dataset)")
    print("   3. Manual curation of remaining unmatched cells")
    status = "PARTIAL"

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "="*70)
print("7. SAVING RESULTS")
print("="*70)

# Save matched cell lines
output_file = DATA_PROCESSED / "cell_line_mapping.csv"
final_matches.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
print(f"  → {len(final_matches)} matched cell lines")

# Save unmatched GDSC cells for reference
unmatched_gdsc = gdsc_cells[~gdsc_cells['CELL_LINE_NAME'].isin(final_matches['GDSC_CELL_LINE'])]
unmatched_file = DATA_PROCESSED / "unmatched_gdsc_cells.csv"
unmatched_gdsc.to_csv(unmatched_file, index=False)
print(f"\n✓ Saved unmatched: {unmatched_file}")
print(f"  → {len(unmatched_gdsc)} unmatched GDSC cell lines")

# Create summary report
summary = {
    'Total_GDSC_Cells': len(gdsc_cells),
    'Total_DepMap_Cells': len(depmap_cells),
    'Matched_Cells': total_matched,
    'Match_Rate': f"{total_matched/len(gdsc_cells)*100:.1f}%",
    'Status': status,
    'Goal_Met': total_matched >= GOAL,
    'COSMIC_Matches': len(cosmic_matches),
    'Exact_Matches': len(exact_matches),
    'Fuzzy_Matches': len(fuzzy_matches)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(DATA_PROCESSED / "matching_summary.csv", index=False)

print("\n" + "="*70)
print("8. NEXT STEPS")
print("="*70)

if status == "PASS":
    print("\n✅ Ready for final integration!")
    print("\nRun: python scripts/data_processing/04_prepare_final_dataset.py")
else:
    print("\n⚠️  Review matches before proceeding")
    print("\nOptions:")
    print("  A. Proceed with current matches (recommended if >500)")
    print("  B. Lower threshold and re-run")
    print("  C. Manual curation")

print("\n" + "="*70)
print("MATCHING COMPLETE")
print("="*70)
