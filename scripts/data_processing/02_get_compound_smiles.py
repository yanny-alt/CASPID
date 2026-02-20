#!/usr/bin/env python3
"""
Get SMILES structures for GDSC compounds from ChEMBL/PubChem
CRITICAL: We need SMILES for docking
"""

import pandas as pd
import requests
from pathlib import Path
import time
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_RAW = PROJECT_ROOT / "DATA"
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"
DATA_PROCESSED.mkdir(exist_ok=True)

print("="*70)
print("FETCHING COMPOUND SMILES FROM ChEMBL")
print("="*70)

# Load compound annotations
compounds = pd.read_csv(DATA_RAW / "GDSC" / "compound_annotations.csv")
print(f"\nTotal compounds: {len(compounds)}")

# Focus on EGFR, BRAF, and MEK drugs
target_compounds = compounds[
    compounds['TARGET'].str.contains('EGFR|BRAF|MEK', na=False, case=False, regex=True)
].copy()

print(f"EGFR/BRAF/MEK compounds: {len(target_compounds)}")

def get_smiles_from_chembl(drug_name):
    """
    Query ChEMBL API for SMILES
    """
    try:
        # Clean drug name
        name = drug_name.strip()
        
        # Query ChEMBL
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={name}&format=json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'molecules' in data and len(data['molecules']) > 0:
                # Get first result
                molecule = data['molecules'][0]
                
                # Get canonical SMILES
                smiles = molecule.get('molecule_structures', {}).get('canonical_smiles')
                chembl_id = molecule.get('molecule_chembl_id')
                
                return {
                    'smiles': smiles,
                    'chembl_id': chembl_id,
                    'source': 'ChEMBL',
                    'success': smiles is not None
                }
        
        return {'smiles': None, 'chembl_id': None, 'source': None, 'success': False}
    
    except Exception as e:
        print(f"  Error for {drug_name}: {e}")
        return {'smiles': None, 'chembl_id': None, 'source': None, 'success': False}

def get_smiles_from_pubchem(drug_name):
    """
    Fallback: Query PubChem for SMILES
    """
    try:
        name = drug_name.strip()
        
        # PubChem PUG REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                properties = data['PropertyTable']['Properties'][0]
                smiles = properties.get('CanonicalSMILES')
                cid = properties.get('CID')
                
                return {
                    'smiles': smiles,
                    'pubchem_cid': cid,
                    'source': 'PubChem',
                    'success': smiles is not None
                }
        
        return {'smiles': None, 'pubchem_cid': None, 'source': None, 'success': False}
    
    except Exception as e:
        return {'smiles': None, 'pubchem_cid': None, 'source': None, 'success': False}

# Fetch SMILES for all target compounds
print("\nFetching SMILES (this will take a few minutes)...")
print("Trying ChEMBL first, then PubChem as fallback...")

results = []

for idx, row in tqdm(target_compounds.iterrows(), total=len(target_compounds)):
    drug_name = row['DRUG_NAME']
    
    # Try ChEMBL first
    result = get_smiles_from_chembl(drug_name)
    
    # If ChEMBL fails, try PubChem
    if not result['success']:
        time.sleep(0.5)  # Rate limiting
        result = get_smiles_from_pubchem(drug_name)
    
    # Store result
    results.append({
        'DRUG_ID': row['DRUG_ID'],
        'DRUG_NAME': drug_name,
        'TARGET': row['TARGET'],
        'SMILES': result.get('smiles'),
        'ChEMBL_ID': result.get('chembl_id'),
        'PubChem_CID': result.get('pubchem_cid'),
        'Source': result.get('source'),
        'Success': result.get('success')
    })
    
    time.sleep(0.3)  # Be nice to APIs

# Create DataFrame
results_df = pd.DataFrame(results)

# Summary
print("\n" + "="*70)
print("SMILES RETRIEVAL SUMMARY")
print("="*70)

total = len(results_df)
success = results_df['Success'].sum()
from_chembl = (results_df['Source'] == 'ChEMBL').sum()
from_pubchem = (results_df['Source'] == 'PubChem').sum()

print(f"\nTotal compounds queried: {total}")
print(f"✓ SMILES found: {success} ({success/total*100:.1f}%)")
print(f"  - From ChEMBL: {from_chembl}")
print(f"  - From PubChem: {from_pubchem}")
print(f"✗ Not found: {total - success}")

# Show successful retrievals
print("\n" + "="*70)
print("SUCCESSFUL RETRIEVALS")
print("="*70)

successful = results_df[results_df['Success']]
print(f"\nEGFR compounds with SMILES: {(successful['TARGET'].str.contains('EGFR', case=False)).sum()}")
print(f"BRAF compounds with SMILES: {(successful['TARGET'].str.contains('BRAF', case=False)).sum()}")
print(f"MEK compounds with SMILES: {(successful['TARGET'].str.contains('MEK', case=False)).sum()}")

print("\nExamples:")
print(successful[['DRUG_NAME', 'TARGET', 'Source']].head(10))

# Show failures
if total > success:
    print("\n" + "="*70)
    print("FAILED RETRIEVALS (Manual Check Needed)")
    print("="*70)
    
    failed = results_df[~results_df['Success']]
    print(failed[['DRUG_NAME', 'TARGET']])
    
    print("\nThese will need manual SMILES lookup or exclusion from analysis")

# Save results
output_file = DATA_PROCESSED / "compounds_with_smiles.csv"
results_df.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")

# Also save only successful compounds
successful_file = DATA_PROCESSED / "compounds_smiles_only.csv"
successful.to_csv(successful_file, index=False)

print(f"✓ Saved successful only to: {successful_file}")

print("\n" + "="*70)
print("NEXT STEP")
print("="*70)
print("\nRun: python scripts/data_processing/03_integrate_gdsc_depmap.py")
print("(This will match GDSC cell lines with DepMap using fuzzy matching)")
print("="*70)
