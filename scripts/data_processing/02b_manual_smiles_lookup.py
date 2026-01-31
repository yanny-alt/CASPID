#!/usr/bin/env python3
"""
Manual SMILES lookup for failed compounds
"""

import pandas as pd
from pathlib import Path
import requests
import time

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"

print("="*70)
print("MANUAL SMILES LOOKUP FOR FAILED COMPOUNDS")
print("="*70)

# Load existing results
compounds_df = pd.read_csv(DATA_PROCESSED / "compounds_with_smiles.csv")

# Failed compounds
failed = compounds_df[~compounds_df['Success']]
print(f"\nFailed compounds: {len(failed)}")
print(failed[['DRUG_NAME', 'TARGET']])

# Manual SMILES database (from literature/databases)
manual_smiles = {
    'Cetuximab': {
        'smiles': None,  # Antibody - no small molecule structure
        'note': 'Monoclonal antibody - exclude from docking (not a small molecule)',
        'exclude': True
    },
    'RAF_9304': {
        'smiles': None,  # Try alternative search
        'note': 'Proprietary compound - will search with alternatives',
        'exclude': False
    },
    'Sapitinib': {
        'smiles': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1',  # From DrugBank
        'chembl_id': 'CHEMBL2105735',
        'note': 'Found in DrugBank',
        'exclude': False
    }
}

# Try alternative searches
def search_alternative_names(drug_name):
    """Try synonyms and alternative spellings"""
    
    # Alternative names
    alternatives = {
        'RAF_9304': ['RAF-9304', 'RAF 9304', 'pan-RAF inhibitor']
    }
    
    if drug_name in alternatives:
        for alt_name in alternatives[drug_name]:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{alt_name}/property/CanonicalSMILES/JSON"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'PropertyTable' in data:
                        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                        print(f"  ✓ Found {drug_name} as '{alt_name}': {smiles[:50]}...")
                        return smiles
                
                time.sleep(0.5)
            except:
                continue
    
    return None

print("\n" + "="*70)
print("SEARCHING FOR MISSING SMILES")
print("="*70)

updates = []

for idx, row in failed.iterrows():
    drug_name = row['DRUG_NAME']
    
    print(f"\n{drug_name}:")
    
    # Check manual database
    if drug_name in manual_smiles:
        info = manual_smiles[drug_name]
        
        if info.get('exclude'):
            print(f"  ⚠️  {info['note']}")
            print(f"  → Will EXCLUDE from analysis (not dockable)")
            
            updates.append({
                'DRUG_NAME': drug_name,
                'SMILES': None,
                'Note': info['note'],
                'Exclude': True
            })
        
        elif info.get('smiles'):
            print(f"  ✓ Found manually: {info['smiles'][:50]}...")
            print(f"  Source: {info.get('note', 'Manual lookup')}")
            
            # Update DataFrame
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'SMILES'] = info['smiles']
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'ChEMBL_ID'] = info.get('chembl_id')
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'Source'] = 'Manual'
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'Success'] = True
            
            updates.append({
                'DRUG_NAME': drug_name,
                'SMILES': info['smiles'],
                'Note': info['note'],
                'Exclude': False
            })
    
    # Try alternative search
    else:
        alt_smiles = search_alternative_names(drug_name)
        if alt_smiles:
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'SMILES'] = alt_smiles
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'Source'] = 'Alternative search'
            compounds_df.loc[compounds_df['DRUG_NAME'] == drug_name, 'Success'] = True
            
            updates.append({
                'DRUG_NAME': drug_name,
                'SMILES': alt_smiles,
                'Note': 'Found via alternative search',
                'Exclude': False
            })

# Summary
print("\n" + "="*70)
print("UPDATED SUMMARY")
print("="*70)

total = len(compounds_df)
success = compounds_df['Success'].sum()
excluded = sum(1 for u in updates if u.get('Exclude'))

print(f"\nTotal EGFR/BRAF compounds: {total}")
print(f"✓ With SMILES: {success} ({success/total*100:.1f}%)")
print(f"✗ Excluded (antibodies, etc.): {excluded}")
print(f"? Still missing: {total - success - excluded}")

# Save updated file
compounds_df.to_csv(DATA_PROCESSED / "compounds_with_smiles.csv", index=False)
print(f"\n✓ Updated: {DATA_PROCESSED / 'compounds_with_smiles.csv'}")

# Create final dockable compounds file
dockable = compounds_df[compounds_df['Success'] == True].copy()
dockable.to_csv(DATA_PROCESSED / "dockable_compounds.csv", index=False)

print(f"✓ Created: {DATA_PROCESSED / 'dockable_compounds.csv'}")
print(f"  → {len(dockable)} compounds ready for docking")

# Breakdown by target
egfr_count = (dockable['TARGET'].str.contains('EGFR', case=False)).sum()
braf_count = (dockable['TARGET'].str.contains('BRAF', case=False)).sum()

print(f"\n  EGFR inhibitors: {egfr_count}")
print(f"  BRAF inhibitors: {braf_count}")

print("\n" + "="*70)
print("READY FOR NEXT STEP")
print("="*70)
print("\nRun: python scripts/data_processing/03_integrate_gdsc_depmap.py")
print("="*70)
