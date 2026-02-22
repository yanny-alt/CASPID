#!/usr/bin/env python3
"""
CRITICAL LIGAND PREPARATION FOR MEK DOCKING
Converts 4 MEK compounds (SMILES → PDBQT) for AutoDock Vina

MEK VALIDATION VERSION - Extensive validation and error checking
- Reads MEK compounds from dockable_compounds.csv
- Generates 3D structures with RDKit
- Creates PDBQT files with Meeko
- Quality checks at every step

Author: CASPID Research Team
Date: February 2026
"""

import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
LIGANDS_DIR = DOCKING_DIR / "LIGANDS"
LIGANDS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("CRITICAL: MEK LIGAND PREPARATION FOR DOCKING")
print("="*70)
print("\nMEK validation - 4 MEK inhibitors with full validation")

# ============================================
# DEPENDENCY CHECK
# ============================================

print("\n" + "="*70)
print("1. CHECKING DEPENDENCIES")
print("="*70)

required_packages = {
    'RDKit': 'rdkit',
    'Pandas': 'pandas',
}

# Check Meeko separately (it has submodules)
meeko_installed = False
try:
    from meeko import MoleculePreparation
    meeko_installed = True
except ImportError:
    pass

missing = []
for name, package in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - NOT FOUND")
        missing.append(name)

# Check Meeko
if meeko_installed:
    print(f"  ✓ Meeko")
else:
    print(f"  ✗ Meeko - NOT FOUND")
    missing.append('Meeko')

if missing:
    print(f"\n❌ MISSING DEPENDENCIES: {', '.join(missing)}")
    print("\nInstall with:")
    print("  conda install -c conda-forge rdkit")
    print("  pip install meeko")
    sys.exit(1)

print("\n✓ All dependencies found")

# ============================================
# LOAD COMPOUNDS
# ============================================

print("\n" + "="*70)
print("2. LOADING MEK COMPOUNDS")
print("="*70)

compounds_file = DATA_PROCESSED / "dockable_compounds.csv"

if not compounds_file.exists():
    print(f"\n❌ ERROR: {compounds_file} not found!")
    print("Please ensure dockable_compounds.csv is in DATA/PROCESSED/")
    sys.exit(1)

# Load compound data
compounds_df = pd.read_csv(compounds_file)

print(f"\n✓ Loaded: {compounds_file.name}")
print(f"  Total compounds: {len(compounds_df)}")

# Check required columns
required_cols = ['DRUG_NAME', 'SMILES', 'TARGET']
missing_cols = [col for col in required_cols if col not in compounds_df.columns]

if missing_cols:
    print(f"\n❌ ERROR: Missing columns: {missing_cols}")
    print(f"   Available columns: {list(compounds_df.columns)}")
    sys.exit(1)

# Filter for MEK compounds only
print("\n  Filtering for MEK inhibitors...")
mek_compounds = compounds_df[
    compounds_df['TARGET'].str.contains('MEK', case=False, na=False)
].copy()

print(f"  MEK compounds found: {len(mek_compounds)}")

# Get compounds that actually made it to final MEK dataset
final_dataset = pd.read_csv(DATA_PROCESSED / "caspid_mek_integrated_dataset.csv")
final_drug_names = set(final_dataset['DRUG_NAME'].unique())

print(f"  Compounds in final MEK dataset: {len(final_drug_names)}")

# Filter to only compounds in final dataset
compounds_df = mek_compounds[mek_compounds['DRUG_NAME'].isin(final_drug_names)].copy()
compounds_df = compounds_df.drop_duplicates(subset='DRUG_NAME')

print(f"  MEK compounds to prepare: {len(compounds_df)}")

if len(compounds_df) == 0:
    print("\n❌ ERROR: No MEK compounds found in final dataset!")
    sys.exit(1)

# Show the compounds
print("\n  MEK inhibitors:")
for idx, row in compounds_df.iterrows():
    target = row['TARGET']
    print(f"    {row['DRUG_NAME']} ({target})")

# ============================================
# SMILES VALIDATION
# ============================================

print("\n" + "="*70)
print("3. VALIDATING SMILES STRUCTURES")
print("="*70)

valid_compounds = []
invalid_compounds = []

for idx, row in compounds_df.iterrows():
    drug_name = row['DRUG_NAME']
    smiles = row['SMILES']
    
    # Check SMILES is not empty
    if pd.isna(smiles) or smiles.strip() == '':
        print(f"\n  ✗ {drug_name}: Empty SMILES")
        invalid_compounds.append({
            'drug': drug_name,
            'reason': 'Empty SMILES'
        })
        continue
    
    # Try to parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"\n  ✗ {drug_name}: Invalid SMILES")
        invalid_compounds.append({
            'drug': drug_name,
            'reason': 'Invalid SMILES'
        })
        continue
    
    # Check molecular weight (typical drugs: 200-800 Da)
    mw = Descriptors.MolWt(mol)
    
    if mw < 100 or mw > 1000:
        print(f"\n  ⚠️  {drug_name}: Unusual MW = {mw:.1f} Da")
        # Still proceed but warn
    
    # Check atom count
    num_atoms = mol.GetNumAtoms()
    
    if num_atoms < 10:
        print(f"\n  ⚠️  {drug_name}: Very small molecule ({num_atoms} atoms)")
    
    valid_compounds.append({
        'drug_name': drug_name,
        'smiles': smiles,
        'mol': mol,
        'mw': mw,
        'num_atoms': num_atoms
    })
    
    print(f"  ✓ {drug_name}: MW={mw:.1f} Da, Atoms={num_atoms}")

print(f"\n✓ Valid MEK compounds: {len(valid_compounds)}/{len(compounds_df)}")

if invalid_compounds:
    print(f"\n⚠️  Invalid compounds: {len(invalid_compounds)}")
    for item in invalid_compounds:
        print(f"    - {item['drug']}: {item['reason']}")

if len(valid_compounds) == 0:
    print("\n❌ ERROR: No valid MEK compounds to process!")
    sys.exit(1)

# ============================================
# 3D STRUCTURE GENERATION
# ============================================

print("\n" + "="*70)
print("4. GENERATING 3D STRUCTURES")
print("="*70)

structures_3d = []

for compound in valid_compounds:
    drug_name = compound['drug_name']
    mol = compound['mol']
    
    print(f"\n  {drug_name}:")
    
    # Add hydrogens
    mol_h = Chem.AddHs(mol)
    print(f"    ✓ Hydrogens added")
    
    # Generate 3D coordinates
    try:
        # Use ETKDG method (best for drug-like molecules)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # Reproducibility
        
        result = AllChem.EmbedMolecule(mol_h, params)
        
        if result == -1:
            print(f"    ✗ 3D embedding failed")
            continue
        
        print(f"    ✓ 3D coordinates generated")
        
        # Optimize geometry with MMFF94
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            print(f"    ✓ Geometry optimized (MMFF94)")
        except:
            print(f"    ⚠️  MMFF optimization failed, using UFF")
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        
        structures_3d.append({
            'drug_name': drug_name,
            'mol_3d': mol_h,
            'smiles': compound['smiles']
        })
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        continue

print(f"\n✓ 3D structures generated: {len(structures_3d)}/{len(valid_compounds)}")

if len(structures_3d) == 0:
    print("\n❌ ERROR: No 3D structures generated!")
    sys.exit(1)

# ============================================
# PDBQT CONVERSION WITH MEEKO
# ============================================

print("\n" + "="*70)
print("5. CONVERTING TO PDBQT FORMAT (MEEKO)")
print("="*70)

try:
    from meeko import MoleculePreparation
    from meeko import PDBQTWriterLegacy
except ImportError:
    try:
        # Try alternative import (newer Meeko versions)
        from meeko import MoleculePreparation
        from meeko import RDKitMolCreate
        PDBQTWriterLegacy = None
    except ImportError:
        print("\n❌ ERROR: Cannot import Meeko properly")
        print("Try: pip install meeko --upgrade")
        sys.exit(1)

successful_conversions = []
failed_conversions = []

for structure in structures_3d:
    drug_name = structure['drug_name']
    mol_3d = structure['mol_3d']
    
    print(f"\n  {drug_name}:")
    
    try:
        # Check for multiple fragments (salts, counterions)
        fragments = Chem.GetMolFrags(mol_3d, asMols=True)
        
        if len(fragments) > 1:
            print(f"    ⚠️  Multiple fragments detected ({len(fragments)})")
            print(f"    → Keeping largest fragment (removing salts/counterions)")
            # Keep largest fragment
            mol_3d = max(fragments, key=lambda m: m.GetNumAtoms())
        
        # Prepare molecule with Meeko
        preparator = MoleculePreparation()
        
        # Prepare molecule (adds charges, sets rotatable bonds)
        mol_setups = preparator.prepare(mol_3d)
        
        if len(mol_setups) == 0:
            print(f"    ✗ Meeko preparation failed")
            failed_conversions.append(drug_name)
            continue
        
        print(f"    ✓ Meeko preparation complete")
        
        # Write PDBQT
        output_file = LIGANDS_DIR / f"{drug_name.lower().replace(' ', '_').replace('-', '_')}.pdbqt"
        
        # Get PDBQT string - try different API versions
        try:
            # Newer Meeko API
            pdbqt_string = mol_setups[0].write_pdbqt_string()
        except AttributeError:
            try:
                # Older Meeko API - returns tuple (string, is_ok, error_msg)
                result = PDBQTWriterLegacy.write_string(mol_setups[0])
                if isinstance(result, tuple):
                    pdbqt_string, is_ok, error_msg = result
                    if not is_ok:
                        raise Exception(f"PDBQT writing failed: {error_msg}")
                else:
                    pdbqt_string = result
            except Exception as e:
                raise Exception(f"All Meeko API attempts failed: {e}")
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(pdbqt_string)
        
        # Verify file was created
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"    ✓ PDBQT saved: {output_file.name} ({file_size} bytes)")
            
            successful_conversions.append({
                'drug_name': drug_name,
                'pdbqt_file': str(output_file),
                'smiles': structure['smiles']
            })
        else:
            print(f"    ✗ File not created")
            failed_conversions.append(drug_name)
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        failed_conversions.append(drug_name)

print(f"\n✓ PDBQT files created: {len(successful_conversions)}/{len(structures_3d)}")

if failed_conversions:
    print(f"\n⚠️  Failed conversions: {len(failed_conversions)}")
    for drug in failed_conversions:
        print(f"    - {drug}")

# ============================================
# SAVE LIGAND SUMMARY
# ============================================

print("\n" + "="*70)
print("6. SAVING MEK LIGAND SUMMARY")
print("="*70)

if len(successful_conversions) > 0:
    summary_df = pd.DataFrame(successful_conversions)
    summary_file = LIGANDS_DIR / "mek_ligands_prepared.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*70)
print("MEK LIGAND PREPARATION COMPLETE")
print("="*70)

print(f"\n📊 FINAL SUMMARY:")
print(f"  MEK compounds in dataset: {len(compounds_df)}")
print(f"  Valid SMILES: {len(valid_compounds)}")
print(f"  3D structures generated: {len(structures_3d)}")
print(f"  ✓ PDBQT files created: {len(successful_conversions)}")

if len(successful_conversions) < len(compounds_df):
    print(f"  ⚠️  Some compounds failed: {len(compounds_df) - len(successful_conversions)}")

print(f"\n📁 OUTPUT FILES:")
print(f"  Directory: {LIGANDS_DIR}")
print(f"  MEK ligand files created:")
for item in successful_conversions:
    filename = Path(item['pdbqt_file']).name
    print(f"    - {filename}")

# Check if we have all 4 MEK compounds
if len(successful_conversions) == 4:
    print("\n✅ ALL 4 MEK LIGANDS READY FOR DOCKING!")
    print(f"\n📝 MEK inhibitors prepared: {len(successful_conversions)}")
    print("\nNext steps:")
    print("  1. Update docking_config.json with MEK1 entry")
    print("  2. Run: python scripts/docking/03_run_vina_SAFE.py")
elif len(successful_conversions) >= 3:
    print("\n⚠️  MOST MEK COMPOUNDS READY")
    print(f"   {len(successful_conversions)}/4 compounds prepared")
    print("   You can proceed with MEK validation")
else:
    print("\n❌ CRITICAL: TOO FEW MEK COMPOUNDS")
    print(f"   Only {len(successful_conversions)}/4 compounds prepared")
    print("   Need at least 3 for meaningful MEK validation")

print("\n" + "="*70)
