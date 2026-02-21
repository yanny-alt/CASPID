#!/usr/bin/env python3
"""
Convert prepared protein PDB files to PDBQT format
Required for AutoDock Vina
"""

from pathlib import Path
import subprocess

DOCKING_DIR = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID/DATA/DOCKING")

proteins = {
    'egfr': 'egfr_prepared.pdb',
    'braf': 'braf_prepared.pdb',
    'mek1': 'mek1_prepared.pdb'
}

print("="*70)
print("CONVERTING PROTEINS: PDB → PDBQT")
print("="*70)

for name, pdb_file in proteins.items():
    input_pdb = DOCKING_DIR / pdb_file
    output_pdbqt = DOCKING_DIR / f"{name}_prepared.pdbqt"
    
    print(f"\n{name.upper()}:")
    print(f"  Input:  {pdb_file}")
    
    if not input_pdb.exists():
        print(f"  ✗ Error: {pdb_file} not found")
        continue
    
    # Try using obabel (if available)
    try:
        cmd = [
            'obabel',
            str(input_pdb),
            '-O', str(output_pdbqt),
            '-xr'  # rigid receptor
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and output_pdbqt.exists():
            print(f"  ✓ Converted with obabel")
            print(f"  Output: {output_pdbqt.name}")
        else:
            raise Exception("obabel failed")
    
    except:
        # Fallback: Use Meeko for proteins
        print(f"  Trying Meeko...")
        
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            from rdkit import Chem
            
            # Read PDB with RDKit
            mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False)
            
            if mol is None:
                # Try with sanitize=False
                mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False, sanitize=False)
            
            if mol is not None:
                # Prepare
                preparator = MoleculePreparation()
                mol_setups = preparator.prepare(mol)
                
                # Write PDBQT
                result = PDBQTWriterLegacy.write_string(mol_setups[0])
                if isinstance(result, tuple):
                    pdbqt_string = result[0]
                else:
                    pdbqt_string = result
                
                with open(output_pdbqt, 'w') as f:
                    f.write(pdbqt_string)
                
                print(f"  ✓ Converted with Meeko")
                print(f"  Output: {output_pdbqt.name}")
            else:
                print(f"  ✗ Could not convert")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"\n  Manual conversion needed:")
            print(f"  Use ChimeraX or MGLTools to convert {pdb_file} to PDBQT")

print("\n" + "="*70)
print("CONVERSION COMPLETE")
print("="*70)

# Check results
print("\nVerifying files:")
for name in proteins.keys():
    pdbqt_file = DOCKING_DIR / f"{name}_prepared.pdbqt"
    if pdbqt_file.exists():
        size = pdbqt_file.stat().st_size
        print(f"  ✓ {pdbqt_file.name} ({size:,} bytes)")
    else:
        print(f"  ✗ {pdbqt_file.name} - NOT FOUND")

print("\n" + "="*70)
