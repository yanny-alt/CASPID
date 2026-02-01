#!/usr/bin/env python3
"""
SIMPLIFIED PROTEIN PREPARATION FOR DOCKING
Let ChimeraX handle ALL cleaning (it's better at it anyway)

This script just:
1. Creates ChimeraX scripts
2. Calculates binding site centers AFTER you run ChimeraX

Author: CASPID Research Team
"""

import json
from pathlib import Path

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DATA_RAW = PROJECT_ROOT / "DATA" / "PDB"
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
DOCKING_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PROTEIN PREPARATION - CHIMERAX WORKFLOW")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

PROTEINS = {
    'EGFR': {
        'pdb_file': '1m17.pdb',
        'chain': 'A',
        'description': 'EGFR kinase domain with erlotinib',
    },
    'BRAF': {
        'pdb_file': '4mne.pdb',
        'chain': 'A',
        'description': 'BRAF V600E with vemurafenib',
    }
}

# Approximate binding site centers (from literature/visualization)
# You can refine these in ChimeraX if needed
BINDING_CENTERS = {
    'EGFR': {'x': 29.0, 'y': 6.3, 'z': 54.2},
    'BRAF': {'x': 6.6, 'y': -16.9, 'z': -35.7}
}

print("\n" + "="*70)
print("CREATING CHIMERAX SCRIPTS")
print("="*70)

for protein_name, config in PROTEINS.items():
    print(f"\n{protein_name}:")
    
    input_pdb = DATA_RAW / config['pdb_file']
    output_pdb = DOCKING_DIR / f"{protein_name.lower()}_prepared.pdb"
    script_file = DOCKING_DIR / f"{protein_name.lower()}_prepare.cxc"
    
    if not input_pdb.exists():
        print(f"  ✗ {input_pdb} not found!")
        continue
    
    # Create ChimeraX script
    script_content = f"""# ChimeraX Preparation Script for {protein_name}
# {config['description']}

# Open structure
open {input_pdb}

# Keep only chain {config['chain']}
delete ~/{config['chain']}

# Remove water, ions, and small molecules
delete solvent
delete ions
delete ligand

# Add hydrogens at pH 7.4
addh hbond true

# Save prepared structure
save {output_pdb} format pdb models #1

# Show binding site for visual verification
# (You can adjust box center if needed)
"""
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"  ✓ Script created: {script_file.name}")
    print(f"  → Will create: {output_pdb.name}")

# ============================================
# CREATE DOCKING CONFIGURATION
# ============================================

print("\n" + "="*70)
print("CREATING DOCKING CONFIGURATION")
print("="*70)

docking_config = {}

for protein_name in PROTEINS.keys():
    prepared_pdb = DOCKING_DIR / f"{protein_name.lower()}_prepared.pdb"
    center = BINDING_CENTERS[protein_name]
    
    docking_config[protein_name] = {
        'receptor_file': str(prepared_pdb),
        'center_x': center['x'],
        'center_y': center['y'],
        'center_z': center['z'],
        'size_x': 25.0,
        'size_y': 25.0,
        'size_z': 25.0,
        'exhaustiveness': 32,
        'num_modes': 20,
        'description': PROTEINS[protein_name]['description']
    }

config_file = DOCKING_DIR / "docking_config.json"
with open(config_file, 'w') as f:
    json.dump(docking_config, f, indent=2)

print(f"\n✓ Configuration saved: {config_file.name}")

# ============================================
# INSTRUCTIONS
# ============================================

print("\n" + "="*70)
print("NEXT STEPS - RUN IN CHIMERAX")
print("="*70)

print("\nOption 1: Run scripts automatically")
print("-" * 70)
for protein_name in PROTEINS.keys():
    script_file = DOCKING_DIR / f"{protein_name.lower()}_prepare.cxc"
    print(f"\n{protein_name}:")
    print(f"  /Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX \\")
    print(f"    --nogui --script {script_file}")

print("\n\nOption 2: Run manually in ChimeraX GUI")
print("-" * 70)

for protein_name, config in PROTEINS.items():
    input_pdb = DATA_RAW / config['pdb_file']
    output_pdb = DOCKING_DIR / f"{protein_name.lower()}_prepared.pdb"
    
    print(f"\n{protein_name}:")
    print(f"  1. Open ChimeraX")
    print(f"  2. File → Open → {input_pdb}")
    print(f"  3. In command line, run:")
    print(f"     delete ~/{config['chain']}")
    print(f"     delete solvent")
    print(f"     delete ions")
    print(f"     delete ligand")
    print(f"     addh")
    print(f"     save {output_pdb} format pdb")

print("\n" + "="*70)
print("AFTER CHIMERAX PREPARATION")
print("="*70)

print("\nVerify these files exist:")
for protein_name in PROTEINS.keys():
    output_pdb = DOCKING_DIR / f"{protein_name.lower()}_prepared.pdb"
    print(f"  - {output_pdb}")

print("\nThen proceed to:")
print("  python scripts/docking/02_prepare_ligands.py")

print("\n" + "="*70)
print("FILES CREATED")
print("="*70)

print(f"\n✓ {DOCKING_DIR}:")
for protein_name in PROTEINS.keys():
    script_file = f"{protein_name.lower()}_prepare.cxc"
    print(f"  - {script_file} (ChimeraX script)")
print(f"  - docking_config.json (binding box configuration)")

print("\n" + "="*70)
