#!/usr/bin/env python3
"""
Physicochemical Feature Extraction for Molecular Docking Analysis

Extracts electrostatic interactions, van der Waals energies, desolvation metrics,
surface properties, and ligand molecular descriptors from docking poses.

Author: CASPID Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.spatial import distance
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
CONSENSUS_DIR = DOCKING_DIR / "CONSENSUS_RESULTS"
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES"

# Output directory
PHYSICOCHEMICAL_DIR = FEATURES_DIR / "03_physicochemical"
PHYSICOCHEMICAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PHYSICOCHEMICAL_DIR / f"physicochemical_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("PHYSICOCHEMICAL FEATURE EXTRACTION")
print("="*70)
log("Starting physicochemical feature extraction")

def parse_pdbqt_with_charges(pdbqt_file, top_pose_only=True):
    """
    Parse PDBQT and extract coordinates with partial charges
    Returns: list of (x, y, z, charge, atom_type)
    """
    atoms = []
    in_model = False
    has_model_tag = False
    
    try:
        with open(pdbqt_file) as f:
            lines = f.readlines()
        
        has_model_tag = any(line.startswith('MODEL') for line in lines)
        
        for line in lines:
            if line.startswith('MODEL'):
                in_model = True
                continue
            elif line.startswith('ENDMDL') and top_pose_only:
                break
            
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            if has_model_tag and not in_model:
                continue
            
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                # Extract partial charge (column 70-76 in PDBQT)
                try:
                    charge = float(line[70:76])
                except:
                    charge = 0.0
                
                # Extract atom type (column 77-79)
                atom_type = line[77:79].strip()
                
                atoms.append((x, y, z, charge, atom_type))
            except:
                continue
        
        return atoms
    except:
        return []

def calculate_electrostatic_energy(lig_atoms, prot_atoms, dielectric=4.0, cutoff=10.0):
    """
    Calculate simplified Coulombic electrostatic energy
    E = (q1 * q2) / (dielectric * r)
    """
    if not lig_atoms or not prot_atoms:
        return np.nan
    
    total_energy = 0.0
    
    # Coulomb constant (kcal·Å/mol·e²)
    ke = 332.0
    
    for lig_atom in lig_atoms:
        lx, ly, lz, lq, _ = lig_atom
        
        if abs(lq) < 0.01:  # Skip neutral atoms
            continue
        
        for prot_atom in prot_atoms:
            px, py, pz, pq, _ = prot_atom
            
            if abs(pq) < 0.01:  # Skip neutral atoms
                continue
            
            # Calculate distance
            r = np.sqrt((lx-px)**2 + (ly-py)**2 + (lz-pz)**2)
            
            if r < 0.5 or r > cutoff:  # Skip too close or too far
                continue
            
            # Coulombic energy
            e = (ke * lq * pq) / (dielectric * r)
            total_energy += e
    
    return float(total_energy)

def calculate_vdw_energy(lig_coords, prot_coords, cutoff=8.0):
    """
    Calculate simplified Lennard-Jones 6-12 potential
    E = 4ε[(σ/r)^12 - (σ/r)^6]
    Using generic parameters
    """
    if not lig_coords or not prot_coords:
        return np.nan
    
    # Generic LJ parameters
    epsilon = 0.2  # kcal/mol
    sigma = 3.5    # Angstroms
    
    lig_arr = np.array(lig_coords)
    prot_arr = np.array(prot_coords)
    
    distances = distance.cdist(lig_arr, prot_arr)
    
    # Only consider interactions within cutoff
    valid_dists = distances[distances < cutoff]
    
    if len(valid_dists) == 0:
        return 0.0
    
    # Avoid division by zero
    valid_dists = valid_dists[valid_dists > 0.5]
    
    # LJ potential
    sr6 = (sigma / valid_dists) ** 6
    sr12 = sr6 ** 2
    
    energy = 4 * epsilon * (sr12 - sr6)
    
    return float(np.sum(energy))

def calculate_sasa_approximate(coords, probe_radius=1.4):
    """
    Approximate solvent accessible surface area
    Using sphere approximation
    """
    if len(coords) < 1:
        return np.nan
    
    # Approximate each atom as sphere with typical vdW radius
    atom_radius = 1.7  # Angstroms (average)
    
    # SASA per atom
    sasa_per_atom = 4 * np.pi * (atom_radius + probe_radius) ** 2
    
    # Simple approximation: some atoms are buried
    # Use contact counting
    dists = distance.pdist(np.array(coords))
    
    # Atoms with close neighbors are more buried
    burial_factor = 1.0 - (np.sum(dists < 4.0) / len(coords)) * 0.3
    burial_factor = max(0.3, min(1.0, burial_factor))
    
    total_sasa = len(coords) * sasa_per_atom * burial_factor
    
    return float(total_sasa)

def count_polar_atoms(atoms):
    """Count polar atoms (N, O, S)"""
    polar_types = ['N', 'O', 'S', 'NA', 'OA', 'SA']
    return sum(1 for atom in atoms if atom[4] in polar_types)

def count_nonpolar_atoms(atoms):
    """Count nonpolar atoms (C, H)"""
    nonpolar_types = ['C', 'A', 'H', 'HD']
    return sum(1 for atom in atoms if atom[4] in nonpolar_types)

def calculate_molecular_descriptors(smiles):
    """
    Calculate RDKit molecular descriptors
    """
    descriptors = {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {k: np.nan for k in [
                'mol_weight', 'logp', 'num_hbd', 'num_hba', 
                'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings',
                'formal_charge', 'num_heavy_atoms', 'fraction_csp3'
            ]}
        
        # Basic descriptors
        descriptors['mol_weight'] = Descriptors.MolWt(mol)
        descriptors['logp'] = Crippen.MolLogP(mol)
        descriptors['num_hbd'] = Lipinski.NumHDonors(mol)
        descriptors['num_hba'] = Lipinski.NumHAcceptors(mol)
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['num_rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
        descriptors['num_aromatic_rings'] = Lipinski.NumAromaticRings(mol)
        descriptors['formal_charge'] = Chem.GetFormalCharge(mol)
        descriptors['num_heavy_atoms'] = Lipinski.HeavyAtomCount(mol)
        descriptors['fraction_csp3'] = Lipinski.FractionCSP3(mol)
        
        return descriptors
    
    except:
        return {k: np.nan for k in [
            'mol_weight', 'logp', 'num_hbd', 'num_hba', 
            'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings',
            'formal_charge', 'num_heavy_atoms', 'fraction_csp3'
        ]}

def extract_physicochemical_features_single_pose(job):
    """Extract all physicochemical features for one pose"""
    idx, row, protein_files, compounds_df = job
    
    job_id = row['job_id']
    protein = row['protein']
    ligand = row['ligand']
    
    log(f"  Processing {job_id}...")
    
    features = {'job_id': job_id, 'protein': protein, 'ligand': ligand}
    
    try:
        # Parse structures with charges
        ligand_file = Path(row['vina_output'])
        protein_file = protein_files[protein]
        
        lig_atoms = parse_pdbqt_with_charges(ligand_file, top_pose_only=True)
        prot_atoms = parse_pdbqt_with_charges(protein_file, top_pose_only=False)
        
        if not lig_atoms:
            log(f"    Warning: No ligand atoms for {job_id}")
            return None
        
        # Extract coordinates
        lig_coords = [(a[0], a[1], a[2]) for a in lig_atoms]
        prot_coords = [(a[0], a[1], a[2]) for a in prot_atoms]
        
        # Electrostatic energy
        features['electrostatic_energy'] = calculate_electrostatic_energy(
            lig_atoms, prot_atoms, dielectric=4.0, cutoff=10.0
        )
        
        # van der Waals energy
        features['vdw_energy'] = calculate_vdw_energy(
            lig_coords, prot_coords, cutoff=8.0
        )
        
        # Total estimated binding energy (simplified)
        if not np.isnan(features['electrostatic_energy']) and not np.isnan(features['vdw_energy']):
            features['total_energy_estimate'] = (
                features['electrostatic_energy'] + features['vdw_energy']
            )
        else:
            features['total_energy_estimate'] = np.nan
        
        # SASA calculations
        features['ligand_sasa'] = calculate_sasa_approximate(lig_coords, probe_radius=1.4)
        
        # Atom type counts
        features['num_polar_atoms'] = count_polar_atoms(lig_atoms)
        features['num_nonpolar_atoms'] = count_nonpolar_atoms(lig_atoms)
        
        if features['num_polar_atoms'] + features['num_nonpolar_atoms'] > 0:
            features['polar_fraction'] = features['num_polar_atoms'] / (
                features['num_polar_atoms'] + features['num_nonpolar_atoms']
            )
        else:
            features['polar_fraction'] = np.nan
        
        # Charge distribution
        charges = [a[3] for a in lig_atoms]
        features['total_charge'] = sum(charges)
        features['abs_total_charge'] = sum(abs(c) for c in charges)
        features['max_positive_charge'] = max(charges) if charges else 0.0
        features['max_negative_charge'] = min(charges) if charges else 0.0
        
        # Get SMILES for molecular descriptors
        smiles = None
        compound_row = compounds_df[compounds_df['DRUG_NAME'].str.lower() == ligand.lower()]
        
        if len(compound_row) > 0:
            smiles = compound_row.iloc[0]['SMILES']
        
        if smiles:
            mol_descriptors = calculate_molecular_descriptors(smiles)
            features.update(mol_descriptors)
        else:
            # Fill with NaN if SMILES not found
            features.update({
                'mol_weight': np.nan,
                'logp': np.nan,
                'num_hbd': np.nan,
                'num_hba': np.nan,
                'tpsa': np.nan,
                'num_rotatable_bonds': np.nan,
                'num_aromatic_rings': np.nan,
                'formal_charge': np.nan,
                'num_heavy_atoms': np.nan,
                'fraction_csp3': np.nan
            })
        
        # Binding affinity from docking
        features['vina_affinity'] = row['vina_affinity']
        features['smina_affinity'] = row['smina_affinity']
        features['best_affinity'] = row['best_affinity']
        
        # Ligand efficiency (LE = affinity / heavy atoms)
        if not np.isnan(features['num_heavy_atoms']) and features['num_heavy_atoms'] > 0:
            features['ligand_efficiency'] = features['best_affinity'] / features['num_heavy_atoms']
        else:
            features['ligand_efficiency'] = np.nan
        
        # Lipophilic efficiency (LLE = pIC50 - LogP)
        # Using binding affinity as proxy for pIC50
        if not np.isnan(features['logp']):
            # Convert kcal/mol to approximate pIC50 (rough estimate)
            pic50_estimate = -features['best_affinity'] / 1.36  # RT at 298K
            features['lipophilic_efficiency'] = pic50_estimate - features['logp']
        else:
            features['lipophilic_efficiency'] = np.nan
        
        log(f"    ✓ Extracted {len(features)-3} features")
        return features
    
    except Exception as e:
        log(f"    ✗ Error: {e}")
        import traceback
        log(f"    {traceback.format_exc()}")
        return None

if __name__ == '__main__':
    log("\n1. Loading high-confidence poses...")
    
    consensus_file = CONSENSUS_DIR / "high_confidence_poses.csv"
    if not consensus_file.exists():
        log(f"✗ ERROR: {consensus_file} not found!")
        exit(1)
    
    poses_df = pd.read_csv(consensus_file)
    log(f"✓ Loaded {len(poses_df)} poses")
    
    # Load compound SMILES for molecular descriptors
    log("\n2. Loading compound information...")
    compounds_file = PROJECT_ROOT / "DATA" / "PROCESSED" / "dockable_compounds.csv"
    
    if compounds_file.exists():
        compounds_df = pd.read_csv(compounds_file)
        log(f"✓ Loaded {len(compounds_df)} compounds")
    else:
        log("⚠️  Warning: Compound file not found, molecular descriptors will be NaN")
        compounds_df = pd.DataFrame()
    
    # Prepare protein files
    protein_files = {
        'EGFR': DOCKING_DIR / "egfr_prepared.pdbqt",
        'BRAF': DOCKING_DIR / "braf_prepared.pdbqt",
        'MEK1': DOCKING_DIR / "mek1_prepared.pdbqt"
    }
    
    for protein, pfile in protein_files.items():
        if not pfile.exists():
            log(f"✗ ERROR: {pfile} not found!")
            exit(1)
    
    log("\n3. Preparing extraction jobs...")
    jobs = [(idx, row, protein_files, compounds_df) for idx, row in poses_df.iterrows()]
    log(f"✓ {len(jobs)} jobs prepared")
    
    log("\n4. Extracting physicochemical features (parallelized)...")
    start_time = datetime.now()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_physicochemical_features_single_pose, jobs)
    
    results = [r for r in results if r is not None]
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n✓ Extracted {len(results)}/{len(jobs)} poses ({elapsed:.1f}s)")
    
    if not results:
        log("✗ ERROR: No features extracted!")
        exit(1)
    
    features_df = pd.DataFrame(results)
    output_file = PHYSICOCHEMICAL_DIR / "physicochemical_features.csv"
    features_df.to_csv(output_file, index=False)
    
    log(f"\n✓ Saved: {output_file}")
    log(f"✓ Shape: {features_df.shape}")
    log(f"\n{'='*70}")
    log("PHYSICOCHEMICAL EXTRACTION COMPLETE")
    log(f"{'='*70}")
