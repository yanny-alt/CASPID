#!/usr/bin/env python3
"""
Interaction Feature Extraction for Molecular Docking Analysis

Extracts detailed geometric descriptors for protein-ligand interactions:
hydrogen bonds, aromatic interactions, hydrophobic contacts, and other
non-covalent interactions critical for binding.

Author: CASPID Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
CONSENSUS_DIR = DOCKING_DIR / "CONSENSUS_RESULTS"
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES"

# Output directory
INTERACTION_DIR = FEATURES_DIR / "04_interactions"
INTERACTION_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = INTERACTION_DIR / f"interaction_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("INTERACTION FEATURE EXTRACTION")
print("="*70)
log("Starting interaction feature extraction")

def parse_pdbqt_detailed(pdbqt_file, top_pose_only=True):
    """
    Parse PDBQT with full atom details
    Returns list of dicts with atom properties
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
                atom = {
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'res_num': line[22:26].strip(),
                    'element': line[77:79].strip(),
                    'charge': 0.0
                }
                
                # Try to extract charge
                try:
                    atom['charge'] = float(line[70:76])
                except:
                    pass
                
                atoms.append(atom)
            except:
                continue
        
        return atoms
    except:
        return []

def is_donor_atom(atom):
    """Check if atom can be H-bond donor"""
    element = atom['element']
    # N, O with H can donate
    return element in ['N', 'O', 'NA', 'OA']

def is_acceptor_atom(atom):
    """Check if atom can be H-bond acceptor"""
    element = atom['element']
    # N, O, F can accept
    return element in ['N', 'O', 'F', 'NA', 'OA']

def is_aromatic_atom(atom):
    """Check if atom is part of aromatic system"""
    # Simplified: aromatic C in ring systems
    element = atom['element']
    atom_name = atom['atom_name']
    
    # Carbon in aromatic rings (PHE, TYR, TRP, HIS)
    if element == 'C' and atom['res_name'] in ['PHE', 'TYR', 'TRP', 'HIS']:
        return True
    
    # Also consider ligand aromatic carbons (named A in PDBQT)
    if element == 'A':
        return True
    
    return False

def is_hydrophobic_atom(atom):
    """Check if atom is hydrophobic"""
    element = atom['element']
    # C, S (excluding polar contexts)
    return element in ['C', 'A', 'S', 'SA']

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by p1-p2-p3
    Returns angle in degrees
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle) * 180 / np.pi
    return float(angle)

def find_hbond_interactions(lig_atoms, prot_atoms, distance_cutoff=3.5, angle_cutoff=120):
    """
    Find potential hydrogen bond interactions
    Returns list of (distance, angle) tuples for top interactions
    """
    hbonds = []
    
    # Ligand donor - Protein acceptor
    for l_atom in lig_atoms:
        if not is_donor_atom(l_atom):
            continue
        
        l_pos = [l_atom['x'], l_atom['y'], l_atom['z']]
        
        for p_atom in prot_atoms:
            if not is_acceptor_atom(p_atom):
                continue
            
            p_pos = [p_atom['x'], p_atom['y'], p_atom['z']]
            dist = np.linalg.norm(np.array(l_pos) - np.array(p_pos))
            
            if dist < distance_cutoff:
                # Simplified angle (just using donor-acceptor)
                hbonds.append({
                    'distance': dist,
                    'type': 'lig_donor'
                })
    
    # Protein donor - Ligand acceptor
    for p_atom in prot_atoms:
        if not is_donor_atom(p_atom):
            continue
        
        p_pos = [p_atom['x'], p_atom['y'], p_atom['z']]
        
        for l_atom in lig_atoms:
            if not is_acceptor_atom(l_atom):
                continue
            
            l_pos = [l_atom['x'], l_atom['y'], l_atom['z']]
            dist = np.linalg.norm(np.array(p_pos) - np.array(l_pos))
            
            if dist < distance_cutoff:
                hbonds.append({
                    'distance': dist,
                    'type': 'prot_donor'
                })
    
    return hbonds

def find_aromatic_interactions(lig_atoms, prot_atoms, distance_cutoff=6.0):
    """
    Find aromatic-aromatic interactions (pi-pi stacking)
    Returns list of distances for aromatic atom pairs
    """
    aromatic_interactions = []
    
    # Get aromatic atoms
    lig_aromatic = [a for a in lig_atoms if is_aromatic_atom(a)]
    prot_aromatic = [a for a in prot_atoms if is_aromatic_atom(a)]
    
    for l_atom in lig_aromatic:
        l_pos = [l_atom['x'], l_atom['y'], l_atom['z']]
        
        for p_atom in prot_aromatic:
            p_pos = [p_atom['x'], p_atom['y'], p_atom['z']]
            dist = np.linalg.norm(np.array(l_pos) - np.array(p_pos))
            
            if dist < distance_cutoff:
                aromatic_interactions.append(dist)
    
    return aromatic_interactions

def find_hydrophobic_contacts(lig_atoms, prot_atoms, distance_cutoff=4.5):
    """
    Find hydrophobic contacts (C-C interactions)
    Returns count and mean distance
    """
    contacts = []
    
    lig_hydrophobic = [a for a in lig_atoms if is_hydrophobic_atom(a)]
    prot_hydrophobic = [a for a in prot_atoms if is_hydrophobic_atom(a)]
    
    for l_atom in lig_hydrophobic:
        l_pos = [l_atom['x'], l_atom['y'], l_atom['z']]
        
        for p_atom in prot_hydrophobic:
            p_pos = [p_atom['x'], p_atom['y'], p_atom['z']]
            dist = np.linalg.norm(np.array(l_pos) - np.array(p_pos))
            
            if dist < distance_cutoff:
                contacts.append(dist)
    
    return contacts

def find_salt_bridges(lig_atoms, prot_atoms, distance_cutoff=4.0):
    """
    Find potential salt bridges (charge-charge interactions)
    Returns list of distances
    """
    salt_bridges = []
    
    for l_atom in lig_atoms:
        if abs(l_atom['charge']) < 0.3:  # Skip neutral
            continue
        
        l_pos = [l_atom['x'], l_atom['y'], l_atom['z']]
        l_charge = l_atom['charge']
        
        for p_atom in prot_atoms:
            if abs(p_atom['charge']) < 0.3:
                continue
            
            p_charge = p_atom['charge']
            
            # Opposite charges attract
            if l_charge * p_charge < 0:
                p_pos = [p_atom['x'], p_atom['y'], p_atom['z']]
                dist = np.linalg.norm(np.array(l_pos) - np.array(p_pos))
                
                if dist < distance_cutoff:
                    salt_bridges.append(dist)
    
    return salt_bridges

def extract_interaction_features_single_pose(job):
    """Extract all interaction features for one pose"""
    idx, row, protein_files = job
    
    job_id = row['job_id']
    protein = row['protein']
    ligand = row['ligand']
    
    log(f"  Processing {job_id}...")
    
    features = {'job_id': job_id, 'protein': protein, 'ligand': ligand}
    
    try:
        # Parse structures
        ligand_file = Path(row['vina_output'])
        protein_file = protein_files[protein]
        
        lig_atoms = parse_pdbqt_detailed(ligand_file, top_pose_only=True)
        prot_atoms = parse_pdbqt_detailed(protein_file, top_pose_only=False)
        
        if not lig_atoms:
            log(f"    Warning: No ligand atoms for {job_id}")
            return None
        
        # Hydrogen bond features
        hbonds = find_hbond_interactions(lig_atoms, prot_atoms)
        
        features['num_hbonds'] = len(hbonds)
        features['num_hbonds_lig_donor'] = sum(1 for h in hbonds if h['type'] == 'lig_donor')
        features['num_hbonds_prot_donor'] = sum(1 for h in hbonds if h['type'] == 'prot_donor')
        
        if hbonds:
            hbond_distances = [h['distance'] for h in hbonds]
            features['hbond_min_distance'] = min(hbond_distances)
            features['hbond_mean_distance'] = np.mean(hbond_distances)
            features['hbond_max_distance'] = max(hbond_distances)
        else:
            features['hbond_min_distance'] = np.nan
            features['hbond_mean_distance'] = np.nan
            features['hbond_max_distance'] = np.nan
        
        # Aromatic interactions
        aromatic = find_aromatic_interactions(lig_atoms, prot_atoms)
        
        features['num_aromatic_interactions'] = len(aromatic)
        
        if aromatic:
            features['aromatic_min_distance'] = min(aromatic)
            features['aromatic_mean_distance'] = np.mean(aromatic)
        else:
            features['aromatic_min_distance'] = np.nan
            features['aromatic_mean_distance'] = np.nan
        
        # Hydrophobic contacts
        hydrophobic = find_hydrophobic_contacts(lig_atoms, prot_atoms)
        
        features['num_hydrophobic_contacts'] = len(hydrophobic)
        
        if hydrophobic:
            features['hydrophobic_min_distance'] = min(hydrophobic)
            features['hydrophobic_mean_distance'] = np.mean(hydrophobic)
            features['hydrophobic_surface_area'] = len(hydrophobic) * 15.0  # Approximate
        else:
            features['hydrophobic_min_distance'] = np.nan
            features['hydrophobic_mean_distance'] = np.nan
            features['hydrophobic_surface_area'] = 0.0
        
        # Salt bridges
        salt_bridges = find_salt_bridges(lig_atoms, prot_atoms)
        
        features['num_salt_bridges'] = len(salt_bridges)
        
        if salt_bridges:
            features['salt_bridge_min_distance'] = min(salt_bridges)
            features['salt_bridge_mean_distance'] = np.mean(salt_bridges)
        else:
            features['salt_bridge_min_distance'] = np.nan
            features['salt_bridge_mean_distance'] = np.nan
        
        # Halogen atoms (F, Cl, Br)
        halogen_elements = ['F', 'CL', 'BR']
        num_halogens = sum(1 for a in lig_atoms if a['element'] in halogen_elements)
        features['num_halogen_atoms'] = num_halogens
        
        # Overall interaction profile
        total_interactions = (
            features['num_hbonds'] + 
            features['num_aromatic_interactions'] + 
            features['num_hydrophobic_contacts'] +
            features['num_salt_bridges']
        )
        features['total_interactions'] = total_interactions
        
        # Interaction density (interactions per ligand atom)
        if len(lig_atoms) > 0:
            features['interaction_density'] = total_interactions / len(lig_atoms)
        else:
            features['interaction_density'] = 0.0
        
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
    
    # Prepare protein files
    protein_files = {
        'EGFR': DOCKING_DIR / "egfr_prepared.pdbqt",
        'BRAF': DOCKING_DIR / "braf_prepared.pdbqt"
    }
    
    for protein, pfile in protein_files.items():
        if not pfile.exists():
            log(f"✗ ERROR: {pfile} not found!")
            exit(1)
    
    log("\n2. Preparing extraction jobs...")
    jobs = [(idx, row, protein_files) for idx, row in poses_df.iterrows()]
    log(f"✓ {len(jobs)} jobs prepared")
    
    log("\n3. Extracting interaction features (parallelized)...")
    start_time = datetime.now()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_interaction_features_single_pose, jobs)
    
    results = [r for r in results if r is not None]
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n✓ Extracted {len(results)}/{len(jobs)} poses ({elapsed:.1f}s)")
    
    if not results:
        log("✗ ERROR: No features extracted!")
        exit(1)
    
    features_df = pd.DataFrame(results)
    output_file = INTERACTION_DIR / "interaction_features.csv"
    features_df.to_csv(output_file, index=False)
    
    log(f"\n✓ Saved: {output_file}")
    log(f"✓ Shape: {features_df.shape}")
    log(f"\n{'='*70}")
    log("INTERACTION EXTRACTION COMPLETE")
    log(f"{'='*70}")
