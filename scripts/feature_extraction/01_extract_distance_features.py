#!/usr/bin/env python3
"""
Distance-Based Feature Extraction for Molecular Docking Analysis

Extracts comprehensive distance metrics between ligand and protein atoms,
focusing on key binding site residues relevant to kinase inhibition.

Author: CASPID Research Team
Date: February 2026
"""

import os
import json
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
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Output directory
DISTANCE_FEATURES_DIR = FEATURES_DIR / "01_distance"
DISTANCE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = DISTANCE_FEATURES_DIR / f"distance_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("DISTANCE FEATURE EXTRACTION")
print("="*70)
log("Starting distance-based feature extraction")

# Key residues for kinase binding (from structural biology literature)
KEY_RESIDUES_EGFR = {
    'hinge': ['MET769', 'GLN767', 'LEU768'],
    'gatekeeper': ['THR790'],
    'dfg': ['ASP831', 'PHE832', 'GLY833'],
    'p_loop': ['GLY695', 'GLY696', 'GLY697'],
    'c_helix': ['GLU738', 'LYS721'],
    'a_loop': ['ASP831', 'LEU834', 'ARG841'],
    'selectivity': ['THR766', 'LEU764', 'VAL702']
}

KEY_RESIDUES_BRAF = {
    'hinge': ['CYS532', 'GLN530', 'ALA531'],
    'gatekeeper': ['THR529'],
    'dfg': ['ASP594', 'PHE595', 'GLY596'],
    'p_loop': ['GLY463', 'GLY464', 'GLY466'],
    'c_helix': ['GLU501', 'LYS483'],
    'v600e': ['GLU600'],
    'a_loop': ['ASP594', 'LEU597', 'TRP604'],
    'selectivity': ['THR529', 'ILE527', 'VAL471']
}

def get_key_residues(protein):
    if protein == 'EGFR':
        return KEY_RESIDUES_EGFR
    elif protein == 'BRAF':
        return KEY_RESIDUES_BRAF
    return {}

def parse_pdbqt_protein(pdbqt_file):
    """Parse protein PDBQT and extract coordinates by residue"""
    residues = {}
    try:
        with open(pdbqt_file) as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except:
                    continue
                
                res_id = f"{res_name}{res_num}"
                if res_id not in residues:
                    residues[res_id] = []
                residues[res_id].append((x, y, z))
        return residues
    except Exception as e:
        log(f"  Error parsing protein: {e}")
        return {}

def parse_pdbqt_ligand(pdbqt_file):
    """Parse ligand PDBQT and extract top pose coordinates"""
    coords = []
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
            elif line.startswith('ENDMDL'):
                break
            
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            if has_model_tag and not in_model:
                continue
            
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append((x, y, z))
            except:
                continue
        return coords
    except Exception as e:
        log(f"  Error parsing ligand: {e}")
        return []

def calculate_min_distance(lig_coords, res_coords):
    """Minimum distance between ligand and residue atoms"""
    if not lig_coords or not res_coords:
        return np.nan
    distances = distance.cdist(np.array(lig_coords), np.array(res_coords))
    return float(np.min(distances))

def calculate_mean_distance(lig_coords, res_coords):
    """Mean of closest distances per ligand atom"""
    if not lig_coords or not res_coords:
        return np.nan
    distances = distance.cdist(np.array(lig_coords), np.array(res_coords))
    return float(np.mean(np.min(distances, axis=1)))

def count_contacts(lig_coords, res_coords, cutoff=4.0):
    """Count atom pairs within cutoff distance"""
    if not lig_coords or not res_coords:
        return 0
    distances = distance.cdist(np.array(lig_coords), np.array(res_coords))
    return int(np.sum(distances < cutoff))

def extract_distance_features_single_pose(job):
    """Extract all distance features for one pose"""
    idx, row, protein_file = job
    
    job_id = row['job_id']
    protein = row['protein']
    ligand = row['ligand']
    ligand_file = Path(row['vina_output'])
    
    log(f"  Processing {job_id}...")
    
    features = {'job_id': job_id, 'protein': protein, 'ligand': ligand}
    
    try:
        protein_residues = parse_pdbqt_protein(protein_file)
        ligand_coords = parse_pdbqt_ligand(ligand_file)
        
        if not protein_residues or not ligand_coords:
            log(f"    Warning: Failed to parse {job_id}")
            return None
        
        key_residues = get_key_residues(protein)
        
        # Per-residue distance features
        for region, residue_list in key_residues.items():
            for res_id in residue_list:
                if res_id not in protein_residues:
                    features[f'{region}_{res_id}_min_dist'] = np.nan
                    features[f'{region}_{res_id}_mean_dist'] = np.nan
                    features[f'{region}_{res_id}_contacts'] = 0
                    continue
                
                res_coords = protein_residues[res_id]
                features[f'{region}_{res_id}_min_dist'] = calculate_min_distance(ligand_coords, res_coords)
                features[f'{region}_{res_id}_mean_dist'] = calculate_mean_distance(ligand_coords, res_coords)
                features[f'{region}_{res_id}_contacts'] = count_contacts(ligand_coords, res_coords)
        
        # Global distance metrics
        lig_centroid = np.mean(np.array(ligand_coords), axis=0)
        
        all_protein_coords = []
        for res_coords in protein_residues.values():
            all_protein_coords.extend(res_coords)
        
        prot_centroid = np.mean(np.array(all_protein_coords), axis=0)
        
        features['centroid_distance'] = float(np.linalg.norm(lig_centroid - prot_centroid))
        features['min_dist_to_protein'] = calculate_min_distance(ligand_coords, all_protein_coords)
        
        if len(ligand_coords) > 1:
            deviations = np.array(ligand_coords) - lig_centroid
            features['ligand_radius_gyration'] = float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))
        else:
            features['ligand_radius_gyration'] = np.nan
        
        log(f"    ✓ Extracted {len(features)-3} features")
        return features
    
    except Exception as e:
        log(f"    ✗ Error: {e}")
        return None

if __name__ == '__main__':
    log("\n1. Loading high-confidence poses...")
    
    consensus_file = CONSENSUS_DIR / "high_confidence_poses.csv"
    if not consensus_file.exists():
        log(f"✗ ERROR: {consensus_file} not found!")
        exit(1)
    
    poses_df = pd.read_csv(consensus_file)
    log(f"✓ Loaded {len(poses_df)} poses")
    
    protein_files = {
        'EGFR': DOCKING_DIR / "egfr_prepared.pdbqt",
        'BRAF': DOCKING_DIR / "braf_prepared.pdbqt"
    }
    
    for protein, pfile in protein_files.items():
        if not pfile.exists():
            log(f"✗ ERROR: {pfile} not found!")
            exit(1)
    
    log("\n2. Preparing extraction jobs...")
    jobs = [(idx, row, protein_files[row['protein']]) for idx, row in poses_df.iterrows()]
    log(f"✓ {len(jobs)} jobs prepared")
    
    log("\n3. Extracting features (parallelized)...")
    start_time = datetime.now()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_distance_features_single_pose, jobs)
    
    results = [r for r in results if r is not None]
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n✓ Extracted {len(results)}/{len(jobs)} poses ({elapsed:.1f}s)")
    
    if not results:
        log("✗ ERROR: No features extracted!")
        exit(1)
    
    features_df = pd.DataFrame(results)
    output_file = DISTANCE_FEATURES_DIR / "distance_features.csv"
    features_df.to_csv(output_file, index=False)
    
    log(f"\n✓ Saved: {output_file}")
    log(f"✓ Shape: {features_df.shape}")
    log(f"\n{'='*70}")
    log("DISTANCE EXTRACTION COMPLETE")
    log(f"{'='*70}")
