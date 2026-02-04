#!/usr/bin/env python3
"""
Pharmacophore Feature Extraction for Molecular Docking Analysis

Extracts pharmacophore-based features including donor/acceptor patterns,
hydrophobic regions, aromatic features, and 3D spatial arrangements
that define binding modes.

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
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
CONSENSUS_DIR = DOCKING_DIR / "CONSENSUS_RESULTS"
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES"
DATA_PROCESSED = PROJECT_ROOT / "DATA" / "PROCESSED"

# Output directory
PHARMACOPHORE_DIR = FEATURES_DIR / "05_pharmacophore"
PHARMACOPHORE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PHARMACOPHORE_DIR / f"pharmacophore_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("PHARMACOPHORE FEATURE EXTRACTION")
print("="*70)
log("Starting pharmacophore feature extraction")

def parse_pdbqt_coords(pdbqt_file, top_pose_only=True):
    """Parse PDBQT and return coordinates"""
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
            elif line.startswith('ENDMDL') and top_pose_only:
                break
            
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            if has_model_tag and not in_model:
                continue
            
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
            except:
                continue
        
        return np.array(coords) if coords else np.array([])
    except:
        return np.array([])

def get_pharmacophore_features_from_smiles(smiles):
    """
    Extract pharmacophore features from SMILES using RDKit
    Returns counts of different pharmacophore types
    """
    features = {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                'num_hbd_pharmacophore': np.nan,
                'num_hba_pharmacophore': np.nan,
                'num_aromatic_pharmacophore': np.nan,
                'num_hydrophobic_pharmacophore': np.nan,
                'num_positive_ionizable': 0,
                'num_negative_ionizable': 0
            }
        
        # H-bond donors and acceptors
        features['num_hbd_pharmacophore'] = Descriptors.NumHDonors(mol)
        features['num_hba_pharmacophore'] = Descriptors.NumHAcceptors(mol)
        
        # Aromatic rings
        features['num_aromatic_pharmacophore'] = Descriptors.NumAromaticRings(mol)
        
        # Hydrophobic centers (aliphatic carbons)
        num_aliphatic = sum(1 for atom in mol.GetAtoms() 
                           if atom.GetSymbol() == 'C' and not atom.GetIsAromatic())
        features['num_hydrophobic_pharmacophore'] = num_aliphatic
        
        # Ionizable groups (basic and acidic)
        # Basic nitrogens
        num_basic = sum(1 for atom in mol.GetAtoms()
                       if atom.GetSymbol() == 'N' and atom.GetTotalDegree() < 4)
        features['num_positive_ionizable'] = num_basic
        
        # Acidic oxygens (carboxylic, phenolic)
        num_acidic = sum(1 for atom in mol.GetAtoms()
                        if atom.GetSymbol() == 'O' and 
                        any(n.GetSymbol() == 'C' for n in atom.GetNeighbors()))
        features['num_negative_ionizable'] = num_acidic
        
        return features
    
    except Exception as e:
        return {
            'num_hbd_pharmacophore': np.nan,
            'num_hba_pharmacophore': np.nan,
            'num_aromatic_pharmacophore': np.nan,
            'num_hydrophobic_pharmacophore': np.nan,
            'num_positive_ionizable': 0,
            'num_negative_ionizable': 0
        }

def calculate_pharmacophore_distances(coords):
    """
    Calculate key 3D distances in pharmacophore
    Uses principal points in the molecule
    """
    if len(coords) < 3:
        return {
            'pharmacophore_span': np.nan,
            'pharmacophore_radius': np.nan,
            'pharmacophore_triangular_area': np.nan
        }
    
    features = {}
    
    # Maximum span (largest pairwise distance)
    dists = distance.pdist(coords)
    features['pharmacophore_span'] = float(np.max(dists))
    
    # Average radius from centroid
    centroid = np.mean(coords, axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    features['pharmacophore_radius'] = float(np.mean(radii))
    
    # Triangular arrangement (take 3 most distant points)
    if len(coords) >= 3:
        # Find 3 most spread points
        idx1 = 0
        idx2 = np.argmax(np.linalg.norm(coords - coords[0], axis=1))
        
        remaining = [i for i in range(len(coords)) if i not in [idx1, idx2]]
        if remaining:
            dists_to_both = [
                np.linalg.norm(coords[i] - coords[idx1]) + 
                np.linalg.norm(coords[i] - coords[idx2])
                for i in remaining
            ]
            idx3 = remaining[np.argmax(dists_to_both)]
            
            # Calculate triangle area (Heron's formula)
            p1, p2, p3 = coords[idx1], coords[idx2], coords[idx3]
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            s = (a + b + c) / 2
            
            if s > a and s > b and s > c:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                features['pharmacophore_triangular_area'] = float(area)
            else:
                features['pharmacophore_triangular_area'] = 0.0
        else:
            features['pharmacophore_triangular_area'] = np.nan
    else:
        features['pharmacophore_triangular_area'] = np.nan
    
    return features

def calculate_spatial_distribution(coords):
    """
    Calculate how atoms are distributed in 3D space
    """
    if len(coords) < 2:
        return {
            'spatial_distribution_x': np.nan,
            'spatial_distribution_y': np.nan,
            'spatial_distribution_z': np.nan,
            'spatial_volume_ratio': np.nan
        }
    
    features = {}
    
    # Standard deviation along each axis
    features['spatial_distribution_x'] = float(np.std(coords[:, 0]))
    features['spatial_distribution_y'] = float(np.std(coords[:, 1]))
    features['spatial_distribution_z'] = float(np.std(coords[:, 2]))
    
    # Volume ratio (how 3D vs flat)
    ranges = np.ptp(coords, axis=0)  # Range along each axis
    if np.min(ranges) > 0:
        features['spatial_volume_ratio'] = float(np.prod(ranges))
    else:
        features['spatial_volume_ratio'] = 0.0
    
    return features

def calculate_compactness(coords):
    """
    Measure how compact the molecule is
    Ratio of actual span to ideal sphere
    """
    if len(coords) < 2:
        return np.nan
    
    # Actual maximum distance
    max_dist = float(np.max(distance.pdist(coords)))
    
    # Volume estimate (convex hull would be better but simpler here)
    centroid = np.mean(coords, axis=0)
    avg_radius = np.mean(np.linalg.norm(coords - centroid, axis=1))
    
    if max_dist > 0:
        compactness = (2 * avg_radius) / max_dist
        return float(compactness)
    
    return np.nan

def extract_pharmacophore_features_single_pose(job):
    """Extract all pharmacophore features for one pose"""
    idx, row, compounds_df = job
    
    job_id = row['job_id']
    protein = row['protein']
    ligand = row['ligand']
    
    log(f"  Processing {job_id}...")
    
    features = {'job_id': job_id, 'protein': protein, 'ligand': ligand}
    
    try:
        # Get ligand coordinates
        ligand_file = Path(row['vina_output'])
        lig_coords = parse_pdbqt_coords(ligand_file, top_pose_only=True)
        
        if len(lig_coords) == 0:
            log(f"    Warning: No coordinates for {job_id}")
            return None
        
        # Get SMILES for pharmacophore analysis
        smiles = None
        compound_row = compounds_df[compounds_df['DRUG_NAME'].str.lower() == ligand.lower()]
        
        if len(compound_row) > 0:
            smiles = compound_row.iloc[0]['SMILES']
        
        # Pharmacophore features from SMILES
        if smiles:
            pharm_features = get_pharmacophore_features_from_smiles(smiles)
            features.update(pharm_features)
        else:
            features.update({
                'num_hbd_pharmacophore': np.nan,
                'num_hba_pharmacophore': np.nan,
                'num_aromatic_pharmacophore': np.nan,
                'num_hydrophobic_pharmacophore': np.nan,
                'num_positive_ionizable': 0,
                'num_negative_ionizable': 0
            })
        
        # 3D pharmacophore geometry
        pharm_distances = calculate_pharmacophore_distances(lig_coords)
        features.update(pharm_distances)
        
        # Spatial distribution
        spatial_features = calculate_spatial_distribution(lig_coords)
        features.update(spatial_features)
        
        # Compactness
        features['molecular_compactness'] = calculate_compactness(lig_coords)
        
        # Atom density (atoms per unit volume)
        if 'pharmacophore_span' in features and not np.isnan(features['pharmacophore_span']):
            if features['pharmacophore_span'] > 0:
                volume_estimate = (4/3) * np.pi * (features['pharmacophore_span']/2)**3
                features['atom_density'] = len(lig_coords) / volume_estimate
            else:
                features['atom_density'] = np.nan
        else:
            features['atom_density'] = np.nan
        
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
    
    # Load compound SMILES
    log("\n2. Loading compound information...")
    compounds_file = DATA_PROCESSED / "dockable_compounds.csv"
    
    if compounds_file.exists():
        compounds_df = pd.read_csv(compounds_file)
        log(f"✓ Loaded {len(compounds_df)} compounds")
    else:
        log("⚠️  Warning: Compound file not found, some features will be NaN")
        compounds_df = pd.DataFrame()
    
    log("\n3. Preparing extraction jobs...")
    jobs = [(idx, row, compounds_df) for idx, row in poses_df.iterrows()]
    log(f"✓ {len(jobs)} jobs prepared")
    
    log("\n4. Extracting pharmacophore features (parallelized)...")
    start_time = datetime.now()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_pharmacophore_features_single_pose, jobs)
    
    results = [r for r in results if r is not None]
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n✓ Extracted {len(results)}/{len(jobs)} poses ({elapsed:.1f}s)")
    
    if not results:
        log("✗ ERROR: No features extracted!")
        exit(1)
    
    features_df = pd.DataFrame(results)
    output_file = PHARMACOPHORE_DIR / "pharmacophore_features.csv"
    features_df.to_csv(output_file, index=False)
    
    log(f"\n✓ Saved: {output_file}")
    log(f"✓ Shape: {features_df.shape}")
    log(f"\n{'='*70}")
    log("PHARMACOPHORE EXTRACTION COMPLETE")
    log(f"{'='*70}")
