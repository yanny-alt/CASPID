#!/usr/bin/env python3
"""
Geometric Feature Extraction for Molecular Docking Analysis

Extracts shape complementarity, ligand conformational features,
orientation descriptors, and spatial metrics from docking poses.

Author: CASPID Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.spatial import ConvexHull, distance
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
CONSENSUS_DIR = DOCKING_DIR / "CONSENSUS_RESULTS"
FEATURES_DIR = PROJECT_ROOT / "DATA" / "FEATURES"

# Output directory
GEOMETRIC_FEATURES_DIR = FEATURES_DIR / "02_geometric"
GEOMETRIC_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = GEOMETRIC_FEATURES_DIR / f"geometric_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("GEOMETRIC FEATURE EXTRACTION")
print("="*70)
log("Starting geometric feature extraction")

def parse_pdbqt_coords(pdbqt_file, top_pose_only=True):
    """Parse PDBQT file and return coordinates"""
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

def calculate_radius_of_gyration(coords):
    """Calculate radius of gyration (spatial extent measure)"""
    if len(coords) < 2:
        return np.nan
    centroid = np.mean(coords, axis=0)
    deviations = coords - centroid
    rg = np.sqrt(np.mean(np.sum(deviations**2, axis=1)))
    return float(rg)

def calculate_asphericity(coords):
    """
    Measure of deviation from spherical shape
    0 = perfect sphere, higher = more elongated
    """
    if len(coords) < 3:
        return np.nan
    
    # Center coordinates
    centered = coords - np.mean(coords, axis=0)
    
    # Calculate moment of inertia tensor
    I = np.dot(centered.T, centered) / len(coords)
    
    # Eigenvalues of inertia tensor
    eigenvalues = np.linalg.eigvalsh(I)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    if eigenvalues[0] == 0:
        return np.nan
    
    # Asphericity calculation
    asph = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
    asph = asph / eigenvalues.sum()
    
    return float(asph)

def calculate_eccentricity(coords):
    """
    Measure of elongation
    0 = spherical, 1 = highly elongated
    """
    if len(coords) < 3:
        return np.nan
    
    centered = coords - np.mean(coords, axis=0)
    I = np.dot(centered.T, centered) / len(coords)
    eigenvalues = np.linalg.eigvalsh(I)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    if eigenvalues[0] == 0:
        return np.nan
    
    ecc = 1.0 - (eigenvalues[2] / eigenvalues[0])
    return float(ecc)

def calculate_volume_approximate(coords):
    """Approximate molecular volume using convex hull"""
    if len(coords) < 4:
        return np.nan
    
    try:
        hull = ConvexHull(coords)
        return float(hull.volume)
    except:
        return np.nan

def calculate_surface_area_approximate(coords):
    """Approximate surface area using convex hull"""
    if len(coords) < 4:
        return np.nan
    
    try:
        hull = ConvexHull(coords)
        return float(hull.area)
    except:
        return np.nan

def calculate_max_dimension(coords):
    """Maximum pairwise distance (molecular span)"""
    if len(coords) < 2:
        return np.nan
    
    dists = distance.pdist(coords, metric='euclidean')
    return float(np.max(dists))

def calculate_planarity_index(coords):
    """
    Measure of molecular planarity
    Based on variance in third principal component
    """
    if len(coords) < 3:
        return np.nan
    
    try:
        pca = PCA(n_components=3)
        pca.fit(coords)
        
        # Ratio of smallest to largest eigenvalue
        # Low ratio = planar, high ratio = 3D
        explained_var = pca.explained_variance_
        if explained_var[0] == 0:
            return np.nan
        
        planarity = 1.0 - (explained_var[2] / explained_var[0])
        return float(planarity)
    except:
        return np.nan

def calculate_pca_features(coords):
    """Calculate PCA-based orientation features"""
    if len(coords) < 3:
        return [np.nan] * 6
    
    try:
        pca = PCA(n_components=3)
        pca.fit(coords)
        
        # Explained variance ratios (shape descriptor)
        var_ratios = pca.explained_variance_ratio_
        
        # Principal axes (for orientation)
        axes = pca.components_
        
        # Return variance ratios and first principal axis direction
        return [
            float(var_ratios[0]),
            float(var_ratios[1]),
            float(var_ratios[2]),
            float(axes[0][0]),  # First PC direction x
            float(axes[0][1]),  # First PC direction y
            float(axes[0][2])   # First PC direction z
        ]
    except:
        return [np.nan] * 6

def calculate_shape_complementarity(lig_coords, prot_coords, distance_cutoff=5.0):
    """
    Approximate shape complementarity score
    Based on local surface matching within cutoff distance
    """
    if len(lig_coords) < 3 or len(prot_coords) < 3:
        return np.nan
    
    try:
        # Find protein atoms near ligand
        dists = distance.cdist(lig_coords, prot_coords)
        close_pairs = np.sum(dists < distance_cutoff)
        
        # Normalize by ligand size
        if len(lig_coords) == 0:
            return np.nan
        
        sc_score = close_pairs / len(lig_coords)
        return float(sc_score)
    except:
        return np.nan

def calculate_buried_surface_area(lig_coords, prot_coords, probe_radius=1.4):
    """
    Estimate buried surface area upon binding
    Simplified calculation based on contact atoms
    """
    if len(lig_coords) < 3 or len(prot_coords) < 3:
        return np.nan
    
    try:
        # Count ligand atoms in contact with protein
        dists = distance.cdist(lig_coords, prot_coords)
        contact_atoms = np.sum(np.any(dists < probe_radius * 2, axis=1))
        
        # Approximate BSA (rough estimate)
        atom_sa = 4 * np.pi * (probe_radius ** 2)
        bsa = contact_atoms * atom_sa
        
        return float(bsa)
    except:
        return np.nan

def extract_geometric_features_single_pose(job):
    """Extract all geometric features for one pose"""
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
        
        lig_coords = parse_pdbqt_coords(ligand_file, top_pose_only=True)
        prot_coords = parse_pdbqt_coords(protein_file, top_pose_only=False)
        
        if len(lig_coords) == 0:
            log(f"    Warning: No ligand coordinates for {job_id}")
            return None
        
        # Shape and volume features
        features['ligand_volume'] = calculate_volume_approximate(lig_coords)
        features['ligand_surface_area'] = calculate_surface_area_approximate(lig_coords)
        features['ligand_max_dimension'] = calculate_max_dimension(lig_coords)
        
        # Spatial extent features
        features['ligand_radius_gyration'] = calculate_radius_of_gyration(lig_coords)
        features['ligand_asphericity'] = calculate_asphericity(lig_coords)
        features['ligand_eccentricity'] = calculate_eccentricity(lig_coords)
        features['ligand_planarity'] = calculate_planarity_index(lig_coords)
        
        # PCA features
        pca_features = calculate_pca_features(lig_coords)
        features['pca_variance_pc1'] = pca_features[0]
        features['pca_variance_pc2'] = pca_features[1]
        features['pca_variance_pc3'] = pca_features[2]
        features['pca_axis1_x'] = pca_features[3]
        features['pca_axis1_y'] = pca_features[4]
        features['pca_axis1_z'] = pca_features[5]
        
        # Shape complementarity with protein
        if len(prot_coords) > 0:
            features['shape_complementarity'] = calculate_shape_complementarity(
                lig_coords, prot_coords, distance_cutoff=5.0
            )
            features['buried_surface_area'] = calculate_buried_surface_area(
                lig_coords, prot_coords, probe_radius=1.4
            )
        else:
            features['shape_complementarity'] = np.nan
            features['buried_surface_area'] = np.nan
        
        # Ligand centroid position
        centroid = np.mean(lig_coords, axis=0)
        features['ligand_centroid_x'] = float(centroid[0])
        features['ligand_centroid_y'] = float(centroid[1])
        features['ligand_centroid_z'] = float(centroid[2])
        
        # Number of atoms (size proxy)
        features['ligand_num_atoms'] = len(lig_coords)
        
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
    
    log("\n3. Extracting geometric features (parallelized)...")
    start_time = datetime.now()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_geometric_features_single_pose, jobs)
    
    results = [r for r in results if r is not None]
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n✓ Extracted {len(results)}/{len(jobs)} poses ({elapsed:.1f}s)")
    
    if not results:
        log("✗ ERROR: No features extracted!")
        exit(1)
    
    features_df = pd.DataFrame(results)
    output_file = GEOMETRIC_FEATURES_DIR / "geometric_features.csv"
    features_df.to_csv(output_file, index=False)
    
    log(f"\n✓ Saved: {output_file}")
    log(f"✓ Shape: {features_df.shape}")
    log(f"\n{'='*70}")
    log("GEOMETRIC EXTRACTION COMPLETE")
    log(f"{'='*70}")
