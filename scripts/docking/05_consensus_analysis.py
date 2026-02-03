#!/usr/bin/env python3
"""
CRITICAL: Consensus Docking Analysis
PRODUCTION VERSION - RMSD comparison and confidence assignment

Compares Vina vs SMINA docking poses
Assigns confidence levels based on agreement
Validates against expected kinase docking performance

This is CRITICAL for paper credibility - maximum rigor applied

Author: CASPID Research Team
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"

# Input directories
VINA_DIR = DOCKING_DIR / "VINA_RESULTS"
SMINA_DIR = DOCKING_DIR / "SMINA_RESULTS"

# Output directory
CONSENSUS_DIR = DOCKING_DIR / "CONSENSUS_RESULTS"
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = CONSENSUS_DIR / f"consensus_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("CONSENSUS DOCKING ANALYSIS - VINA vs SMINA")
print("="*70)
log("Starting consensus analysis pipeline")

# ============================================
# LOAD DOCKING RESULTS
# ============================================

log("\n1. Loading docking results...")

# Load Vina results
vina_results_file = VINA_DIR / "vina_successful_dockings.csv"
if not vina_results_file.exists():
    log(f"✗ ERROR: Vina results not found at {vina_results_file}")
    exit(1)

vina_df = pd.read_csv(vina_results_file)
log(f"✓ Vina: {len(vina_df)} successful dockings")

# Load SMINA results
smina_results_file = SMINA_DIR / "smina_successful_dockings.csv"
if not smina_results_file.exists():
    log(f"✗ ERROR: SMINA results not found at {smina_results_file}")
    exit(1)

smina_df = pd.read_csv(smina_results_file)
log(f"✓ SMINA: {len(smina_df)} successful dockings")

# ============================================
# IDENTIFY COMMON DOCKINGS
# ============================================

log("\n2. Identifying common dockings...")

# Create job IDs
vina_df['job_id'] = vina_df['protein'] + '_' + vina_df['ligand']
smina_df['job_id'] = smina_df['protein'] + '_' + smina_df['ligand']

# Find common jobs
common_jobs = set(vina_df['job_id']) & set(smina_df['job_id'])

log(f"✓ Common dockings: {len(common_jobs)}")
log(f"  Vina only: {len(set(vina_df['job_id']) - common_jobs)}")
log(f"  SMINA only: {len(set(smina_df['job_id']) - common_jobs)}")

if len(common_jobs) == 0:
    log("✗ ERROR: No common dockings found!")
    exit(1)

# ============================================
# RMSD CALCULATION FUNCTION
# ============================================

def calculate_rmsd(mol1, mol2):
    """
    Calculate RMSD between two molecules
    Returns None if calculation fails
    """
    try:
        # Align molecules
        rmsd = rdMolAlign.GetBestRMS(mol1, mol2)
        return rmsd
    except Exception as e:
        return None

def extract_top_pose(pdbqt_file):
    """
    Extract top (best) pose from PDBQT file
    Returns RDKit molecule or None
    """
    try:
        # Read PDBQT file
        with open(pdbqt_file) as f:
            lines = f.readlines()
        
        # Extract first MODEL (best pose)
        model_lines = []
        in_model = False
        
        for line in lines:
            if line.startswith('MODEL'):
                in_model = True
                continue
            elif line.startswith('ENDMDL'):
                break
            elif in_model and (line.startswith('ATOM') or line.startswith('HETATM')):
                model_lines.append(line)
        
        # If no MODEL tag, take all ATOM/HETATM lines (single pose file)
        if not model_lines:
            model_lines = [l for l in lines if l.startswith('ATOM') or l.startswith('HETATM')]
        
        if not model_lines:
            return None
        
        # Convert to PDB format (remove charges from PDBQT)
        pdb_lines = []
        for line in model_lines:
            # PDBQT has extra columns at end - keep only standard PDB columns
            pdb_line = line[:66] + '\n'
            pdb_lines.append(pdb_line)
        
        # Create temp PDB string
        pdb_string = ''.join(pdb_lines)
        
        # Parse with RDKit
        mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False, sanitize=False)
        
        return mol
    
    except Exception as e:
        return None

# ============================================
# CALCULATE RMSD FOR ALL COMMON JOBS
# ============================================

log("\n3. Calculating RMSD between Vina and SMINA poses...")
log("   (This may take a few minutes)")

consensus_results = []
failed_rmsd = []

for job_id in sorted(common_jobs):
    # Get Vina and SMINA data
    vina_row = vina_df[vina_df['job_id'] == job_id].iloc[0]
    smina_row = smina_df[smina_df['job_id'] == job_id].iloc[0]
    
    protein = vina_row['protein']
    ligand = vina_row['ligand']
    
    # Get output files
    vina_output = Path(vina_row['output'])
    smina_output = Path(smina_row['output'])
    
    # Check files exist
    if not vina_output.exists() or not smina_output.exists():
        log(f"  ⚠️  {job_id}: Output files missing")
        failed_rmsd.append(job_id)
        continue
    
    # Extract top poses
    vina_mol = extract_top_pose(vina_output)
    smina_mol = extract_top_pose(smina_output)
    
    if vina_mol is None or smina_mol is None:
        log(f"  ⚠️  {job_id}: Could not extract poses")
        failed_rmsd.append(job_id)
        continue
    
    # Calculate RMSD
    rmsd = calculate_rmsd(vina_mol, smina_mol)
    
    if rmsd is None:
        log(f"  ⚠️  {job_id}: RMSD calculation failed")
        failed_rmsd.append(job_id)
        continue
    
    # Assign confidence level
    if rmsd < 2.0:
        confidence = "HIGH"
        confidence_score = 1.0
    elif rmsd < 3.0:
        confidence = "MEDIUM"
        confidence_score = 0.5
    else:
        confidence = "LOW"
        confidence_score = 0.0
    
    # Get binding affinities
    vina_affinity = vina_row['affinity_kcal_mol']
    smina_affinity = smina_row['affinity_kcal_mol']
    
    # Calculate affinity agreement
    affinity_diff = abs(vina_affinity - smina_affinity)
    
    # Store results
    result = {
        'protein': protein,
        'ligand': ligand,
        'job_id': job_id,
        'rmsd_angstrom': round(rmsd, 3),
        'confidence': confidence,
        'confidence_score': confidence_score,
        'vina_affinity': round(vina_affinity, 2),
        'smina_affinity': round(smina_affinity, 2),
        'affinity_diff': round(affinity_diff, 2),
        'best_affinity': round(min(vina_affinity, smina_affinity), 2),
        'vina_output': str(vina_output),
        'smina_output': str(smina_output)
    }
    
    consensus_results.append(result)

log(f"\n✓ RMSD calculated for {len(consensus_results)}/{len(common_jobs)} jobs")

if failed_rmsd:
    log(f"⚠️  Failed RMSD: {len(failed_rmsd)} jobs")
    for job in failed_rmsd[:5]:
        log(f"   - {job}")
    if len(failed_rmsd) > 5:
        log(f"   ... and {len(failed_rmsd)-5} more")

# ============================================
# ANALYZE CONSENSUS RESULTS
# ============================================

log("\n4. Analyzing consensus results...")

consensus_df = pd.DataFrame(consensus_results)

# Overall statistics
high_conf = len(consensus_df[consensus_df['confidence'] == 'HIGH'])
medium_conf = len(consensus_df[consensus_df['confidence'] == 'MEDIUM'])
low_conf = len(consensus_df[consensus_df['confidence'] == 'LOW'])

log(f"\n   Confidence Distribution:")
log(f"   ✓ HIGH (RMSD < 2.0Å):   {high_conf}/{len(consensus_df)} ({high_conf/len(consensus_df)*100:.1f}%)")
log(f"   ⚠ MEDIUM (2-3Å):        {medium_conf}/{len(consensus_df)} ({medium_conf/len(consensus_df)*100:.1f}%)")
log(f"   ✗ LOW (RMSD > 3.0Å):    {low_conf}/{len(consensus_df)} ({low_conf/len(consensus_df)*100:.1f}%)")

# RMSD statistics
log(f"\n   RMSD Statistics:")
log(f"   Mean:   {consensus_df['rmsd_angstrom'].mean():.2f} Å")
log(f"   Median: {consensus_df['rmsd_angstrom'].median():.2f} Å")
log(f"   Min:    {consensus_df['rmsd_angstrom'].min():.2f} Å")
log(f"   Max:    {consensus_df['rmsd_angstrom'].max():.2f} Å")

# Affinity agreement
log(f"\n   Affinity Agreement:")
log(f"   Mean difference: {consensus_df['affinity_diff'].mean():.2f} kcal/mol")
log(f"   Median difference: {consensus_df['affinity_diff'].median():.2f} kcal/mol")

# Per-protein analysis
log(f"\n   Per-Protein Analysis:")
for protein in consensus_df['protein'].unique():
    protein_df = consensus_df[consensus_df['protein'] == protein]
    high_prot = len(protein_df[protein_df['confidence'] == 'HIGH'])
    log(f"   {protein}: {high_prot}/{len(protein_df)} HIGH confidence ({high_prot/len(protein_df)*100:.1f}%)")

# ============================================
# QUALITY CONTROL CHECKPOINT
# ============================================

log("\n" + "="*70)
log("QUALITY CONTROL CHECKPOINT")
log("="*70)

# Calculate success rate (HIGH + MEDIUM confidence)
success_rate = (high_conf + medium_conf) / len(consensus_df)

log(f"\nConsensus Success Rate: {success_rate*100:.1f}%")
log(f"(HIGH + MEDIUM confidence poses)")

# Kinase docking expected: 70-80% success
if success_rate >= 0.70:
    status = "PASS"
    log(f"\n✅ EXCELLENT: {success_rate*100:.1f}% ≥ 70% expected for kinases")
    log("   Consensus docking is HIGH QUALITY")
elif success_rate >= 0.60:
    status = "ACCEPTABLE"
    log(f"\n✓ ACCEPTABLE: {success_rate*100:.1f}% close to 70% threshold")
    log("   Consensus docking is ACCEPTABLE for publication")
else:
    status = "REVIEW"
    log(f"\n⚠️  WARNING: Only {success_rate*100:.1f}% success rate")
    log("   Below expected 70% for kinases - REVIEW NEEDED")

# Check affinity correlation
affinity_corr = consensus_df[['vina_affinity', 'smina_affinity']].corr().iloc[0, 1]
log(f"\nAffinity Correlation (Vina vs SMINA): {affinity_corr:.3f}")

if affinity_corr >= 0.80:
    log(f"✅ EXCELLENT correlation (≥0.80)")
elif affinity_corr >= 0.60:
    log(f"✓ Good correlation (≥0.60)")
else:
    log(f"⚠️  Moderate correlation (<0.60)")

# ============================================
# SAVE RESULTS
# ============================================

log("\n5. Saving consensus results...")

# Save full consensus results
consensus_file = CONSENSUS_DIR / "consensus_docking_results.csv"
consensus_df.to_csv(consensus_file, index=False)
log(f"✓ Full results: {consensus_file}")

# Save high confidence poses only
high_conf_df = consensus_df[consensus_df['confidence'] == 'HIGH']
high_conf_file = CONSENSUS_DIR / "high_confidence_poses.csv"
high_conf_df.to_csv(high_conf_file, index=False)
log(f"✓ High confidence: {high_conf_file} ({len(high_conf_df)} poses)")

# Save acceptable poses (HIGH + MEDIUM)
acceptable_df = consensus_df[consensus_df['confidence'].isin(['HIGH', 'MEDIUM'])]
acceptable_file = CONSENSUS_DIR / "acceptable_poses.csv"
acceptable_df.to_csv(acceptable_file, index=False)
log(f"✓ Acceptable poses: {acceptable_file} ({len(acceptable_df)} poses)")

# Create summary statistics
summary_stats = {
    'total_common_dockings': len(consensus_df),
    'high_confidence': high_conf,
    'medium_confidence': medium_conf,
    'low_confidence': low_conf,
    'success_rate_percent': round(success_rate * 100, 1),
    'mean_rmsd_angstrom': round(consensus_df['rmsd_angstrom'].mean(), 2),
    'median_rmsd_angstrom': round(consensus_df['rmsd_angstrom'].median(), 2),
    'mean_affinity_diff': round(consensus_df['affinity_diff'].mean(), 2),
    'affinity_correlation': round(affinity_corr, 3),
    'quality_status': status
}

summary_file = CONSENSUS_DIR / "consensus_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
log(f"✓ Summary: {summary_file}")

# ============================================
# FINAL SUMMARY
# ============================================

log("\n" + "="*70)
log("CONSENSUS ANALYSIS COMPLETE")
log("="*70)

log(f"\n📊 FINAL STATISTICS:")
log(f"   Total dockings analyzed: {len(consensus_df)}")
log(f"   HIGH confidence (RMSD < 2Å): {high_conf} ({high_conf/len(consensus_df)*100:.1f}%)")
log(f"   MEDIUM confidence (2-3Å): {medium_conf} ({medium_conf/len(consensus_df)*100:.1f}%)")
log(f"   LOW confidence (>3Å): {low_conf} ({low_conf/len(consensus_df)*100:.1f}%)")
log(f"   Success rate: {success_rate*100:.1f}%")
log(f"   Affinity correlation: {affinity_corr:.3f}")

log(f"\n✅ Status: {status}")

if status == "PASS":
    log(f"\n🎉 CONSENSUS DOCKING VALIDATED!")
    log(f"\nNext steps:")
    log(f"  1. Use HIGH confidence poses for feature extraction")
    log(f"  2. Optionally include MEDIUM confidence with lower weight")
    log(f"  3. Proceed to structural feature extraction")
elif status == "ACCEPTABLE":
    log(f"\n✓ Consensus docking acceptable for publication")
    log(f"\nRecommendation:")
    log(f"  - Focus on HIGH confidence poses for main analysis")
    log(f"  - Mention consensus validation in methods")
else:
    log(f"\n⚠️  Review consensus results carefully")
    log(f"\nPossible issues:")
    log(f"  - Binding site definition may need adjustment")
    log(f"  - Some ligands may be challenging for these proteins")

log("\n" + "="*70)
log(f"\n📁 All results saved to: {CONSENSUS_DIR}")
log("="*70)
