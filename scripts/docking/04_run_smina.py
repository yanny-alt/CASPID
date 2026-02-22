#!/usr/bin/env python3
"""
CRITICAL: SMINA Consensus Docking
PRODUCTION VERSION - Validation of Vina results

SMINA is Vina-based with enhanced scoring
Used for consensus validation with AutoDock Vina

Author: CASPID Research Team
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

PROJECT_ROOT = Path("/Users/favourigwezeke/Personal_System/Research/Dr. Charles Nnadi/CASPID")
DOCKING_DIR = PROJECT_ROOT / "DATA" / "DOCKING"
LIGANDS_DIR = DOCKING_DIR / "LIGANDS"
RESULTS_DIR = DOCKING_DIR / "SMINA_RESULTS"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = RESULTS_DIR / f"smina_docking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

print("="*70)
print("SMINA - CONSENSUS DOCKING (PRODUCTION)")
print("="*70)
log("Starting SMINA docking pipeline")

# ============================================
# CHECK SMINA INSTALLATION
# ============================================

log("\n1. Checking SMINA installation...")

try:
    result = subprocess.run(['smina', '--version'], capture_output=True, text=True)
    smina_version = result.stdout.strip() if result.stdout else "SMINA installed"
    log(f"✓ {smina_version}")
except FileNotFoundError:
    log("✗ ERROR: SMINA not found!")
    log("  Install SMINA before proceeding")
    exit(1)

# ============================================
# LOAD CONFIGURATION
# ============================================

log("\n2. Loading configuration...")
config_file = DOCKING_DIR / "docking_config.json"

if not config_file.exists():
    log(f"✗ ERROR: {config_file} not found!")
    exit(1)

with open(config_file) as f:
    config = json.load(f)

log(f"✓ Configuration loaded for {len(config)} proteins")

# Verify receptor files exist
for protein_name, protein_config in config.items():
    receptor_file = Path(protein_config['receptor_file'])
    if receptor_file.exists():
        log(f"  ✓ {protein_name}: {receptor_file.name}")
    else:
        log(f"  ✗ {protein_name}: Receptor not found at {receptor_file}")
        exit(1)

# ============================================
# FIND LIGANDS
# ============================================

log("\n3. Finding prepared ligands...")
ligand_files = sorted(LIGANDS_DIR.glob("*.pdbqt"))

if not ligand_files:
    log(f"✗ ERROR: No .pdbqt files found in {LIGANDS_DIR}")
    exit(1)

log(f"✓ Found {len(ligand_files)} ligand files")
for lf in ligand_files[:5]:
    log(f"  - {lf.name}")
if len(ligand_files) > 5:
    log(f"  ... and {len(ligand_files)-5} more")

# ============================================
# SMINA DOCKING PARAMETERS
# ============================================

SMINA_PARAMS = {
    'exhaustiveness': 32,  # Same as Vina for fair comparison
    'num_modes': 20,       # Generate 20 poses
    'energy_range': 3,     # kcal/mol
    'cpu': 4,              # CPUs per docking job
    'scoring': 'default'   # SMINA's default scoring (vinardo)
}

log(f"\n4. SMINA parameters:")
log(f"   Exhaustiveness: {SMINA_PARAMS['exhaustiveness']}")
log(f"   Number of modes: {SMINA_PARAMS['num_modes']}")
log(f"   Energy range: {SMINA_PARAMS['energy_range']} kcal/mol")
log(f"   CPUs per job: {SMINA_PARAMS['cpu']}")
log(f"   Scoring: {SMINA_PARAMS['scoring']}")

# ============================================
# DOCKING FUNCTION (SINGLE JOB)
# ============================================

def run_smina_docking(job):
    """
    Run single SMINA docking job with quality control
    
    Returns: dict with status and results
    """
    protein_name, ligand_file, protein_config = job
    
    ligand_name = ligand_file.stem
    job_id = f"{protein_name}_{ligand_name}"
    
    # Output files
    output_dir = RESULTS_DIR / protein_name
    output_dir.mkdir(exist_ok=True)
    
    output_pdbqt = output_dir / f"{ligand_name}_docked.pdbqt"
    log_file_path = output_dir / f"{ligand_name}_smina.log"
    
    # Check if already done
    if output_pdbqt.exists():
        return {
            'job_id': job_id,
            'protein': protein_name,
            'ligand': ligand_name,
            'status': 'SKIPPED',
            'reason': 'Already exists',
            'output': str(output_pdbqt)
        }
    
    # Get receptor file
    receptor_file = Path(protein_config['receptor_file'])
    
    # Build SMINA command
    cmd = [
        'smina',
        '--receptor', str(receptor_file),
        '--ligand', str(ligand_file),
        '--center_x', str(protein_config['center_x']),
        '--center_y', str(protein_config['center_y']),
        '--center_z', str(protein_config['center_z']),
        '--size_x', str(protein_config['size_x']),
        '--size_y', str(protein_config['size_y']),
        '--size_z', str(protein_config['size_z']),
        '--exhaustiveness', str(SMINA_PARAMS['exhaustiveness']),
        '--num_modes', str(SMINA_PARAMS['num_modes']),
        '--energy_range', str(SMINA_PARAMS['energy_range']),
        '--cpu', str(SMINA_PARAMS['cpu']),
        '--out', str(output_pdbqt),
        '--log', str(log_file_path)
    ]
    
    # Run docking
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2400  # 40 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # SMINA outputs to log file (has --log option)
        if result.returncode == 0:
            # Parse binding affinity from log
            affinity = parse_smina_affinity(log_file_path)
            
            if affinity is None:
                return {
                    'job_id': job_id,
                    'protein': protein_name,
                    'ligand': ligand_name,
                    'status': 'WARNING',
                    'reason': 'Could not parse affinity',
                    'output': str(output_pdbqt),
                    'time_seconds': round(elapsed, 1)
                }
            
            return {
                'job_id': job_id,
                'protein': protein_name,
                'ligand': ligand_name,
                'status': 'SUCCESS',
                'output': str(output_pdbqt),
                'log': str(log_file_path),
                'affinity_kcal_mol': affinity,
                'time_seconds': round(elapsed, 1)
            }
        else:
            return {
                'job_id': job_id,
                'protein': protein_name,
                'ligand': ligand_name,
                'status': 'FAILED',
                'reason': f'SMINA returned error code {result.returncode}',
                'stderr': result.stderr[:200] if result.stderr else '',
                'time_seconds': round(elapsed, 1)
            }
    
    except subprocess.TimeoutExpired:
        return {
            'job_id': job_id,
            'protein': protein_name,
            'ligand': ligand_name,
            'status': 'FAILED',
            'reason': 'Timeout (>40 min)'
        }
    
    except Exception as e:
        return {
            'job_id': job_id,
            'protein': protein_name,
            'ligand': ligand_name,
            'status': 'FAILED',
            'reason': f'Exception: {str(e)}'
        }

def parse_smina_affinity(log_file_path):
    """
    Extract best binding affinity from SMINA log
    Format similar to Vina
    """
    try:
        with open(log_file_path) as f:
            lines = f.readlines()
        
        # Find result table
        for i, line in enumerate(lines):
            if 'mode |   affinity' in line:
                # Result is 3 lines down (same as Vina)
                if i + 3 < len(lines):
                    result_line = lines[i + 3]
                    parts = result_line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
        
        return None
    except Exception as e:
        return None

# ============================================
# PREPARE JOBS
# ============================================

log("\n5. Preparing docking jobs...")

jobs = []
for protein_name, protein_config in config.items():
    for ligand_file in ligand_files:
        jobs.append((protein_name, ligand_file, protein_config))

log(f"✓ Total jobs: {len(jobs)}")
log(f"  = {len(config)} proteins × {len(ligand_files)} ligands")

# ============================================
# FILTER LIGANDS BY TARGET
# ============================================

log("\n6. Filtering ligands by target...")

# Define which ligands belong to which protein
TARGET_LIGANDS = {
    'EGFR': ['afatinib', 'azd3759', 'erlotinib', 'gefitinib', 
             'lapatinib', 'osimertinib', 'sapitinib'],
    'BRAF': ['dabrafenib', 'plx_4720', 'sb590885'],
    'MEK1': ['pd0325901', 'refametinib', 'selumetinib', 'trametinib']
}

# Prepare target-specific jobs
jobs = []
for protein_name, protein_config in config.items():
    # Get ligands for this protein
    target_ligands = TARGET_LIGANDS.get(protein_name, [])
    
    matched_ligands = []
    for ligand_file in ligand_files:
        ligand_name = ligand_file.stem.lower().replace('-', '_')
        
        # Only dock if ligand belongs to this protein
        if ligand_name in [l.lower().replace('-', '_') for l in target_ligands]:
            jobs.append((protein_name, ligand_file, protein_config))
            matched_ligands.append(ligand_name)
    
    log(f"  {protein_name}: {len(matched_ligands)} ligands")
    for lig in matched_ligands:
        log(f"    - {lig}")

log(f"\n✓ Total jobs: {len(jobs)}")
log(f"  (target-specific filtering applied)")

# ============================================
# PARALLEL EXECUTION SETUP
# ============================================

log("\n7. Calculating parallelization strategy...")

total_cpus = cpu_count()
cpus_per_job = SMINA_PARAMS['cpu']
max_parallel = max(1, int((total_cpus // cpus_per_job) * 0.75))

log(f"   System CPUs: {total_cpus}")
log(f"   CPUs per job: {cpus_per_job}")
log(f"   Parallel jobs: {max_parallel} (conservative)")

avg_time_per_job = 20  # minutes
total_time_estimate = (len(jobs) * avg_time_per_job) / max_parallel / 60

log(f"\n   Estimated total time: {total_time_estimate:.1f} hours")
log(f"   (assuming {avg_time_per_job} min per job average)")

# ============================================
# USER CONFIRMATION
# ============================================

print(f"\n{'='*70}")
print(f"READY TO START SMINA DOCKING")
print(f"{'='*70}")
print(f"\n  Jobs to run: {len(jobs)}")
print(f"  Parallel jobs: {max_parallel}")
print(f"  Estimated time: {total_time_estimate:.1f} hours")
print(f"\n  Output directory: {RESULTS_DIR}")
print(f"\n  Target-specific ligands:")
for protein in config.keys():
    protein_jobs = [j for j in jobs if j[0] == protein]
    print(f"    {protein}: {len(protein_jobs)} ligands")

response = input(f"\n  Continue? (yes/no): ").strip().lower()

if response != 'yes':
    log("\n✗ Aborted by user")
    print("\nYou can restart anytime - completed jobs will be skipped")
    exit(0)

# ============================================
# RUN ALL JOBS
# ============================================

if __name__ == '__main__':
    log("\n8. Starting full SMINA docking run...")
    log(f"   Time started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Run in parallel
    try:
        with Pool(processes=max_parallel) as pool:
            results = pool.map(run_smina_docking, jobs)
    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user (Ctrl+C)")
        log("   Partial results will be saved")
        results = []

    total_time = time.time() - start_time

    log(f"\n   Time completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"   Total elapsed: {total_time/3600:.2f} hours")

    # ============================================
    # ANALYZE RESULTS
    # ============================================

    log("\n9. Analyzing results...")

    success = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    skipped = [r for r in results if r['status'] == 'SKIPPED']
    warning = [r for r in results if r['status'] == 'WARNING']

    log(f"\n   ✓ SUCCESS: {len(success)}/{len(jobs)} ({len(success)/len(jobs)*100:.1f}%)")
    log(f"   ✗ FAILED:  {len(failed)}/{len(jobs)}")
    log(f"   ⚠ WARNING: {len(warning)}/{len(jobs)}")
    log(f"   - SKIPPED: {len(skipped)}/{len(jobs)}")

    if failed:
        log(f"\n   Failed jobs:")
        for f in failed[:10]:
            log(f"     {f['job_id']}: {f.get('reason', 'Unknown')}")
        if len(failed) > 10:
            log(f"     ... and {len(failed) - 10} more")

    # ============================================
    # SAVE RESULTS
    # ============================================

    log("\n10. Saving results...")

    if results:
        results_df = pd.DataFrame(results)
        results_file = RESULTS_DIR / "smina_results_summary.csv"
        results_df.to_csv(results_file, index=False)
        log(f"    ✓ All results: {results_file}")

    if success:
        success_df = pd.DataFrame(success)
        success_file = RESULTS_DIR / "smina_successful_dockings.csv"
        success_df.to_csv(success_file, index=False)
        log(f"    ✓ Successful dockings: {success_file}")
        
        affinities = success_df['affinity_kcal_mol'].dropna()
        if len(affinities) > 0:
            log(f"\n    Binding affinity statistics ({len(affinities)} values):")
            log(f"      Mean:   {affinities.mean():.2f} kcal/mol")
            log(f"      Median: {affinities.median():.2f} kcal/mol")
            log(f"      Best:   {affinities.min():.2f} kcal/mol")
            log(f"      Worst:  {affinities.max():.2f} kcal/mol")

    # ============================================
    # QUALITY CONTROL
    # ============================================

    log("\n" + "="*70)
    log("QUALITY CONTROL CHECKPOINT")
    log("="*70)

    success_rate = len(success) / len(jobs) if len(jobs) > 0 else 0

    if success_rate >= 0.90:
        log(f"\n✅ EXCELLENT: {success_rate*100:.1f}% success rate")
        status = "PASS"
    elif success_rate >= 0.70:
        log(f"\n✓ ACCEPTABLE: {success_rate*100:.1f}% success rate")
        status = "PASS"
    else:
        log(f"\n⚠️  WARNING: Only {success_rate*100:.1f}% success rate")
        status = "REVIEW"

    for protein in config.keys():
        protein_success = [r for r in success if r.get('protein') == protein]
        log(f"\n   {protein}: {len(protein_success)}/{len(ligand_files)} ligands")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    log("\n" + "="*70)
    log("SMINA DOCKING COMPLETE")
    log("="*70)

    log(f"\n📊 FINAL STATISTICS:")
    log(f"   Total jobs: {len(jobs)}")
    log(f"   Successful: {len(success)} ({success_rate*100:.1f}%)")
    log(f"   Failed: {len(failed)}")
    log(f"   Total time: {total_time/3600:.2f} hours")
    if len(jobs) > 0:
        log(f"   Average per job: {total_time/len(jobs)/60:.1f} minutes")

    log(f"\n✅ Status: {status}")

    if status == "PASS":
        log(f"\n🎉 SMINA validation complete!")
        log(f"\nNext step: Consensus analysis")
        log(f"  Compare Vina vs SMINA results")
    else:
        log(f"\n⚠️  Review failures before proceeding")

    log("\n" + "="*70)
