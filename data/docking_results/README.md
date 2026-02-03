# Molecular Docking Preparation

This directory contains prepared structures for molecular docking studies of EGFR/BRAF inhibitors.

## Prepared Structures

### Protein Receptors (2)

* `egfr_prepared.pdb` - EGFR kinase domain (PDB: 1M17, Chain A)
   * Cleaned, hydrogens added at pH 7.4
   * Ready for docking with AutoDock Vina
   
* `braf_prepared.pdb` - BRAF V600E kinase (PDB: 4MNE, Chain A)
   * Cleaned, hydrogens added at pH 7.4
   * Ready for docking with AutoDock Vina

**Preparation method:** ChimeraX 1.11.1 (removed water/ions/ligands, added hydrogens)

### Ligands (15 compounds)

Located in `ligands/` directory - all prepared as PDBQT format for AutoDock Vina.

**EGFR Inhibitors (9):**
* AZD3759, Afatinib, Erlotinib, Gefitinib, Lapatinib
* Osimertinib, Sapitinib, Foretinib, Cediranib

**BRAF Inhibitors (3):**
* Dabrafenib, PLX-4720, SB590885

**Multi-kinase Inhibitors (3):**
* Axitinib, Motesanib, Sorafenib

**Preparation workflow:**
1. SMILES → 3D structure (RDKit ETKDG)
2. Geometry optimization (MMFF94 force field)
3. PDBQT conversion (Meeko v0.7.1)
4. Salt/counterion removal (kept largest fragment)

## Configuration

### `docking_config.json`

Contains binding box coordinates and docking parameters:

* **EGFR:** Center (29.0, 6.3, 54.2), Box size 25×25×25 Å
* **BRAF:** Center (6.6, -16.9, -35.7), Box size 25×25×25 Å
* Exhaustiveness: 32 (high thoroughness)
* Modes: 20 poses per ligand

## Data Files

* `ligands_prepared.csv` - Summary of prepared ligands with SMILES and file paths

## Software Used

* **ChimeraX 1.11.1** - Protein preparation
* **RDKit** - 3D structure generation and optimization
* **Meeko 0.7.1** - Ligand PDBQT conversion
* **BioPython** - PDB file processing

---

## Docking Results (COMPLETED ✅)

### Methods Used

Consensus docking performed with:
* **AutoDock Vina 1.2.7** (primary)
* **SMINA** (validation, based on Vina 1.1.2)

### Results Summary

**AutoDock Vina:**
* Total jobs: 30 (15 ligands × 2 proteins)
* Successful: 29/30 (96.7%)
* Runtime: 1.89 hours
* Mean affinity: -8.84 kcal/mol
* Results: `vina_results/vina_successful_dockings.csv`

**SMINA:**
* Total jobs: 30 (15 ligands × 2 proteins)
* Successful: 29/30 (96.7%)
* Runtime: 1.58 hours
* Mean affinity: -8.83 kcal/mol
* Results: `smina_results/smina_successful_dockings.csv`

### Consensus Validation

**Method:** RMSD comparison between Vina and SMINA top poses

**Confidence Levels:**
* HIGH: RMSD < 2.0Å (strong agreement)
* MEDIUM: RMSD 2.0-3.0Å (moderate agreement)  
* LOW: RMSD > 3.0Å (poor agreement)

**Results:**
* HIGH confidence: 24/29 (82.8%) ⭐
* MEDIUM confidence: 5/29 (17.2%)
* LOW confidence: 0/29 (0.0%)
* **Success rate: 100.0%** (exceeds 70% benchmark for kinases)

**Agreement Statistics:**
* Mean RMSD: 0.65 Å
* Median RMSD: 0.23 Å
* Affinity correlation: r = 0.966 (exceptional)
* Mean affinity difference: 0.12 kcal/mol

**Quality Control:** ✅ PASS - Consensus docking validated

**Output Files:**
* `consensus_results/consensus_docking_results.csv` - All 29 dockings with RMSD
* `consensus_results/high_confidence_poses.csv` - 24 HIGH confidence poses
* `consensus_results/acceptable_poses.csv` - All 29 acceptable poses
* `consensus_results/consensus_summary.json` - Summary statistics

### Per-Protein Performance

* **EGFR:** 11/14 HIGH confidence (78.6%)
* **BRAF:** 13/15 HIGH confidence (86.7%)

---

## Next Steps

**Status:** Docking phase complete, ready for feature extraction

**Upcoming:**
* Extract structural features from 24 high-confidence poses
* Integrate with transcriptomic data (95 genes)
* Train neural conditioning model for drug sensitivity prediction

---

**Created:** 2026  
**Project:** CASPID
**Last Updated:** February 2, 2026
