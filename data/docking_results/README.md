# Molecular Docking Preparation

This directory contains prepared structures for molecular docking studies of EGFR/BRAF inhibitors.

## Prepared Structures

### Protein Receptors (2)
- **`egfr_prepared.pdb`** - EGFR kinase domain (PDB: 1M17, Chain A)
  - Cleaned, hydrogens added at pH 7.4
  - Ready for docking with AutoDock Vina
  
- **`braf_prepared.pdb`** - BRAF V600E kinase (PDB: 4MNE, Chain A)
  - Cleaned, hydrogens added at pH 7.4
  - Ready for docking with AutoDock Vina

**Preparation method:** ChimeraX 1.11.1 (removed water/ions/ligands, added hydrogens)

### Ligands (15 compounds)
Located in `LIGANDS/` directory - all prepared as PDBQT format for AutoDock Vina.

**EGFR Inhibitors (9):**
- AZD3759, Afatinib, Erlotinib, Gefitinib, Lapatinib
- Osimertinib, Sapitinib, Foretinib, Cediranib

**BRAF Inhibitors (3):**
- Dabrafenib, PLX-4720, SB590885

**Multi-kinase Inhibitors (3):**
- Axitinib, Motesanib, Sorafenib

**Preparation workflow:**
1. SMILES → 3D structure (RDKit ETKDG)
2. Geometry optimization (MMFF94 force field)
3. PDBQT conversion (Meeko v0.7.1)
4. Salt/counterion removal (kept largest fragment)

## Configuration

### `docking_config.json`
Contains binding box coordinates and docking parameters:
- **EGFR**: Center (29.0, 6.3, 54.2), Box size 25×25×25 Å
- **BRAF**: Center (6.6, -16.9, -35.7), Box size 25×25×25 Å
- Exhaustiveness: 32 (high thoroughness)
- Modes: 20 poses per ligand

## Data Files

- **`ligands_prepared.csv`** - Summary of prepared ligands with SMILES and file paths

## Software Used

- **ChimeraX 1.11.1** - Protein preparation
- **RDKit** - 3D structure generation and optimization
- **Meeko 0.7.1** - Ligand PDBQT conversion
- **BioPython** - PDB file processing

## Next Steps

Molecular docking will be performed using:
- AutoDock Vina (primary)
- SMINA (validation)
- LeDock (consensus scoring)

---

**Created:** 2026
**Project:** CASPID 
