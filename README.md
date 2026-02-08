# CASPID: Context-Aware Structural Pattern Integration for Drug Discovery

## Overview

CASPID is a machine learning framework that discovers context-dependent protein-drug binding features by integrating:
- Structural features from consensus molecular docking
- Transcriptomic cellular context
- Neural conditioning layer for context-dependent feature reweighting

## Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA ACQUISITION                    │
├─────────────────────────────────────────────────────────────────┤
│  GDSC Drug Response  │  DepMap Transcriptomics  │  PDB Structures│
│   (IC50 values)      │    (RNAseq TPM)          │  (EGFR, BRAF)  │
└──────────────┬───────────────────┬────────────────────┬──────────┘
               │                   │                    │
               ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DATA INTEGRATION                    │
├─────────────────────────────────────────────────────────────────┤
│  • Match cell lines (GDSC ↔ DepMap)                            │
│  • Filter EGFR/BRAF inhibitors                                 │
│  • Select transcriptomic features (100 genes)                  │
│  • Quality control and preprocessing                           │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: CONSENSUS MOLECULAR DOCKING               │
├─────────────────────────────────────────────────────────────────┤
│  AutoDock Vina  │  SMINA  →  Consensus Poses       │
│  (RMSD < 2.0Å agreement criterion)                             │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 4: STRUCTURAL FEATURE EXTRACTION               │
├─────────────────────────────────────────────────────────────────┤
│  • Distance-based (100)     • Geometric (60)                   │
│  • Physicochemical (80)     • Interaction (60)                 │
│  • Pharmacophore (40)                                          │
│  Total: ~340 features per compound-pose                       │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 5: FEATURE SELECTION (ML)                    │
├─────────────────────────────────────────────────────────────────┤
│  Boruta ──┐                                                    │
│  MI ──────┼──→  Consensus (≥2/3 methods)  →  ~30 features     │
│  SHAP ────┘                                                    │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 6: NEURAL CONDITIONING LAYER (INNOVATION)         │
├─────────────────────────────────────────────────────────────────┤
│  Structural Features (30) ──┐                                  │
│                             ├──→ Conditioning → Weighted        │
│  Transcriptomics (100) ─────┘      Layer        Features       │
│                                                                 │
│  weights = Neural_Net(Transcriptomics)                         │
│  conditioned_features = features ⊙ weights                     │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 7: ACTIVITY PREDICTION (XGBoost)               │
├─────────────────────────────────────────────────────────────────┤
│  Conditioned Features + Transcriptomics + Interactions         │
│                          ↓                                      │
│                  XGBoost Regressor                              │
│                          ↓                                      │
│                  Predicted pIC50                                │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 8: VALIDATION & INTERPRETATION                    │
├─────────────────────────────────────────────────────────────────┤
│  • 5-fold CV × 10 repeats                                      │
│  • Benchmark comparisons (Structure-only, Context-only, etc.)  │
│  • BRAF validation (generalizability)                          │
│  • SHAP analysis (feature importance)                          │
│  • Biological interpretation                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Targets

**Primary**: EGFR (Epidermal Growth Factor Receptor)
- 15-25 inhibitors from GDSC
- ~600-800 cell lines
- PDB: 1M17 (wild-type + erlotinib)

**Validation**: BRAF (V-Raf Murine Sarcoma Viral Oncogene Homolog B)
- 10-20 inhibitors from GDSC
- Same cell line panel
- PDB: 4MNE (V600E mutant)

## Installation
```bash
# Create conda environment
conda env create -f environment/environment.yml

# Or install manually
conda create -n caspid python=3.10 -y
conda activate caspid
pip install -r environment/requirements.txt
```

## Usage
```bash
# Activate environment
conda activate caspid

# Run full pipeline
python scripts/run_caspid_pipeline.py --target EGFR --config config.yaml

# Or step-by-step
python scripts/data_processing/01_integrate_data.py
python scripts/docking/02_consensus_docking.py
python scripts/feature_extraction/03_extract_features.py
python scripts/modeling/04_train_caspid.py
```

## Project Status

- [x] Project structure created
- [x] Data downloaded
- [x] Data integration
- [x] Consensus docking
- [x] Feature extraction
- [x] Feature selection
- [x] Neural conditioning layer
- [x] Model training
- [ ] Validation
- [ ] Manuscript

## Data Sources

- **Drug Activity**: GDSC v2 (https://www.cancerrxgene.org)
- **Transcriptomics**: DepMap 25Q3 (https://depmap.org)
- **Structures**: RCSB PDB (https://www.rcsb.org)

## License

Academic research use only.

## Contact

Favour Igwezeke
University of Nigeria 
Beingfave@gmail.com
