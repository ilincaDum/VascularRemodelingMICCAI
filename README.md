# Pulmonary Vessel Damage Prediction (A Graph-Based Approach)

This repository contains the core scripts used in my MSc thesis pipeline for analyzing and predicting longitudinal pulmonary vasculature injury (arteries/veins) following radiotherapy 

The end-to-end workflow is:

1. **Input data**: CT images, segmentations (lobes, artery/vein, tumor) masks, RT dose map  
2. **Pre-processing**: isotropic resampling, follow-up to baseline registration  
3. **VesselVio graph creation & annotation**: skeletonize vessels, create graphs, sample dose along vessel segments, encode anatomy/dose  
4. **Complete graphs**: per patient, modality (Artery/Vein), and timepoint  
5. **Central label sanitization**: fix ambiguous lobe labels (`LobeSanitization.py`)  
6. **Matching pipeline**: follow-up to baseline matching and label assignment (`MatchingPipeline.py`)  
7. **Analysis**: Statistical Tests
8. **Prediction**: ML baselines + graph modeling

---

## Pipeline Overview

<img width="900" height="800" alt="Blank diagram (2)" src="https://github.com/user-attachments/assets/5cea55c6-1823-41c3-ae33-661934dff391" />


---

## Repository files

### `LobeSanitization.py`
CLI tool to sanitize **Central (label=0)** edges in GraphML vessel graphs by:
- selecting anchor central edges (largest `radius_avg`)
- building a central-only subgraph filtered by a radius threshold
- computing multi-source geodesic distances from anchor nodes
- keeping central edges whose endpoints are in the reachable “core”
- otherwise relabeling central edges to the nearest lobar cluster 

**Outputs**
- `central_sanitized.graphml`
- `central_sanitize_audit.csv`
- `central_sanitize_decisions.json`
- batch summary `sanitize_batch_summary.csv`

---

### `MatchingPipeline.py`
Matches follow-up vessel segments to baseline and produces **analysis-ready labeled graphs**:
- optional global alignment (Umeyama similarity transform)
- Stage 1: per-lobe Hungarian matching (LSAP)
- Stage 2: rescue matching (mutual nearest neighbors)
- change labeling on baseline edges: `survived`, `damaged`, `disappeared`
- per-pair modeling table: `edges_bl_fu_modeling_rows.csv`
- match provenance and quality report

**Outputs (per patient/modality/FU)**
- `BL_labeled.graphml`, `FU_labeled.graphml`
- `survived_edges.csv`, `disappeared_edges.csv`
- `matched_edges_damage_flags.csv`
- `change_counts_by_lobe.csv`
- `edges_bl_fu_modeling_rows.csv`
- `match_provenance.csv`
- `overlay_labeled_changes.png`

**Global report**
- `_postmatch_reports/lobe_counts_and_quality.csv`

---

### `VasculatureVesselStatisticalAnalysis.ipynb`
- Wilcoxon signed-rank to identify significant differences in normalized blood volume metrics across artery and vein data
- Wilcoxon signed-rank to identify significant differences in vessel morphology (e.g. volume, tortuosity) across artery and vein data
- Longitudinal Analysis Checking both Immediate and Long Term Changes

---

### `MachineLearningPrediction.py`
Tabular baselines aligned with the GNN setup (patient-wise LOPO-style folds):
- Logistic Regression
- Random Forest
- uses `edges_bl_fu_modeling_rows.csv` from `MatchingPipeline.py`

---

### `VascularGraphPrediction.py`
Graph-based prediction using a **Line-Graph GNN**:
- runs per threshold and modality (Artery/Vein)
- LOPO patient splits with an explicit validation set
- writes dose trend and dose calibration summaries

---


## Installation

### venv + pip 

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

