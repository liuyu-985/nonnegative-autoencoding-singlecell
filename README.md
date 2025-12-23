# Tabula Sapiens Immune — Conditional AE-NMF Benchmarks (Rank 80)

Code for training and evaluating NMF / Autoencoding-NMF variants on the Tabula Sapiens v2 Immune dataset (CZ CELLxGENE), producing cell embeddings and benchmark metrics used for manuscript figures.

Manuscript context: **“Non-negative Autoencoding of 60 Million Single-cell Transcriptomes at Rank 256”**.

This repository focuses on:
1) training 4 embedding models (baseline + AE-NMF variants),
2) exporting compact “results-only” `.h5ad` files containing embeddings + minimal metadata,
3) benchmarking **clustering agreement (ARI/NMI)** and **cell-type / donor classification** on the learned embeddings.

---

## Repository layout

### `Four_models_training_code/`
Training scripts for the 4 models (single-file workflow; donor_id as batch label).

- `nmf_core_1st_model.py`  
  Core implementation for **Conditional AE-NMF (Model 1)**.

- `run_cnmf1_singlefile.py`  
  Runs **Conditional AE-NMF (Model 1)** on one `.h5ad` input and saves a results-only `.h5ad`.

- `run_cnmf2_singlefile.py`  
  Runs **Conditional AE-NMF (Model 2)** (2nd formulation) on one `.h5ad` and saves a results-only `.h5ad`.

- `run_no_cond_nmf_singlefile.py`  
  Runs **No-condition AE-NMF** (autoencoding NMF without batch conditioning) and saves a results-only `.h5ad`.

- `run_base_nmf_singlefile.py`  
  Runs **baseline NMF** (sklearn) and saves a results-only `.h5ad`.

Outputs are written as compact AnnData files containing:
- `.obsm[...]` : cell embeddings (H / latent factors)
- `.varm[...]` : gene loadings (W)
- `.obs`       : metadata (at minimum includes `cell_type`, `donor_id`; additional fields can be kept if desired)

---

### `Benchmarks_code/`
Benchmark scripts that read the results-only `.h5ad` files and compute metrics.

- `Tabula_Immune_cell_clustering.py`  
  Computes **ARI/NMI** for clustering comparisons, reported separately using:
  - labels = `cell_type`
  - labels = `donor_id`

- `Tabula_Immune_cell_classification.py`  
  Computes cross-validated classification summaries (F1/AUROC/Accuracy/Precision/Recall), reported separately using:
  - labels = `cell_type`
  - labels = `donor_id`

---

### `SLRUM_jobs_code/`
SLURM submission scripts (GPU for AE-NMF variants; CPU for sklearn baseline; plus metric jobs).

- `cnmf1_singlefile.sh`
- `cnmf2_singlefile.sh`
- `no_cond_nmf_singlefile.sh`
- `base_nmf_singlefile.sh`
- `Tabula_Immune_cell_classification.sh`
- `Tabula_Immune_metrics.sh`

These scripts mainly set:
- module loads (Python/CUDA),
- conda/venv activation,
- environment variables (rank `K`, epochs, batch size, evaluation blocks),
- input/output paths (or rely on defaults inside the Python runners).

---

## Models (what “4 models” means)

All models are trained on the same filtered input dataset (Tabula Sapiens Immune; donors with >= 5000 cells kept).  
Batch label used for conditioning / invariance evaluation: **`donor_id`**.

1. **Base NMF (sklearn)**  
   Baseline factorization producing H (cells×k) and W (genes×k).

2. **AE-NMF (no condition)**  
   Autoencoding NMF-style model without batch conditioning.

3. **Conditional AE-NMF (Model 1)**  
   Conditioning on donor_id; produces shared and condition-aware embeddings.

4. **Conditional AE-NMF (Model 2)**  
   Conditional formulation using donor_id conditioning.

---

## Results-only `.h5ad` conventions

Each trained model writes a results-only AnnData containing embeddings and minimal metadata.

Typical keys:
- Conditional models: `.obsm["H_shared_k80"]` (and optionally condition-specific / concatenated embeddings)
- Baseline sklearn NMF: `.obsm["H_sklearn_nmf_k80"]`

Metadata required for downstream benchmarks:
- `cell_type` (for biological structure)
- `donor_id` (for batch structure / invariance)

---

## How to run

1) Train models (SLURM recommended)
- Submit one of the scripts in `SLRUM_jobs_code/` (GPU for AE-NMF variants).

2) Run benchmarks
- Clustering (ARI/NMI): `Benchmarks_code/Tabula_Immune_cell_clustering.py`
- Classification: `Benchmarks_code/Tabula_Immune_cell_classification.py`

Each benchmark produces two CSV outputs:
- by `cell_type`
- by `donor_id`

---

## Notes
- The full Tabula Sapiens Immune `.h5ad` can be large; training is intended for HPC/SLURM.
- For publication artifacts (e.g., Zenodo), we typically upload `.h5ad` files with embeddings + required metadata.

---

