#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import anndata as ad
import scipy.sparse as sp

from nmf_core_1st_model import (
    prepare_X_and_conditions,
    train_cond_nmf_for_k,
    csr_to_cpu,
    to_cpu,
    _has_cupy,
)

def safe_compute_H_matrices(X_cpu, C_global_cpu, Wc, Uc):
    """
    Same math as ((C@U)@W.T) but computed as C@(U@W.T) to avoid (n_cells x n_genes) dense intermediate.
    """
    if not sp.isspmatrix_csr(X_cpu):
        X_cpu = sp.csr_matrix(X_cpu)
    if not sp.isspmatrix_csr(C_global_cpu):
        C_global_cpu = sp.csr_matrix(C_global_cpu)

    Wc = np.asarray(Wc, dtype=np.float32)
    Uc = np.asarray(Uc, dtype=np.float32)

    H_shared = (X_cpu @ Wc.T).astype(np.float32, copy=False)          # (n, k)
    UWT = (Uc @ Wc.T).astype(np.float32, copy=False)                   # (m_total, k)
    H_cond_latent = (C_global_cpu @ UWT).astype(np.float32, copy=False)# (n, k)
    H_concat = np.concatenate([H_shared, H_cond_latent], axis=1).astype(np.float32, copy=False)
    return H_shared, H_cond_latent, H_concat

# Config 
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)

H5AD_PATH = os.getenv(
    "H5AD_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad",
)
OUT_PATH = os.getenv(
    "OUT_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_1st_donor_only_k80.h5ad",
)

COND_COLS = ["donor_id"]
CELLTYPE_COL = "cell_type"

LAYER_NAME = os.getenv("LAYER_FOR_NMF", "decontXcounts")
if LAYER_NAME.strip().lower() in {"", "none"}:
    LAYER_NAME = None

# Training setting
K = int(os.getenv("K", "80"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
LAMBDA_W = float(os.getenv("LAMBDA_W", "1e-2"))
LAMBDA_U = float(os.getenv("LAMBDA_U", "1e-1"))
GAMMA_INV = float(os.getenv("GAMMA_INV", "1e-3"))
ETA_HSIC = float(os.getenv("ETA_HSIC", "5e-4"))
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "5"))
EARLY_STOP = os.getenv("EARLY_STOP", "1") == "1"
PATIENCE = int(os.getenv("PATIENCE", "6"))
FULL_EVAL_FRAC = float(os.getenv("FULL_EVAL_FRAC", "0.05"))
EVAL_BLOCK = int(os.getenv("EVAL_BLOCK", "256"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1"))
SEED_OFFSET = int(os.getenv("SEED_OFFSET", str(GLOBAL_SEED)))

# Stratify ONLY by cell type 
USE_STRATIFIED = True
STRATIFY_COLS = [CELLTYPE_COL]

# Optional: do normalize_total+log1p on CPU to avoid GPU sparse multiply OOM
CPU_NORM_LOG1P = os.getenv("CPU_NORM_LOG1P", "1") == "1"

print("=== CNMF_1st donor-only (single file) ===")
print("[cfg] H5AD_PATH =", H5AD_PATH)
print("[cfg] OUT_PATH  =", OUT_PATH)
print("[cfg] COND_COLS =", COND_COLS)
print("[cfg] K         =", K)
print("[cfg] has_cupy  =", _has_cupy, "USE_CUPY env =", os.getenv("USE_CUPY", "1"))
print("[cfg] CPU_NORM_LOG1P =", CPU_NORM_LOG1P)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# Load
adata = ad.read_h5ad(H5AD_PATH)
print("[run] Loaded:", adata)


# Optional CPU normalize/log1p on the chosen layer
if CPU_NORM_LOG1P:
    import scanpy as sc
    layer_for_norm = LAYER_NAME  
    sc.pp.normalize_total(adata, target_sum=1e4, layer=layer_for_norm)
    sc.pp.log1p(adata, layer=layer_for_norm)
    LIBSIZE_LOG1P = False
else:
    LIBSIZE_LOG1P = True  


# Prepare X + conditions (core)
adata_proc, X, cond_info, C_global, C_global_cpu = prepare_X_and_conditions(
    adata,
    cond_cols=COND_COLS,
    celltype_col=CELLTYPE_COL,
    layer_name=LAYER_NAME,
    use_hvg=False,
    libsize_log1p=LIBSIZE_LOG1P,
)

obs_df = adata_proc.obs.copy()
X_cpu = csr_to_cpu(X)

# Train
best_W, best_U, best_ep, best_full = train_cond_nmf_for_k(
    X=X,
    C_global=C_global,
    cond_info=cond_info,
    celltype_col=CELLTYPE_COL,
    k_rank=K,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lambda_W=LAMBDA_W,
    lambda_U=LAMBDA_U,
    gamma_inv_base=GAMMA_INV,
    eta_hsic_base=ETA_HSIC,
    warmup_epochs=WARMUP_EPOCHS,
    early_stop=EARLY_STOP,
    patience=PATIENCE,
    full_eval_frac=FULL_EVAL_FRAC,
    eval_block=EVAL_BLOCK,
    seed_offset=SEED_OFFSET,
    print_every=PRINT_EVERY,
    adata_obs=obs_df,
    use_stratified=USE_STRATIFIED,
    stratify_cols=STRATIFY_COLS,
)

print(f"✅ CNMF_1st done: best epoch={best_ep}, best full loss={best_full:.6e}")

Wc = to_cpu(best_W)
Uc = to_cpu(best_U)


# Safe H computation
H_shared, H_cond_latent, H_concat = safe_compute_H_matrices(X_cpu, C_global_cpu, Wc, Uc)


# Results-only save 
cond_meta_small = {
    col: {
        "categories": cond_info[col]["categories"],
        "size": cond_info[col]["size"],
        "offset": cond_info[col]["offset"],
    }
    for col in cond_info
}

res = ad.AnnData(X=None, obs=adata_proc.obs.copy(), var=adata_proc.var.copy())
res.obsm[f"H_shared_k{K}"] = H_shared
res.obsm[f"H_cond_latent_k{K}"] = H_cond_latent
res.obsm[f"H_concat_full_k{K}"] = H_concat
res.varm[f"W_concat_k{K}"] = Wc.T.astype(np.float32, copy=False)
res.varm[f"U_concat_gene_k{K}"] = Uc.T.astype(np.float32, copy=False)
res.uns[f"cond_meta_concat_k{K}"] = cond_meta_small
res.uns["model_name"] = "CNMF_1st_donor_only"
res.uns["k_rank"] = int(K)
res.uns["best_epoch"] = int(best_ep)
res.uns["best_full_loss"] = float(best_full)

res.write_h5ad(OUT_PATH)
print("Saved:", OUT_PATH)
