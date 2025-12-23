#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline sklearn NMF (no tied weights, no conditions) — SINGLE FILE.

Reads one H5AD, runs sklearn.decomposition.NMF with rank K, saves a RESULTS-ONLY h5ad:
  - obsm["H_sklearn_nmf_k{K}"] : (cells x K)
  - varm["W_sklearn_nmf_k{K}"] : (genes x K)
  - uns["sklearn_nmf_k{K}"]    : metadata including reconstruction error
  - obs includes donor_id + cell_type 
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse as sp
from sklearn.decomposition import NMF

# Paths 
H5AD_PATH = os.getenv(
    "H5AD_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad",
)
OUT_PATH = os.getenv(
    "OUT_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_sklearn_nmf_k80.h5ad",
)

# Config
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)

K = int(os.getenv("K", "80"))
MAX_ITER = int(os.getenv("MAX_ITER", "100"))

CELLTYPE_COL = os.getenv("CELLTYPE_COL", "cell_type")
DONOR_COL = os.getenv("DONOR_COL", "donor_id")


LIBSIZE_LOG1P = os.getenv("LIBSIZE_LOG1P", "0") == "1"
TARGET_SUM = float(os.getenv("TARGET_SUM", "1e4"))

# sklearn NMF verbosity: prints per-iteration progress
VERBOSE = int(os.getenv("VERBOSE", "1"))

def _libsize_log1p_csr_cpu(X_csr, target_sum=1e4):
    rs = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    scale[nz] = target_sum / rs[nz]
    Xn = X_csr.multiply(scale[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data).astype(np.float32, copy=False)
    return Xn

def main():
    print(f"[global] GLOBAL_SEED={GLOBAL_SEED}, K={K}, MAX_ITER={MAX_ITER}, VERBOSE={VERBOSE}")
    print(f"[paths] H5AD_PATH={H5AD_PATH}")
    print(f"[paths] OUT_PATH ={OUT_PATH}")
    print(f"[prep]  LIBSIZE_LOG1P={LIBSIZE_LOG1P} (TARGET_SUM={TARGET_SUM:g})")

    adata = ad.read_h5ad(H5AD_PATH)
    print("[load] adata:", adata)

    # Pick matrix
    layer_for_nmf = os.getenv("LAYER_FOR_NMF", "none")
    if layer_for_nmf.lower() != "none" and layer_for_nmf in adata.layers:
        X = adata.layers[layer_for_nmf]
        print(f"[load] Using layer: {layer_for_nmf}")
    else:
        X = adata.X
        print("[load] Using adata.X")

    # Ensure CSR float32/float64, keep sparse 
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32)

    # Optional normalize/log1p on CPU (keeps sparse)
    if LIBSIZE_LOG1P:
        X = _libsize_log1p_csr_cpu(X, target_sum=TARGET_SUM)

    # Non-negativity check 
    if X.nnz > 0 and X.data.min() < 0:
        raise ValueError(
            f"[sklearn_nmf] Data contains negative values (min(data)={X.data.min()}). "
            "NMF requires non-negative inputs."
        )

    n_cells, n_genes = X.shape
    print(f"[sklearn_nmf] X shape: cells={n_cells}, genes={n_genes}, nnz={X.nnz}")

    nmf = NMF(
        n_components=K,
        init="nndsvda",
        max_iter=MAX_ITER,
        random_state=GLOBAL_SEED,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        verbose=VERBOSE,
    )

    # sklearn convention
    W_cells = nmf.fit_transform(X)         # cells x K
    H_k_genes = nmf.components_            # K x genes
    recon_err = float(nmf.reconstruction_err_)

    print(f"[sklearn_nmf] Reconstruction error: {recon_err:.6e}")

    # H_cell (cells x K), W_gene (genes x K)
    H_cell = W_cells.astype(np.float32, copy=False)
    W_gene = H_k_genes.T.astype(np.float32, copy=False)

    # results-only AnnData
    keep_obs = [c for c in [DONOR_COL, CELLTYPE_COL] if c in adata.obs.columns]
    obs_out = adata.obs[keep_obs].copy() if keep_obs else adata.obs.copy()

    # keep  gene names
    var_out = pd.DataFrame(index=adata.var_names)

    res = ad.AnnData(X=None, obs=obs_out, var=var_out)

    obsm_key = f"H_sklearn_nmf_k{K}"
    varm_key = f"W_sklearn_nmf_k{K}"
    uns_key  = f"sklearn_nmf_k{K}"

    res.obsm[obsm_key] = H_cell
    res.varm[varm_key] = W_gene

    res.uns[uns_key] = {
        "k": int(K),
        "max_iter": int(MAX_ITER),
        "random_state": int(GLOBAL_SEED),
        "reconstruction_err_": recon_err,
        "libsize_log1p": bool(LIBSIZE_LOG1P),
        "target_sum": float(TARGET_SUM),
        "note": "Baseline sklearn.decomposition.NMF; no tied weights; no conditions; single-file run.",
    }
    res.uns["model_name"] = "sklearn_NMF_baseline"
    res.uns["k_rank"] = int(K)

    # gzip compression 
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    res.write_h5ad(OUT_PATH, compression="gzip")
    print("[save] Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
