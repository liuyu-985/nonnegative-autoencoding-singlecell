#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
# Tied-weights NMF (no conditions): Xhat = H W
# - H = X W^T, W shared encoder/decoder  (AE-style)
# - GPU-ready (CuPy when available)
# - Streaming full loss; early stopping
# - SINGLE-FILE run (Tabula Sapiens Immune)
# - Saves RESULTS-ONLY h5ad: H_shared + W + minimal obs (donor_id, cell_type)
#
# NOTE: row_norm is controlled by env var ROW_NORM:
#####################################################

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse as sp

# Paths 
H5AD_PATH = os.getenv(
    "H5AD_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad",
)
OUT_PATH = os.getenv(
    "OUT_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_no_cond_k80_row_norm_false.h5ad",
)

# Reproducibility
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)

# GPU backend
USE_GPU = os.getenv("USE_CUPY", "1") == "1"
_has_cupy = False
if USE_GPU:
    try:
        import cupy as cp
        from cupyx.scipy import sparse as cpx_sp
        _has_cupy = True
        cp.random.seed(GLOBAL_SEED)
    except Exception:
        _has_cupy = False

xp  = cp if _has_cupy else np
spx = cpx_sp if _has_cupy else sp

def to_cpu(a):
    if _has_cupy and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a

def csr_to_gpu(A_csr):
    if not _has_cupy:
        return A_csr
    if sp.isspmatrix_csr(A_csr):
        return cpx_sp.csr_matrix(
            (cp.asarray(A_csr.data),
             cp.asarray(A_csr.indices),
             cp.asarray(A_csr.indptr)),
            shape=A_csr.shape,
        )
    elif isinstance(A_csr, cpx_sp.csr_matrix):
        return A_csr
    else:
        return csr_to_gpu(sp.csr_matrix(A_csr))


# Config
celltype_col = os.getenv("CELLTYPE_COL", "cell_type")
donor_col    = os.getenv("DONOR_COL", "donor_id")

# stratify only affects minibatch sampling 
STRATIFY_COLS = os.getenv("STRATIFY_COLS", celltype_col).split(",")
STRATIFY_COLS = [c.strip() for c in STRATIFY_COLS if c.strip()]

use_hvg       = os.getenv("USE_HVG", "0") == "1"
n_hvg         = int(os.getenv("N_HVG", "10000"))
libsize_log1p = os.getenv("LIBSIZE_LOG1P", "1") == "1"

# avoid GPU sparse multiply OOM in preprocessing
CPU_NORM_LOG1P = os.getenv("CPU_NORM_LOG1P", "1") == "1"

row_norm = os.getenv("ROW_NORM", "1") == "1"

k           = int(os.getenv("K", "80"))
epochs      = int(os.getenv("EPOCHS", "100"))
batch_size  = int(os.getenv("BATCH_SIZE", "512"))
patience    = int(os.getenv("PATIENCE", "6"))
print_every = int(os.getenv("PRINT_EVERY", "1"))

lambda_W = float(os.getenv("LAMBDA_W", "1e-2"))
lrW      = float(os.getenv("LR_W", "1e-3"))

FULL_EVAL_FRAC = float(os.getenv("FULL_EVAL_FRAC", "0.05"))
EVAL_BLOCK     = int(os.getenv("EVAL_BLOCK", "256"))


# Helpers
def libsize_log1p_transform_csr_cpu(X_csr, target_sum=1e4):
    rs = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    scale[nz] = target_sum / rs[nz]
    Xn = X_csr.multiply(scale[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data).astype(np.float32, copy=False)
    return Xn

def iterate_minibatches_indices(n, batch_size, rng):
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s:min(s + batch_size, n)]

def iterate_minibatches_stratified(obs_df, by_cols, batch_size, rng):
    groups = obs_df.groupby(by_cols, dropna=False, observed=True).indices
    pools = {g: rng.permutation(ix) for g, ix in groups.items()}
    keys = list(pools.keys())
    ptrs = {g: 0 for g in keys}
    out = []
    while True:
        added = 0
        for g in keys:
            pool = pools[g]
            ptr = ptrs[g]
            if ptr < len(pool):
                take = min(batch_size - len(out), len(pool) - ptr)
                out.extend(pool[ptr:ptr + take])
                ptrs[g] = ptr + take
                added += take
            if len(out) >= batch_size:
                yield np.array(out, dtype=int)
                out = []
        if added == 0:
            if out:
                yield np.array(out, dtype=int)
            break

class AdamMat:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = xp.zeros(shape, dtype=xp.float32)
        self.v = xp.zeros(shape, dtype=xp.float32)
        self.t = 0

    def step(self, P, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        P -= self.lr * mhat / (xp.sqrt(vhat) + self.eps)
        return P

def _row_blocks(n, block):
    s = 0
    while s < n:
        e = min(s + block, n)
        yield s, e
        s = e

# Main
def main():
    print("=== tied no-cond AE-NMF (single file) ===")
    print("[cfg] H5AD_PATH =", H5AD_PATH)
    print("[cfg] OUT_PATH  =", OUT_PATH)
    print("[cfg] k=", k, "epochs=", epochs, "batch_size=", batch_size)
    print("[cfg] GPU requested=", USE_GPU, "has_cupy=", _has_cupy)
    print("[cfg] CPU_NORM_LOG1P=", CPU_NORM_LOG1P, "LIBSIZE_LOG1P=", libsize_log1p)
    print("[cfg] USE_HVG=", use_hvg, "ROW_NORM=", row_norm)
    print("[cfg] STRATIFY_COLS=", STRATIFY_COLS)
    print("[cfg] FULL_EVAL_FRAC=", FULL_EVAL_FRAC, "EVAL_BLOCK=", EVAL_BLOCK)

    adata = ad.read_h5ad(H5AD_PATH)
    print(adata)

    # Choose layer (optional)
    layer_for_nmf = os.getenv("LAYER_FOR_NMF", "none")
    if layer_for_nmf.lower() != "none" and layer_for_nmf in adata.layers:
        X = adata.layers[layer_for_nmf]
        print(f"[info] Using layer: {layer_for_nmf}")
    else:
        X = adata.X
        print("[info] Using adata.X")

    # CPU CSR float32
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # HVG (optional; CPU) 
    if use_hvg:
        import scanpy as sc
        tmp = ad.AnnData(X.copy(), var=adata.var.copy(), obs=adata.obs[[]].copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=n_hvg)
        hvg_mask = tmp.var["highly_variable"].to_numpy()
        X = X[:, hvg_mask]
        adata = adata[:, hvg_mask].copy()
        print(f"[info] HVG retained genes: {int(hvg_mask.sum())}")

    # Normalize/log1p on CPU
    if libsize_log1p:
        X = libsize_log1p_transform_csr_cpu(X)

    n, p = X.shape
    print(f"[info] X after prep: shape={X.shape}, nnz={X.nnz}")

    X_cpu = X
    X_train = csr_to_gpu(X_cpu) if _has_cupy else X_cpu

    # Stratified batching columns
    strat_cols = [c for c in STRATIFY_COLS if c in adata.obs.columns]
    use_strat = (len(strat_cols) > 0)
    print("[info] Stratified batching", "ON" if use_strat else "OFF", "by:", strat_cols if use_strat else "(none)")

    # Initialize W (k x p) 
    rng = np.random.default_rng(GLOBAL_SEED)
    W = xp.asarray(rng.random((k, p)), dtype=xp.float32)
    if row_norm:
        W /= (xp.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    optW = AdamMat(W.shape, lrW)

    best_total, best_W, best_ep = np.inf, W.copy(), 0
    bad = 0

    # Precompute ||X||_F^2 efficiently
    if _has_cupy:
        import cupy as cp
        x_sse = float(to_cpu(cp.dot(X_train.data, X_train.data)))
    else:
        x_sse = float(np.dot(X_train.data, X_train.data))

    for ep in range(1, epochs + 1):
        running, steps = 0.0, 0

        if use_strat:
            it = iterate_minibatches_stratified(
                adata.obs.reset_index(drop=True),
                strat_cols,
                batch_size,
                rng,
            )
        else:
            it = iterate_minibatches_indices(n, batch_size, rng)

        for rows in it:
            rows = np.asarray(rows, dtype=int)
            Xb = X_train[rows, :]
            B = int(Xb.shape[0])

            Hb = (Xb @ W.T)        # (B,k)
            HW = (Hb @ W)          # (B,p)
            Xb_dense = Xb.toarray().astype(xp.float32, copy=False)

            E = HW - Xb_dense
            data_loss = float(to_cpu((E * E).sum()) / max(1, B))
            reg_loss = float(to_cpu(lambda_W * (W * W).sum()))
            Lb = data_loss + reg_loss
            running += Lb
            steps += 1

            # gradient wrt W (tied weights)
            HtE = (Hb.T @ E)                    # (k,p)
            EWt = (E @ W.T)                     # (B,k)
            XtEWt = (Xb.T @ EWt)                # (p,k)
            cross = XtEWt.T                     # (k,p)
            gW = (2.0 * (HtE + cross)) / max(1, B) + 2.0 * lambda_W * W

            W = optW.step(W, gW)
            W = xp.maximum(W, 0.0)

            # only normalize if row_norm is enabled
            if row_norm:
                W /= (xp.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

            if _has_cupy:
                cp.get_default_memory_pool().free_all_blocks()

        avg_batch_loss = running / max(1, steps)

        # streamed full objective
        if FULL_EVAL_FRAC < 1.0:
            rs = np.random.RandomState(GLOBAL_SEED)
            n_eval = max(1, int(n * FULL_EVAL_FRAC))
            eval_rows = np.sort(rs.choice(np.arange(n), size=n_eval, replace=False))
            row_iter = (
                (s, min(s + EVAL_BLOCK, len(eval_rows)))
                for s in range(0, len(eval_rows), EVAL_BLOCK)
            )
            mode_subset = True
        else:
            row_iter = _row_blocks(n, EVAL_BLOCK)
            mode_subset = False

        pred_sse_acc = 0.0
        cross_acc = 0.0

        for blk in row_iter:
            if mode_subset:
                s0, e0 = blk
                rows_blk = eval_rows[s0:e0]
                Xv = X_train[rows_blk, :]
            else:
                s0, e0 = blk
                Xv = X_train[s0:e0, :]

            Hv = (Xv @ W.T)
            HWv = (Hv @ W)
            pred_sse_acc += float(to_cpu((HWv * HWv).sum()))
            Xv_dense = Xv.toarray().astype(xp.float32, copy=False)
            cross_acc += float(to_cpu((HWv * Xv_dense).sum()))

            del Hv, HWv, Xv_dense, Xv
            if _has_cupy:
                cp.get_default_memory_pool().free_all_blocks()

        if FULL_EVAL_FRAC < 1.0:
            scale = float(n) / float(len(eval_rows))
            pred_sse_acc *= scale
            cross_acc *= scale

        sse = x_sse + pred_sse_acc - 2.0 * cross_acc
        reg_full = float(to_cpu(lambda_W * (W * W).sum()))
        full_total = sse + reg_full

        if (ep % print_every) == 0:
            print(f"[ep {ep:03d}] avg_batch_loss={avg_batch_loss:.6e}  full={full_total:.6e}")

        if full_total < best_total - 1e-6:
            best_total, best_W, best_ep = full_total, W.copy(), ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[early stop] ep={ep}, best_ep={best_ep}, best_full={best_total:.6e}")
                break

    print(f"✅ Done. Best epoch = {best_ep}, best full loss = {best_total:.6e}")

    # Compute H on CPU + save results-only 
    W_best = to_cpu(best_W).astype(np.float32, copy=False)          # (k,p)
    H_shared = (X_cpu @ W_best.T).astype(np.float32, copy=False)    # (n,k)

    keep_obs = [c for c in [donor_col, celltype_col] if c in adata.obs.columns]
    obs_out = adata.obs[keep_obs].copy() if keep_obs else adata.obs.copy()

    var_out = pd.DataFrame(index=adata.var_names)

    res = ad.AnnData(X=None, obs=obs_out, var=var_out)
    res.obsm[f"H_shared_k{k}"] = H_shared
    res.varm[f"W_tied_k{k}"] = W_best.T  # (genes, k)

    res.uns["model_name"] = "AE_NMF_no_cond_tied"
    res.uns["k_rank"] = int(k)
    res.uns["lambda_W"] = float(lambda_W)
    res.uns["lrW"] = float(lrW)
    res.uns["libsize_log1p"] = bool(libsize_log1p)
    res.uns["GLOBAL_SEED"] = int(GLOBAL_SEED)
    res.uns["row_norm"] = bool(row_norm)
    res.uns["use_hvg"] = bool(use_hvg)

    res.write_h5ad(OUT_PATH, compression="gzip")
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
