#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file runner for 2nd model (HU) with ONE condition: donor_id
Writes a small "results-only" .h5ad for later ARI/NMI + cell-type classifier.

Outputs:
  obsm["H_shared_k{k}"]
  obsm["H_cond_latent_k{k}"]
  obsm["H_concat_full_k{k}"]
  varm["W_concat_k{k}"]
  (optional) uns["U_list_k{k}"] 
  uns["cond_meta_concat_k{k}"]
  obs includes donor_id + cell_type 
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse as sp


# Global seed / backend
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)
rng_np = np.random.default_rng(GLOBAL_SEED)

USING_CUPY = os.getenv("USE_CUPY", "0") == "1"
if USING_CUPY:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
        cp.cuda.Device(0).use()
        cp.random.seed(GLOBAL_SEED)
        # warm-up GEMM
        _ = (cp.ones((2, 2), cp.float32) @ cp.ones((2, 2), cp.float32))
        xp = cp
        sps = cpx_sparse
        print("[backend] Using CuPy + cupyx.scipy.sparse")
    except Exception as e:
        print(f"[backend] CuPy requested but unavailable ({e}); using CPU.")
        USING_CUPY = False

if not USING_CUPY:
    xp = np
    sps = sp
    print("[backend] Using NumPy + SciPy (CPU)")

def to_xp(a):
    if USING_CUPY:
        import cupy as cp
        return cp.asarray(a)
    return a

def to_np(a):
    if USING_CUPY:
        import cupy as cp
        return cp.asnumpy(a)
    return a

# Config 
H5AD_PATH = os.getenv(
    "H5AD_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad",
)
OUT_PATH = os.getenv(
    "OUT_PATH",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_2nd_model_donorOnly_k80.h5ad",
)

# condition + labels
cond_cols = ["donor_id"]  
celltype_col = os.getenv("CT_COL", "cell_type")

# preprocessing
use_hvg = os.getenv("USE_HVG", "0") == "1"
n_hvg = int(os.getenv("N_HVG", "10000"))
libsize_log1p = os.getenv("LIBSIZE_LOG1P", "1") == "1"

# training knobs
k_rank = int(os.getenv("K", "80"))
epochs = int(os.getenv("EPOCHS", "100"))
batch_size = int(os.getenv("BATCH_SIZE", "512"))

lambda_W = float(os.getenv("LAMBDA_W", "1e-2"))
lambda_U = float(os.getenv("LAMBDA_U", "5e-2"))  
lrW = float(os.getenv("LR_W", "1e-3"))
lrU = float(os.getenv("LR_U", "5e-4"))
gamma_inv = float(os.getenv("GAMMA_INV", "5e-3"))
eta_hsic = float(os.getenv("ETA_HSIC", "2e-3"))

full_eval_frac = float(os.getenv("FULL_EVAL_FRAC", "0.05"))
EVAL_BLOCK = int(os.getenv("EVAL_BLOCK", "2000"))
SAVE_BLOCK = int(os.getenv("SAVE_BLOCK", str(EVAL_BLOCK)))

early_stop = os.getenv("EARLY_STOP", "1") == "1"
patience = int(os.getenv("PATIENCE", "6"))
print_every = int(os.getenv("PRINT_EVERY", "1"))

# keep or skip U in output 
SAVE_U = os.getenv("SAVE_U", "0") == "1"

def libsize_log1p_transform(X_csr, target_sum=1e4):
    rs = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    scale[nz] = target_sum / rs[nz]
    Xn = X_csr.multiply(scale[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data).astype(np.float32, copy=False)
    return Xn

def iterate_minibatches_indices(n, batch_size, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s : min(s + batch_size, n)]

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
                out.extend(pool[ptr : ptr + take])
                ptrs[g] = ptr + take
                added += take
            if len(out) >= batch_size:
                yield np.array(out)
                out = []
        if added == 0:
            if out:
                yield np.array(out)
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
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        P -= self.lr * mhat / (xp.sqrt(vhat) + self.eps)
        return P

def train_for_k(adata, X_csr, cond_info, has_celltype: bool):
    n, p = X_csr.shape
    m = cond_info[0]["m"]

    print(f"[train] n={n} p={p} m={m} k={k_rank} gpu={USING_CUPY}")

    # init
    W_host = rng_np.random((k_rank, p), dtype=np.float32)
    W_host /= np.linalg.norm(W_host, axis=1, keepdims=True) + 1e-12
    U_host = 0.01 * rng_np.random((m, k_rank, p), dtype=np.float32)

    W = to_xp(W_host)
    U = to_xp(U_host)

    optW = AdamMat(W.shape, lr=lrW)
    optU = AdamMat(U.shape, lr=lrU)

    best_total = np.inf
    best_W = W.copy()
    best_U = U.copy()
    best_ep = 0
    bad = 0

    use_stratified = has_celltype  # stratify by cell_type only 
    STRATIFY_COLS = [celltype_col]

    codes_all = cond_info[0]["codes"]  

    for ep in range(1, epochs + 1):
        running = 0.0
        steps = 0

        if use_stratified:
            it = iterate_minibatches_stratified(
                adata.obs.reset_index(drop=True),
                STRATIFY_COLS,
                batch_size,
                rng_np,
            )
        else:
            it = iterate_minibatches_indices(n, batch_size, rng_np)

        for rows in it:
            Xb_host = X_csr[rows, :]
            B = Xb_host.shape[0]

            if USING_CUPY:
                import cupy as cp
                Xb = sps.csr_matrix(Xb_host)
                Hb = (Xb @ W.T).astype(xp.float32)              # (B,k)
                HW = Hb @ W                                    # (B,p)

                # HU = Hb @ U[donor(rows)]
                codes_b = codes_all[rows]                      # numpy (B,)
                HU = xp.zeros((B, p), dtype=xp.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if not sel.any():
                        continue
                    idx = cp.asarray(np.where(sel)[0])
                    Hb_c = cp.ascontiguousarray(Hb[idx, :], dtype=cp.float32)
                    Uc   = cp.ascontiguousarray(U[int(c), :, :], dtype=cp.float32)
                    HU[idx, :] += Hb_c @ Uc

                Pred = HW + HU
                Xb_dense = xp.asarray(Xb_host.toarray(), dtype=xp.float32)
                E = Pred - Xb_dense

                data_loss = float((E * E).sum() / max(1, B))
                reg_loss = float(lambda_W * (W * W).sum() + lambda_U * (U * U).sum())
                Lb = data_loss + reg_loss
                running += Lb
                steps += 1

                # grads
                # gU per donor
                gU = xp.zeros_like(U, dtype=xp.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if not sel.any():
                        continue
                    idx = cp.asarray(np.where(sel)[0])
                    Hb_c = cp.ascontiguousarray(Hb[idx, :], dtype=cp.float32)
                    E_c  = cp.ascontiguousarray(E[idx, :],  dtype=cp.float32)
                    gU[int(c)] = (2.0 / max(1, B)) * (Hb_c.T @ E_c)
                gU += 2.0 * lambda_U * U

                # gW (simplified, matches HU objective structure)
                HtE = (Hb.T @ E).astype(xp.float32)
                dLdH = (E @ W.T).astype(xp.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if not sel.any():
                        continue
                    idx = cp.asarray(np.where(sel)[0])
                    E_c = cp.ascontiguousarray(E[idx, :], dtype=cp.float32)
                    UcT = cp.ascontiguousarray(U[int(c)].T, dtype=cp.float32)
                    dLdH[idx, :] += E_c @ UcT
                cross = (Xb_dense.T @ dLdH).T.astype(xp.float32)
                gW = (2.0 / max(1, B)) * (HtE + cross) + 2.0 * lambda_W * W

                W = optW.step(W, gW)
                U = optU.step(U, gU)

                W = xp.maximum(W, 0.0)
                U = xp.maximum(U, 0.0)
                W /= xp.linalg.norm(W, axis=1, keepdims=True) + 1e-12

            else:
                # CPU version 
                Xb = Xb_host
                Xb_dense = Xb.toarray().astype(np.float32)

                Hb = (Xb @ W.T).astype(np.float32)
                HW = Hb @ W

                codes_b = codes_all[rows]
                HU = np.zeros((B, p), dtype=np.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if sel.any():
                        HU[sel, :] += Hb[sel, :] @ U[int(c), :, :]

                Pred = HW + HU
                E = Pred - Xb_dense

                data_loss = float((E * E).sum() / max(1, B))
                reg_loss = float(lambda_W * (W * W).sum() + lambda_U * (U * U).sum())
                Lb = data_loss + reg_loss
                running += Lb
                steps += 1

                gU = np.zeros_like(U, dtype=np.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if sel.any():
                        gU[int(c)] = (2.0 / max(1, B)) * (Hb[sel].T @ E[sel])
                gU += 2.0 * lambda_U * U

                HtE = (Hb.T @ E).astype(np.float32)
                dLdH = (E @ W.T).astype(np.float32)
                for c in np.unique(codes_b):
                    sel = (codes_b == c)
                    if sel.any():
                        dLdH[sel] += (E[sel] @ U[int(c)].T).astype(np.float32)
                cross = (Xb_dense.T @ dLdH).T.astype(np.float32)
                gW = (2.0 / max(1, B)) * (HtE + cross) + 2.0 * lambda_W * W

                W = optW.step(W, gW)
                U = optU.step(U, gU)

                W = np.maximum(W, 0.0)
                U = np.maximum(U, 0.0)
                W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12

        avg_batch = running / max(1, steps)

        # subset eval 
        m_eval = max(10000, int(full_eval_frac * n))
        idx_eval = np.sort(rng_np.choice(n, size=min(m_eval, n), replace=False))

        # compute sse on subset 
        W_np = to_np(W).astype(np.float32)
        U_np = to_np(U).astype(np.float32)

        x_sse = float((X_csr[idx_eval, :].power(2)).sum())
        pred_sse = 0.0
        cross = 0.0

        H_eval = (X_csr[idx_eval, :] @ W_np.T).astype(np.float32)
        codes_eval = codes_all[idx_eval]

        for s in range(0, len(idx_eval), EVAL_BLOCK):
            e = min(s + EVAL_BLOCK, len(idx_eval))
            Hb = H_eval[s:e, :]
            HW = Hb @ W_np
            HU = np.zeros_like(HW, dtype=np.float32)

            codes_blk = codes_eval[s:e]
            for c in np.unique(codes_blk):
                sel = (codes_blk == c)
                if sel.any():
                    HU[sel, :] += Hb[sel, :] @ U_np[int(c), :, :]

            Pred = HW + HU
            Xblk = X_csr[idx_eval[s:e], :].toarray().astype(np.float32)
            pred_sse += float((Pred * Pred).sum())
            cross += float((Pred * Xblk).sum())

        sse = pred_sse + x_sse - 2.0 * cross
        reg = float(lambda_W * (W_np * W_np).sum() + lambda_U * (U_np * U_np).sum())
        full_total = sse + reg

        if ep % print_every == 0:
            print(f"[ep {ep:03d}] avg_batch_loss={avg_batch:.6f}  full_sse_norm={full_total / max(1.0, x_sse):.6f}")

        if full_total < best_total - 1e-6:
            best_total = full_total
            best_W = to_xp(W_np).copy()
            best_U = to_xp(U_np).copy()
            best_ep = ep
            bad = 0
        else:
            bad += 1
            if early_stop and bad >= patience:
                print(f"[train] Early stop at ep={ep}, best_ep={best_ep}")
                break

    return to_np(best_W).astype(np.float32), to_np(best_U).astype(np.float32), best_ep, float(best_total)

def main():
    print("[cfg] H5AD_PATH:", H5AD_PATH)
    print("[cfg] OUT_PATH :", OUT_PATH)
    print("[cfg] cond_cols:", cond_cols)
    print("[cfg] k        :", k_rank)

    adata = ad.read_h5ad(H5AD_PATH)

    # pick layer (optional)
    layer = os.getenv("LAYER_FOR_NMF", "none")
    if layer != "none" and layer in adata.layers:
        X = adata.layers[layer]
        print("[info] Using layer:", layer)
    else:
        X = adata.X
        print("[info] Using adata.X")

    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # optional HVG
    if use_hvg:
        tmp = ad.AnnData(X.copy(), var=adata.var.copy(), obs=adata.obs[[]].copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=n_hvg)
        hvg_mask = tmp.var["highly_variable"].to_numpy()
        adata = adata[:, hvg_mask].copy()
        X = X[:, hvg_mask]
        print("[info] HVG genes kept:", int(hvg_mask.sum()))

    # CPU normalize/log1p
    if libsize_log1p:
        X = libsize_log1p_transform(X)

    # check required cols
    for col in cond_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing obs column: {col}")

    has_celltype = celltype_col in adata.obs.columns
    if not has_celltype:
        print(f"[warn] {celltype_col} not found; stratified batching disabled.")

    # encode donor_id
    cat = pd.Categorical(adata.obs["donor_id"])
    codes = cat.codes.astype(np.int64)
    if np.any(codes < 0):
        keep = np.where(codes >= 0)[0]
        adata = adata[keep, :].copy()
        X = X[keep, :]
        cat = pd.Categorical(adata.obs["donor_id"])
        codes = cat.codes.astype(np.int64)

    cond_info = [{
        "name": "donor_id",
        "codes": codes,
        "categories": list(cat.categories),
        "m": len(cat.categories),
    }]

    print("[info] n_obs:", adata.n_obs, "n_vars:", adata.n_vars, "m_donors:", cond_info[0]["m"])

    # train
    Wc, Uc, best_ep, best_total = train_for_k(adata, X, cond_info, has_celltype=has_celltype)

    # compute H matrices (CPU)
    H_shared = (X @ Wc.T).astype(np.float32)

    H_cond_latent = np.zeros((X.shape[0], k_rank), dtype=np.float32)
    for s in range(0, X.shape[0], SAVE_BLOCK):
        e = min(s + SAVE_BLOCK, X.shape[0])
        Hb = (X[s:e, :] @ Wc.T).astype(np.float32)
        codes_blk = codes[s:e]
        contrib = np.zeros_like(Hb)
        for c in np.unique(codes_blk):
            sel = (codes_blk == c)
            if sel.any():
                contrib[sel, :] += np.einsum(
                    "bk,kp,pk->bk",
                    Hb[sel, :],
                    Uc[int(c), :, :],
                    Wc.T,
                    optimize=True,
                ).astype(np.float32)
        H_cond_latent[s:e, :] = contrib

    H_concat = np.concatenate([H_shared, H_cond_latent], axis=1).astype(np.float32)

    # results-only AnnData 
    keep_obs_cols = [c for c in ["donor_id", celltype_col] if c in adata.obs.columns]
    obs_small = adata.obs[keep_obs_cols].copy()
    var_small = pd.DataFrame(index=adata.var_names)  # keep only gene names

    res = ad.AnnData(X=None, obs=obs_small, var=var_small)
    res.obsm[f"H_shared_k{k_rank}"] = H_shared
    res.obsm[f"H_cond_latent_k{k_rank}"] = H_cond_latent
    res.obsm[f"H_concat_full_k{k_rank}"] = H_concat

    res.varm[f"W_concat_k{k_rank}"] = Wc.T

    cond_meta_small = {
        "cond_cols": ["donor_id"],
        "cond_categories": {"donor_id": cond_info[0]["categories"]},
        "m_donors": int(cond_info[0]["m"]),
        "k": int(k_rank),
        "best_epoch": int(best_ep),
        "best_full_loss": float(best_total),
        "lambda_W": float(lambda_W),
        "lambda_U": float(lambda_U),
        "gamma_inv": float(gamma_inv),
        "eta_hsic": float(eta_hsic),
        "note": "2nd model HU; Xhat = H W + H U[donor] (single condition donor_id).",
    }
    res.uns[f"cond_meta_concat_k{k_rank}"] = cond_meta_small
    res.uns["model_name"] = "CNMF_2nd_donorOnly"

    if SAVE_U:
        # Store U in .uns 
        res.uns[f"U_donor_k{k_rank}"] = Uc

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    res.write_h5ad(OUT_PATH)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
