import os
import numpy as np
import pandas as pd
from scipy import sparse as sp

# Global config / backend
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)

USE_GPU = os.getenv("USE_CUPY", "1") == "1"
_has_cupy = False
cp = None
cpx_sp = None

try:
    if USE_GPU:
        import cupy as _cp
        from cupyx.scipy import sparse as _cpx_sp

        cp = _cp
        cpx_sp = _cpx_sp
        _has_cupy = True
        cp.random.seed(GLOBAL_SEED)
except Exception:
    _has_cupy = False
    cp = None
    cpx_sp = None

xp = cp if _has_cupy else np
spx = cpx_sp if _has_cupy else sp


def to_cpu(a):
    """Convert cupy array -> numpy if needed."""
    if _has_cupy and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def to_gpu(a):
    """Convert numpy array -> cupy if possible."""
    if not _has_cupy:
        return a
    if isinstance(a, np.ndarray):
        return cp.asarray(a)
    return a


def csr_to_gpu(A_csr):
    """Convert scipy CSR -> cupyx CSR if needed."""
    if not _has_cupy:
        return A_csr
    if sp.isspmatrix_csr(A_csr):
        return spx.csr_matrix(
            (cp.asarray(A_csr.data),
             cp.asarray(A_csr.indices),
             cp.asarray(A_csr.indptr)),
            shape=A_csr.shape,
        )
    elif isinstance(A_csr, spx.csr_matrix):
        return A_csr
    else:
        B = sp.csr_matrix(A_csr)
        return csr_to_gpu(B)


def csr_to_cpu(A_csr):
    """Convert cupyx CSR -> scipy CSR if needed."""
    if _has_cupy and isinstance(A_csr, spx.csr_matrix):
        return sp.csr_matrix(
            (cp.asnumpy(A_csr.data),
             cp.asnumpy(A_csr.indices),
             cp.asnumpy(A_csr.indptr)),
            shape=A_csr.shape,
        )
    return A_csr


def _libsize_log1p_cpu(X_csr, target_sum=1e4):
    """CPU normalize_total + log1p for CSR."""
    rs = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    scale[nz] = target_sum / rs[nz]
    Xn = X_csr.multiply(scale[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data).astype(np.float32, copy=False)
    return Xn


def libsize_log1p_transform_csr(X_csr, target_sum=1e4):
    """
    Normalize per-cell library size to target_sum and apply log1p.

    IMPORTANT:
    - For very large nnz, cupyx CSR multiply can OOM.
    - Use CPU_NORM_LOG1P=1 to force CPU preprocessing even when training on GPU.
    """
    force_cpu = os.getenv("CPU_NORM_LOG1P", "0") == "1"
    nnz_limit = int(os.getenv("GPU_SPARSE_NNZ_LIMIT", "200000000"))  
    nnz = int(to_cpu(X_csr.nnz)) if _has_cupy else int(X_csr.nnz)
    if (not _has_cupy) or force_cpu or (nnz > nnz_limit):
        X_cpu = csr_to_cpu(X_csr)
        Xn_cpu = _libsize_log1p_cpu(X_cpu, target_sum=target_sum)
        return csr_to_gpu(Xn_cpu) if _has_cupy else Xn_cpu

    # GPU path
    rs = X_csr.sum(axis=1)
    rs = xp.asarray(rs).ravel()
    scale = xp.zeros_like(rs, dtype=xp.float32)
    nz = rs > 0
    scale[nz] = target_sum / rs[nz]
    Xn = X_csr.multiply(scale[:, None])
    Xn.data = xp.log1p(Xn.data).astype(xp.float32, copy=False)
    return Xn


def iterate_minibatches_indices(n, batch_size, rng):
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    for s in range(0, int(n), batch_size):
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


def make_C_from_codes(codes, m):
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    rows = np.arange(n, dtype=np.int64)
    cols = codes
    data = np.ones(n, dtype=np.float32)
    C = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
    return csr_to_gpu(C) if _has_cupy else C


class AdamMat:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
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


# Preprocessing & conditions
def prepare_X_and_conditions(
    adata,
    cond_cols,
    celltype_col="cell_type",
    layer_name="decontXcounts",
    use_hvg=False,
    n_hvg=10000,
    libsize_log1p=True,
):
    import anndata as ad
    import scanpy as sc

    assert isinstance(adata, ad.AnnData)

    # pick layer or X
    if layer_name is not None and layer_name in adata.layers:
        X = adata.layers[layer_name]
        print(f"[core] Using layer: {layer_name}")
    elif "X_original" in adata.layers:
        X = adata.layers["X_original"]
        print("[core] Using layer: X_original")
    else:
        X = adata.X
        print("[core] Using adata.X (no counts layers found).")

    # Ensure CSR on CPU first
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # Optional HVG (CPU side)
    if use_hvg:
        Atmp = ad.AnnData(X.copy(), var=adata.var.copy(), obs=adata.obs[[]].copy())
        sc.pp.normalize_total(Atmp, target_sum=1e4)
        sc.pp.log1p(Atmp)
        sc.pp.highly_variable_genes(Atmp, flavor="seurat_v3", n_top_genes=n_hvg)
        hvg_mask = Atmp.var["highly_variable"].to_numpy()
        X = X[:, hvg_mask]
        adata = adata[:, hvg_mask].copy()
        print(f"[core] HVG retained genes: {hvg_mask.sum()}")

    # normalize+log1p (CPU by default)
    if libsize_log1p:
        X = libsize_log1p_transform_csr(X)

    # move to GPU
    X = csr_to_gpu(X) if _has_cupy else X

    nnz = int(to_cpu(X.nnz)) if _has_cupy else int(X.nnz)
    print(f"[core] X shape after prep: {X.shape}  nnz={nnz}")

    need_cols = list(cond_cols) + [celltype_col]
    for col in need_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing required obs column: {col}")

    cond_info = {}
    codes_list = []
    sizes = []
    offset = 0

    for col in cond_cols:
        cat = pd.Categorical(adata.obs[col])
        codes = cat.codes.astype(np.int64)
        valid = (codes >= 0)
        if not np.all(valid):
            keep = np.where(valid)[0]
            adata = adata[keep, :].copy()
            X = X[keep, :]
            cat = pd.Categorical(adata.obs[col])
            codes = cat.codes.astype(np.int64)

        cats = list(cat.categories)
        m_i = len(cats)
        cond_info[col] = {
            "codes": codes,
            "categories": cats,
            "size": m_i,
            "offset": offset,
        }
        codes_list.append(codes)
        sizes.append(m_i)
        offset += m_i

    m_total = int(np.sum(sizes))
    print(
        f"[core] X: {X.shape}   total cond dims m_total={m_total}   blocks="
        f"{ {c: cond_info[c]['size'] for c in cond_info} }"
    )

    # Build global C = [C_cond1 | C_cond2 | ...]
    C_blocks = [make_C_from_codes(codes_list[i], sizes[i]) for i in range(len(sizes))]
    C_global = spx.hstack(C_blocks, format="csr")
    C_global_cpu = csr_to_cpu(C_global)

    return adata, X, cond_info, C_global, C_global_cpu


# Training
def train_cond_nmf_for_k(
    X,
    C_global,
    cond_info,
    celltype_col="cell_type",
    k_rank=80,
    epochs=40,
    batch_size=512,
    lambda_W=1e-2,
    lambda_U=1e-1,
    gamma_inv_base=1e-3,
    eta_hsic_base=5e-4,
    warmup_epochs=5,
    nonneg_W=True,
    nonneg_U=True,
    row_norm=True,
    early_stop=True,
    patience=6,
    full_eval_frac=1.0,
    eval_block=2000,
    seed_offset=42,
    print_every=1,
    adata_obs=None,
    use_stratified=True,
    stratify_cols=None,
):
    if adata_obs is None:
        raise ValueError("adata_obs (pandas DataFrame) is required for training.")

    n, p = X.shape
    m_total = int(np.sum([info["size"] for info in cond_info.values()]))

    has_celltype = (celltype_col in adata_obs.columns)
    use_str = use_stratified and has_celltype

    if stratify_cols is None:
        stratify_cols = [celltype_col]
    for c in stratify_cols:
        if c not in adata_obs.columns:
            raise ValueError(f"Stratify column '{c}' not in adata.obs.")

    rng = np.random.default_rng(seed_offset + k_rank)

    W = xp.asarray(rng.random((k_rank, p)), dtype=xp.float32)
    W /= (xp.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    U = xp.asarray(0.01 * rng.random((m_total, p)), dtype=xp.float32)

    optW = AdamMat(W.shape, lr=1e-3)
    optU = AdamMat(U.shape, lr=5e-4)

    epoch_losses = []
    full_losses = []
    sse_losses = []

    best_total = np.inf
    best_W = W.copy()
    best_U = U.copy()
    best_ep = 0
    bad = 0

    # Precompute x_sse safely 
    if _has_cupy:
        x_sse = float(to_cpu(cp.dot(X.data, X.data)))
    else:
        x_sse = float(np.dot(X.data, X.data))

    # Also keep a CPU copy of C_global for the full-data G construction
    C_global_cpu = csr_to_cpu(C_global)

    print(f"[core] Start training k={k_rank}  n={n}  p={p}  m_total={m_total}  gpu={_has_cupy}")

    for ep in range(1, epochs + 1):
        gamma_inv = 0.0 if ep <= warmup_epochs else gamma_inv_base
        eta_hsic = 0.0 if ep <= warmup_epochs else eta_hsic_base

        running = 0.0
        steps = 0

        if use_str:
            it = iterate_minibatches_stratified(
                adata_obs.reset_index(drop=True),
                stratify_cols,
                batch_size,
                rng,
            )
        else:
            it = iterate_minibatches_indices(n, batch_size, rng)

        for rows in it:
            rows = np.asarray(rows, dtype=int)

            rows_backend = xp.asarray(rows) if _has_cupy else rows

            Xb = X[rows_backend, :]
            B = int(Xb.shape[0])

            # per-condition one-hots for these rows 
            Cb = spx.hstack(
                [
                    make_C_from_codes(
                        cond_info[c]["codes"][rows].astype(np.int64),
                        cond_info[c]["size"],
                    )
                    for c in cond_info.keys()
                ],
                format="csr",
            )

            Hb = (Xb @ W.T)          # (B,k)
            HW = (Hb @ W)            # (B,p)
            CU = (Cb @ U)            # (B,p)

            Xb_dense = Xb.toarray().astype(xp.float32, copy=False)
            E = (HW + CU) - Xb_dense

            data_loss = float(to_cpu((E * E).sum()) / max(1, B))
            reg_loss = float(to_cpu(lambda_W * (W * W).sum() + lambda_U * (U * U).sum()))

            # invariance penalty
            inv_loss = 0.0
            gW_inv = xp.zeros_like(W, dtype=xp.float32)

            if gamma_inv > 0.0 and has_celltype:
                ct_b = adata_obs[celltype_col].iloc[rows].astype(str).to_numpy()

                for ct in np.unique(ct_b):
                    sel = (ct_b == ct)
                    if sel.sum() < 2:
                        continue

                    sel_backend = xp.asarray(sel) if _has_cupy else sel

                    Cb_ct = Cb[sel_backend, :]
                    Hb_ct = Hb[sel_backend, :]

                    G = (Cb_ct.T @ Cb_ct).toarray()
                    G = xp.asarray(G, dtype=xp.float32)
                    G_inv = xp.linalg.pinv(G + 1e-8 * xp.eye(G.shape[0], dtype=xp.float32))

                    PcHb = (Cb_ct @ (G_inv @ (Cb_ct.T @ Hb_ct)))
                    inv_loss += float(to_cpu((PcHb * PcHb).sum()))

                    Xb_sel_dense = Xb[sel_backend, :].toarray().astype(xp.float32, copy=False)
                    gW_inv += (2.0 * gamma_inv / max(1, B)) * (PcHb.T @ Xb_sel_dense)

                inv_loss = (gamma_inv / max(1, B)) * inv_loss

            # HSIC penalty
            hsic_loss = 0.0
            gW_hsic = xp.zeros_like(W, dtype=xp.float32)
            if eta_hsic > 0.0:
                Cb_dense = xp.asarray(Cb.toarray(), dtype=xp.float32)
                Hb_c = Hb - Hb.mean(axis=0, keepdims=True)
                Cc = Cb_dense - Cb_dense.mean(axis=0, keepdims=True)
                M = Hb_c.T @ Cc
                hsic_loss = float(to_cpu((M * M).sum()) / max(1, B))
                S = (Cc @ Cc.T)
                gH = (2.0 * eta_hsic / max(1, B)) * (S @ Hb_c)
                gW_hsic = (gH.T @ Xb.toarray())

            Lb = data_loss + reg_loss + inv_loss + hsic_loss
            running += Lb
            steps += 1

            # gradients
            gU = (2.0 * (Cb.T @ E)) / max(1, B) + 2.0 * lambda_U * U
            HtE = (Hb.T @ E)
            EWt = (E @ W.T)
            XtEWt = (Xb.T @ EWt)
            cross = XtEWt.T
            gW = (2.0 * (HtE + cross)) / max(1, B) + 2.0 * lambda_W * W
            gW += gW_inv + gW_hsic

            W = optW.step(W, gW)
            U = optU.step(U, gU)

            if nonneg_W:
                W = xp.maximum(W, 0.0)
            if nonneg_U:
                U = xp.maximum(U, 0.0)
            if row_norm:
                W /= (xp.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

            # free batch tensors
            del Hb, HW, CU, Xb_dense, E, HtE, EWt, XtEWt, cross, Cb
            if _has_cupy:
                cp.get_default_memory_pool().free_all_blocks()

        epoch_losses.append(running / max(1, steps))

        # full objective (streamed) 
        if full_eval_frac < 1.0:
            rs = np.random.RandomState(seed_offset + k_rank)
            n_eval = max(1, int(int(n) * float(full_eval_frac)))
            eval_rows = np.sort(rs.choice(np.arange(int(n)), size=n_eval, replace=False))
            row_iter = ((s, min(s + eval_block, len(eval_rows))) for s in range(0, len(eval_rows), eval_block))
            mode_subset = True
        else:
            row_iter = _row_blocks(int(n), eval_block)
            mode_subset = False

        pred_sse_acc = 0.0
        cross_acc = 0.0

        for blk in row_iter:
            if mode_subset:
                s0, e0 = blk
                rows_blk_cpu = eval_rows[s0:e0]
                rows_blk = xp.asarray(rows_blk_cpu) if _has_cupy else rows_blk_cpu
                Xb = X[rows_blk, :]
                Cb = C_global[rows_blk, :]
            else:
                s0, e0 = blk
                Xb = X[s0:e0, :]
                Cb = C_global[s0:e0, :]

            Hb = (Xb @ W.T)
            HWb = (Hb @ W)
            CUb = (Cb @ U)
            Pb = HWb + CUb

            pred_sse_acc += float(to_cpu((Pb * Pb).sum()))
            Xb_dense = Xb.toarray().astype(xp.float32, copy=False)
            cross_acc += float(to_cpu((Pb * Xb_dense).sum()))

            del Hb, HWb, CUb, Pb, Xb_dense, Xb, Cb
            if _has_cupy:
                cp.get_default_memory_pool().free_all_blocks()

        sse = x_sse + pred_sse_acc - 2.0 * cross_acc

        # invariance full-data term
        full_inv = 0.0
        if gamma_inv_base > 0.0 and has_celltype:
            ct_all = adata_obs[celltype_col].astype(str).to_numpy()

            # Build G on CPU
            G = np.zeros((m_total, m_total), dtype=np.float32)
            for s0, e0 in _row_blocks(int(n), eval_block):
                Cb_cpu = C_global_cpu[s0:e0, :]
                G += (Cb_cpu.T @ Cb_cpu).toarray().astype(np.float32)

            G = xp.asarray(G, dtype=xp.float32)
            G_inv = xp.linalg.pinv(G + 1e-8 * xp.eye(G.shape[0], dtype=xp.float32))

            for ct in np.unique(ct_all):
                sel = np.where(ct_all == ct)[0]
                if sel.size < 2:
                    continue
                for s in range(0, len(sel), eval_block):
                    rows_blk_cpu = sel[s:s + eval_block]
                    rows_blk = xp.asarray(rows_blk_cpu) if _has_cupy else rows_blk_cpu
                    Xb = X[rows_blk, :]
                    Hb = (Xb @ W.T)
                    Cb = C_global[rows_blk, :]
                    PcHb = (Cb @ (G_inv @ (Cb.T @ Hb)))
                    full_inv += float(to_cpu((PcHb * PcHb).sum()))
                    del Xb, Hb, Cb, PcHb
                    if _has_cupy:
                        cp.get_default_memory_pool().free_all_blocks()

            full_inv = float(gamma_inv_base) * full_inv

        reg_loss_full = float(to_cpu(lambda_W * (W * W).sum() + lambda_U * (U * U).sum()))
        full_total = sse + reg_loss_full + full_inv
        sse_losses.append(sse)
        full_losses.append(full_total)

        if (ep % print_every) == 0:
            print(
                f"[core][ep {ep:03d}] avg_batch_loss={epoch_losses[-1]:.6f}  "
                f"full_sse_norm={(sse / max(1.0, x_sse)):.6f}"
            )

        if full_total < best_total - 1e-6:
            best_total = full_total
            best_W = W.copy()
            best_U = U.copy()
            best_ep = ep
            bad = 0
        else:
            bad += 1
            if early_stop and bad >= patience:
                print(f"[core] Early stop at ep={ep}, best_ep={best_ep}, best_full={best_total / max(1.0, x_sse):.6f}")
                break

    print(f"✅ CNMF_1st finished. Best epoch = {best_ep}, best loss = {best_total / max(1.0, x_sse):.3e}")
    return best_W, best_U, best_ep, (best_total / max(1.0, x_sse))



# H computation
def compute_H_matrices(X_cpu, C_global_cpu, Wc, Uc):
    Wc = np.asarray(Wc, dtype=np.float32)
    Uc = np.asarray(Uc, dtype=np.float32)

    H_shared = (X_cpu @ Wc.T).astype(np.float32)

    # memory-safe: (m_total x p) @ (p x k) -> (m_total x k), then (n x m_total) @ -> (n x k)
    UWt = (Uc @ Wc.T).astype(np.float32)
    H_cond_latent = (C_global_cpu @ UWt).astype(np.float32)

    H_concat = np.concatenate([H_shared, H_cond_latent], axis=1).astype(np.float32)
    return H_shared, H_cond_latent, H_concat
