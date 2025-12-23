#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute clustering agreement (ARI/NMI) for multiple models using Leiden clustering
on each model's embedding, then score against two labelings:
  - cell_type
  - donor_id

Outputs two CSV files with columns:
  Model, ARI, ARI Std, NMI, NMI Std

Concept:
  embedding -> neighbors graph -> Leiden repeats -> ARI/NMI
"""

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Config 
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)


LABEL_KEYS = ["cell_type", "donor_id"]


OUT_CELLTYPE = os.getenv(
    "METRICS_OUT_CELLTYPE",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/clustering_metrics_by_cell_type_k80.csv",
)
OUT_DONOR = os.getenv(
    "METRICS_OUT_DONOR",
    "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/clustering_metrics_by_donor_id_k80.csv",
)

# Leiden / neighbors config
RESOLUTION = float(os.getenv("LEIDEN_RES", "0.8"))
N_NEIGH = int(os.getenv("N_NEIGH", "25"))
USE_WEIGHTS = os.getenv("USE_WEIGHTS", "0") == "1"
LEIDEN_REPEATS = int(os.getenv("LEIDEN_REPEATS", "10"))
LEIDEN_SEEDS = list(range(LEIDEN_REPEATS))
METRIC = os.getenv("NN_METRIC", "euclidean")   # scanpy neighbors metric
ZSCORE = os.getenv("ZSCORE_EMB", "1") == "1"   # zscore columns before neighbors


MODELS = [
    {
        "name": "CAE_NMF1",
        "h5ad_path": os.getenv(
            "H5AD_CNM1",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_1st_donor_only_k80.h5ad",
        ),
        
        "emb_keys_try": ["H_shared_k80"],
    },
    {
        "name": "CAE_NMF2",
        "h5ad_path": os.getenv(
            "H5AD_CNM2",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_2nd_donor_only_k80.h5ad",
        ),
        "emb_keys_try": ["H_shared_k80"],
    },
    {
        "name": "AE_NMF",
        "h5ad_path": os.getenv(
            "H5AD_AE",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_no_cond_k80.h5ad",
        ),
        "emb_keys_try": ["H_shared_k80"],
    },
    {
        "name": "Base_NMF",
        "h5ad_path": os.getenv(
            "H5AD_BASE",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_sklearn_nmf_k80.h5ad",
        ),
        "emb_keys_try": ["H_sklearn_nmf_k80"],
    },
]

t0 = time.time()


def log(msg: str):
    dt = time.time() - t0
    print(f"[{dt:8.1f}s] {msg}", flush=True)


def zscore_cols(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True) + 1e-8
    return (A - mu) / sd


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def pick_embedding(adata_m: ad.AnnData, emb_keys_try):
    for k in emb_keys_try:
        if k in adata_m.obsm:
            return k, np.asarray(adata_m.obsm[k], dtype=np.float32)
    raise KeyError(f"None of these embeddings exist in adata.obsm: {emb_keys_try}")


def neighbors_graph_from_emb(Emb: np.ndarray):
    n = Emb.shape[0]
    a_tmp = ad.AnnData(X=sp.csr_matrix((n, 1), dtype=np.float32))
    a_tmp.obsm["X_embed"] = Emb

    sc.pp.neighbors(
        a_tmp,
        use_rep="X_embed",
        n_neighbors=N_NEIGH,
        random_state=GLOBAL_SEED,
        metric=METRIC,
    )
    A = a_tmp.obsp["connectivities"]
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    A = A.maximum(A.T).astype(np.float64, copy=False)
    return a_tmp, A


def leiden_repeats(a_tmp: ad.AnnData, A):
    clusters = []
    for i, sd in enumerate(LEIDEN_SEEDS):
        key = "leiden_tmp"
        sc.tl.leiden(
            a_tmp,
            adjacency=A,
            key_added=key,
            resolution=RESOLUTION,
            directed=False,
            use_weights=USE_WEIGHTS,
            flavor="igraph",
            n_iterations=2,
            random_state=sd,
        )
        clusters.append(a_tmp.obs[key].astype(str).to_numpy())
        log(f"  Leiden repeat {i + 1}/{LEIDEN_REPEATS} done.")
    return clusters


def score_against_labels(cluster_list, labels: np.ndarray):
    ari_list, nmi_list = [], []
    labels = labels.astype(str)
    for cl in cluster_list:
        ari_list.append(adjusted_rand_score(labels, cl))
        nmi_list.append(
            normalized_mutual_info_score(labels, cl, average_method="arithmetic")
        )
    ari_mu, ari_sd = mean_std(ari_list)
    nmi_mu, nmi_sd = mean_std(nmi_list)
    return ari_mu, ari_sd, nmi_mu, nmi_sd


def main():
    log("[metrics] Starting clustering metrics job")
    log(f"[cfg] OUT_CELLTYPE = {OUT_CELLTYPE}")
    log(f"[cfg] OUT_DONOR    = {OUT_DONOR}")
    log(f"[cfg] N_NEIGH={N_NEIGH}  LEIDEN_RES={RESOLUTION}  REPEATS={LEIDEN_REPEATS}  ZSCORE={ZSCORE}")

    rows_celltype = []
    rows_donor = []

    for spec in MODELS:
        name = spec["name"]
        path = spec["h5ad_path"]
        log("=" * 70)
        log(f"[model] {name}")
        log(f"[load ] {path}")

        adata_m = sc.read_h5ad(path)
        log(f"[data ] n_obs={adata_m.n_obs}, n_vars={adata_m.n_vars}")

        # Check required labels exist
        for lk in LABEL_KEYS:
            if lk not in adata_m.obs.columns:
                raise ValueError(f"[{name}] Missing adata.obs['{lk}'] in {path}")

        emb_key, Emb = pick_embedding(adata_m, spec["emb_keys_try"])
        log(f"[emb  ] Using obsm['{emb_key}'] with shape {Emb.shape}")

        if ZSCORE:
            Emb = zscore_cols(Emb)

        log("[knn  ] Building neighbors graph")
        a_tmp, A = neighbors_graph_from_emb(Emb)

        log(f"[leid ] Running Leiden {LEIDEN_REPEATS}x")
        cluster_list = leiden_repeats(a_tmp, A)

        # Score vs cell_type
        ct = adata_m.obs["cell_type"].astype(str).to_numpy()
        ari_mu, ari_sd, nmi_mu, nmi_sd = score_against_labels(cluster_list, ct)
        rows_celltype.append(
            {"Model": name, "ARI": ari_mu, "ARI Std": ari_sd, "NMI": nmi_mu, "NMI Std": nmi_sd}
        )
        log(f"[score] vs cell_type: ARI={ari_mu:.3f}±{ari_sd:.3f}, NMI={nmi_mu:.3f}±{nmi_sd:.3f}")

        # Score vs donor_id
        dn = adata_m.obs["donor_id"].astype(str).to_numpy()
        ari_mu, ari_sd, nmi_mu, nmi_sd = score_against_labels(cluster_list, dn)
        rows_donor.append(
            {"Model": name, "ARI": ari_mu, "ARI Std": ari_sd, "NMI": nmi_mu, "NMI Std": nmi_sd}
        )
        log(f"[score] vs donor_id : ARI={ari_mu:.3f}±{ari_sd:.3f}, NMI={nmi_mu:.3f}±{nmi_sd:.3f}")

        # Cleanup
        del adata_m, a_tmp, A, Emb, cluster_list

    df_ct = pd.DataFrame(rows_celltype)[["Model", "ARI", "ARI Std", "NMI", "NMI Std"]]
    df_dn = pd.DataFrame(rows_donor)[["Model", "ARI", "ARI Std", "NMI", "NMI Std"]]

    os.makedirs(os.path.dirname(OUT_CELLTYPE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DONOR), exist_ok=True)

    df_ct.to_csv(OUT_CELLTYPE, index=False)
    df_dn.to_csv(OUT_DONOR, index=False)

    log(f"[save ] Wrote: {OUT_CELLTYPE}")
    log(f"[save ] Wrote: {OUT_DONOR}")
    log("[done ] All done.\n")

    print("\n=== cell_type metrics ===")
    print(df_ct)
    print("\n=== donor_id metrics ===")
    print(df_dn)


if __name__ == "__main__":
    main()
