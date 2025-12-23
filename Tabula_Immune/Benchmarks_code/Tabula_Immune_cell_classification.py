#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CZ-style classification benchmark for multiple embedding models (single-file per model),
producing TWO summary CSVs:
  1) label = cell_type
  2) label = donor_id

Each summary CSV columns (exact order):
Model, MeanFoldF1, F1_std, MeanFoldAUROC, AUROC_std, MeanFoldPrecision, Precision_std,
MeanFoldAccuracy, Accuracy_std, MeanFoldRecall, Recall_std, n_rows, n_cells, n_classes,
min_cells_per_class

Notes:
- Uses embedding arrays stored in adata.obsm[emb_key]
- Requires obs columns: cell_type, donor_id
- the core evaluation idea: StratifiedKFold × multiple seeds × 3 classifiers
"""

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Logging helper
t0 = time.time()


def log(msg: str):
    elapsed = time.time() - t0
    print(f"[{elapsed:8.1f}s] {msg}", flush=True)


def summarize_metric(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def mean_fold_metric(results_df: pd.DataFrame, metric: str):
    if results_df.shape[0] == 0:
        return float("nan")
    return float(np.nanmean(results_df[metric].to_numpy(dtype=float)))



# Classifiers 
def build_classifiers(seed: int):
    clf_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
        ),
    )
    clf_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15),
    )
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
    )
    return {"log_reg": clf_lr, "knn": clf_knn, "rf": clf_rf}



# Config 
MODELS = [
    {
        "name": "CAE_NMF1",
        "h5ad_path": os.getenv(
            "MODEL1_PATH",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_1st_donor_only_k80.h5ad",
        ),
        "emb_key": os.getenv("MODEL1_EMB", "H_shared_k80"),
    },
    {
        "name": "CAE_NMF2",
        "h5ad_path": os.getenv(
            "MODEL2_PATH",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_2nd_donor_only_k80.h5ad",
        ),
        "emb_key": os.getenv("MODEL2_EMB", "H_shared_k80"),
    },
    {
        "name": "AE_NMF",
        "h5ad_path": os.getenv(
            "MODEL3_PATH",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_no_cond_k80.h5ad",
        ),
        "emb_key": os.getenv("MODEL3_EMB", "H_shared_k80"),
    },
    {
        "name": "Base_NMF",
        "h5ad_path": os.getenv(
            "MODEL4_PATH",
            "/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_sklearn_nmf_k80.h5ad",
        ),
        "emb_key": os.getenv("MODEL4_EMB", "H_sklearn_nmf_k80"),
    },
]


LABEL_KEYS = ["cell_type", "donor_id"]

# Filter rare classes
MIN_CELLS_PER_CLASS = int(os.getenv("MIN_CELLS_PER_CLASS", "10"))

# CV config
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
SEEDS = [int(x) for x in os.getenv("SEEDS", "42,52,62,72,82").split(",")]

# Optional: subsample for speed (default = use all)
SUBSAMPLE_FRAC = float(os.getenv("SUBSAMPLE_FRAC", "1.0"))  
SUBSAMPLE_MAX = int(os.getenv("SUBSAMPLE_MAX", "0"))       

# Output CSVs (summary only)
CELLTYPE_SUMMARY_OUT = os.getenv(
    "CELLTYPE_SUMMARY_OUT",
    "/mnt/projects/debruinz_project/yu_ting/celltype_classification_summary_k80.csv",
)
DONOR_SUMMARY_OUT = os.getenv(
    "DONOR_SUMMARY_OUT",
    "/mnt/projects/debruinz_project/yu_ting/donor_classification_summary_k80.csv",
)


def maybe_subsample(X, y, seed: int):
    if SUBSAMPLE_FRAC >= 1.0 and SUBSAMPLE_MAX <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    idx_all = np.arange(n)

    # target size
    target = int(np.floor(n * SUBSAMPLE_FRAC))
    if SUBSAMPLE_MAX > 0:
        target = min(target, SUBSAMPLE_MAX)
    target = max(target, 2)

    # simple stratified sampling
    out_idx = []
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    per_class = np.maximum(1, (props * target).astype(int))

    for c, k in zip(classes, per_class):
        idx_c = idx_all[y == c]
        if idx_c.size <= k:
            out_idx.append(idx_c)
        else:
            out_idx.append(rng.choice(idx_c, size=k, replace=False))

    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    return X[out_idx], y[out_idx]


def run_one_label(label_key: str) -> pd.DataFrame:
    log("\n" + "=" * 80)
    log(f"[cls] Running classification summary for LABEL_KEY = {label_key}")
    log("=" * 80)

    summary_rows = []

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        h5ad_path = model_cfg["h5ad_path"]
        emb_key = model_cfg["emb_key"]

        log(f"[cls] Model: {model_name}")
        log(f"[cls]   Loading: {h5ad_path}")

        if not os.path.exists(h5ad_path):
            log(f"[cls]   WARNING: file not found, skipping: {h5ad_path}")
            continue

        adata_m: ad.AnnData = sc.read_h5ad(h5ad_path)
        log(f"[cls]   Loaded: n_obs={adata_m.n_obs}, n_vars={adata_m.n_vars}")

        # Required metadata
        for req in ["cell_type", "donor_id"]:
            if req not in adata_m.obs.columns:
                log(f"[cls]   ERROR: obs['{req}'] missing; skipping.")
                continue

        if emb_key not in adata_m.obsm:
            log(f"[cls]   ERROR: obsm['{emb_key}'] missing; skipping.")
            continue

        # Labels
        y_str = adata_m.obs[label_key].astype(str)
        counts = y_str.value_counts()
        keep_labels = counts[counts >= MIN_CELLS_PER_CLASS].index.to_list()

        log(
            f"[cls]   Label counts (before filter): "
            f"{len(counts)} labels, min={int(counts.min())}, max={int(counts.max())}"
        )
        log(f"[cls]   Keeping {len(keep_labels)} labels with >= {MIN_CELLS_PER_CLASS} cells.")

        if len(keep_labels) < 2:
            log("[cls]   WARNING: <2 labels after filtering; skipping this model.")
            continue

        mask_keep = y_str.isin(keep_labels).to_numpy()
        n_before = adata_m.n_obs
        adata_f = adata_m[mask_keep].copy()
        n_after = adata_f.n_obs
        log(f"[cls]   After filter: {n_after}/{n_before} cells remain.")

        # Features
        X = np.asarray(adata_f.obsm[emb_key], dtype=np.float32)
        y = adata_f.obs[label_key].astype(str).to_numpy()

        # Encode labels -> ints for AUROC
        le = LabelEncoder()
        y_int = le.fit_transform(y)
        n_classes = int(len(le.classes_))

        # Optional subsample (default none)
        X, y_int = maybe_subsample(X, y_int, seed=SEEDS[0])
        n_cells = int(X.shape[0])

        # CV
        results_rows = []

        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            clfs = build_classifiers(seed)

            for fold_idx, (tr, te) in enumerate(skf.split(X, y_int), start=1):
                X_train, X_test = X[tr], X[te]
                y_train, y_test = y_int[tr], y_int[te]

                for clf_name, clf in clfs.items():
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
                    prec_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    rec_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)

                    # AUROC (macro OVR)
                    try:
                        if hasattr(clf, "predict_proba"):
                            y_proba = clf.predict_proba(X_test)
                        else:
                            # fallback
                            y_proba = np.zeros((y_test.shape[0], n_classes), dtype=float)
                            y_proba[np.arange(y_test.shape[0]), y_pred] = 1.0

                        auroc_mac = roc_auc_score(
                            y_test,
                            y_proba,
                            multi_class="ovr",
                            average="macro",
                        )
                    except Exception as e:
                        log(
                            f"[cls]   AUROC failed ({model_name} / {label_key} / {clf_name} "
                            f"fold {fold_idx}): {e}; using NaN."
                        )
                        auroc_mac = float("nan")

                    results_rows.append(
                        {
                            "model": model_name,
                            "label_key": label_key,
                            "seed": seed,
                            "fold": fold_idx,
                            "classifier": clf_name,
                            "accuracy": acc,
                            "f1": f1_mac,
                            "precision": prec_mac,
                            "recall": rec_mac,
                            "auroc": auroc_mac,
                            "n_cells": n_cells,
                            "n_classes": n_classes,
                            "min_cells_per_class": MIN_CELLS_PER_CLASS,
                        }
                    )

        df = pd.DataFrame(results_rows)

        # Summary row for this model
        acc_mu = mean_fold_metric(df, "accuracy")
        f1_mu = mean_fold_metric(df, "f1")
        prec_mu = mean_fold_metric(df, "precision")
        rec_mu = mean_fold_metric(df, "recall")
        auroc_mu = mean_fold_metric(df, "auroc")

        acc_mu2, acc_sd = summarize_metric(df["accuracy"])
        f1_mu2, f1_sd = summarize_metric(df["f1"])
        prec_mu2, prec_sd = summarize_metric(df["precision"])
        rec_mu2, rec_sd = summarize_metric(df["recall"])
        auroc_mu2, auroc_sd = summarize_metric(df["auroc"])

        summary_rows.append(
            {
                "Model": model_name,
                "MeanFoldF1": f1_mu,
                "F1_std": f1_sd,
                "MeanFoldAUROC": auroc_mu,
                "AUROC_std": auroc_sd,
                "MeanFoldPrecision": prec_mu,
                "Precision_std": prec_sd,
                "MeanFoldAccuracy": acc_mu,
                "Accuracy_std": acc_sd,
                "MeanFoldRecall": rec_mu,
                "Recall_std": rec_sd,
                "n_rows": int(df.shape[0]),
                "n_cells": int(n_cells),
                "n_classes": int(n_classes),
                "min_cells_per_class": int(MIN_CELLS_PER_CLASS),
            }
        )

        log(
            f"[cls]   Summary ({label_key}) {model_name}: "
            f"F1={f1_mu:.3f}, AUROC={auroc_mu:.3f}, Acc={acc_mu:.3f}"
        )

    # final DF with exact column order requested
    summary_df = pd.DataFrame(summary_rows)[
        [
            "Model",
            "MeanFoldF1",
            "F1_std",
            "MeanFoldAUROC",
            "AUROC_std",
            "MeanFoldPrecision",
            "Precision_std",
            "MeanFoldAccuracy",
            "Accuracy_std",
            "MeanFoldRecall",
            "Recall_std",
            "n_rows",
            "n_cells",
            "n_classes",
            "min_cells_per_class",
        ]
    ]
    return summary_df


def main():
    log("[cls] Starting classification summaries (single-file per model).")
    log(f"[cfg] MIN_CELLS_PER_CLASS={MIN_CELLS_PER_CLASS}, N_SPLITS={N_SPLITS}, SEEDS={SEEDS}")
    log(f"[cfg] SUBSAMPLE_FRAC={SUBSAMPLE_FRAC}, SUBSAMPLE_MAX={SUBSAMPLE_MAX}")

    df_celltype = run_one_label("cell_type")
    os.makedirs(os.path.dirname(CELLTYPE_SUMMARY_OUT), exist_ok=True)
    df_celltype.to_csv(CELLTYPE_SUMMARY_OUT, index=False)
    log(f"[cls] Saved cell_type summary -> {CELLTYPE_SUMMARY_OUT}")
    print(df_celltype)

    df_donor = run_one_label("donor_id")
    os.makedirs(os.path.dirname(DONOR_SUMMARY_OUT), exist_ok=True)
    df_donor.to_csv(DONOR_SUMMARY_OUT, index=False)
    log(f"[cls] Saved donor_id summary -> {DONOR_SUMMARY_OUT}")
    print(df_donor)

    log("[cls] All done.")


if __name__ == "__main__":
    main()
