#!/bin/bash
#SBATCH --job-name=nmf_no_cond_k80_single
#SBATCH --output=/mnt/home/liuyu/%x_%j.out
#SBATCH --error=/mnt/home/liuyu/%x_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-user=liuyu@mail.gvsu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module --force purge
module load python/3.10.19
module load cuda/12.8.1

source "$HOME/.venvs/sc-nmf-gpu/bin/activate"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

# GPU on
export USE_CUPY=1
export GLOBAL_SEED=42

# Avoid GPU sparse multiply OOM during preprocessing
export CPU_NORM_LOG1P=1
export LIBSIZE_LOG1P=1

export USE_HVG=0
export ROW_NORM=0

# Model/training knobs
export K=80
export EPOCHS=100
export BATCH_SIZE=512
export PATIENCE=6
export LAMBDA_W=1e-2
export LR_W=1e-3

# Full-loss eval knobs
export FULL_EVAL_FRAC=0.05
export EVAL_BLOCK=256
export PRINT_EVERY=1

# Keep stratified batching 
export STRATIFY_COLS="cell_type"
export CELLTYPE_COL="cell_type"
export DONOR_COL="donor_id"

export H5AD_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad"
export OUT_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_no_cond_k80.h5ad"

echo "Python: $(python -V)"
nvidia-smi || true

python "$HOME/run_no_cond_nmf_singlefile.py"
