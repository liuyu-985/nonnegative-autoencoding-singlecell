#!/bin/bash
#SBATCH --job-name=ts_k80_donorOnly_cnmf1
#SBATCH --output=/mnt/home/liuyu/%x_%j.out
#SBATCH --error=/mnt/home/liuyu/%x_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g[001-002,005]
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

export USE_CUPY=1
export GLOBAL_SEED=42

# Avoid GPU sparse multiply OOM 
export CPU_NORM_LOG1P=1
export LIBSIZE_LOG1P=0

# CPU normalize/log1p uses LAYER_FOR_NMF; set to none to use adata.X
export LAYER_FOR_NMF="none"

# Training knobs
export K=80
export EPOCHS=100
export BATCH_SIZE=512
export FULL_EVAL_FRAC=0.05
export EVAL_BLOCK=256
export WARMUP_EPOCHS=5
export LAMBDA_W=1e-2
export LAMBDA_U=1e-1
export GAMMA_INV=1e-3
export ETA_HSIC=5e-4
export PATIENCE=6
export EARLY_STOP=1
export PRINT_EVERY=1


export H5AD_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad"
export OUT_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_1st_donor_only_k80.h5ad"

echo "Python: $(python -V)"
nvidia-smi || true

python "$HOME/run_cnmf1_singlefile.py"
