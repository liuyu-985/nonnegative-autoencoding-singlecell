#!/bin/bash
#SBATCH --job-name=ts_k80_donor_2nd
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

export USE_CUPY=1
export GLOBAL_SEED=42


export H5AD_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad"
export OUT_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_CNMF_2nd_donor_only_k80.h5ad"

export CT_COL="cell_type"
export LAYER_FOR_NMF="none"

# training knobs
export K=80
export EPOCHS=100
export BATCH_SIZE=512

# eval streaming
export FULL_EVAL_FRAC=0.05
export EVAL_BLOCK=2000
export SAVE_BLOCK=2000

# regularization / lr
export LAMBDA_W=1e-2
export LAMBDA_U=5e-2
export LR_W=1e-3
export LR_U=5e-4

# early stop
export EARLY_STOP=1
export PATIENCE=6
export PRINT_EVERY=1

# preprocessing (CPU normalize/log1p inside python)
export LIBSIZE_LOG1P=1
export USE_HVG=0
export SAVE_U=0

echo "Python: $(python -V)"
nvidia-smi || true

python "$HOME/run_cnmf2_singlefile.py"
