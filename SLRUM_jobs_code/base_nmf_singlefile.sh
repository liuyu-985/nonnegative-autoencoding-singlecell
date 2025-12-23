#!/bin/bash
#SBATCH --job-name=sklearn_nmf_base_k80_single
#SBATCH --output=/mnt/home/liuyu/%x_%j.out
#SBATCH --error=/mnt/home/liuyu/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --mail-user=liuyu@mail.gvsu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module --force purge
module load python/3.10.19

source "$HOME/.venvs/sc-nmf-gpu/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# Config
export GLOBAL_SEED=42
export K=80
export MAX_ITER=100
export VERBOSE=1

# Keep obs columns consistent with other models
export DONOR_COL="donor_id"
export CELLTYPE_COL="cell_type"

# Set to 1 to do normalize_total + log1p (sparse) before sklearn NMF
export LIBSIZE_LOG1P=1

# Paths
export H5AD_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/tabula_sapiens_immune_donor_ge5000.h5ad"
export OUT_PATH="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/results_only_sklearn_nmf_k80.h5ad"

echo "Python: $(python -V)"
echo "H5AD_PATH=$H5AD_PATH"
echo "OUT_PATH=$OUT_PATH"

python "$HOME/run_base_nmf_singlefile.py"
