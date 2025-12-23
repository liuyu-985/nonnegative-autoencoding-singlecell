#!/bin/bash
#SBATCH --job-name=clustering_metrics_k80
#SBATCH --output=/mnt/home/liuyu/%x_%j.out
#SBATCH --error=/mnt/home/liuyu/%x_%j.err
#SBATCH --time=3-00:00:00
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

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR="${SLURM_TMPDIR:-/tmp}"


export METRICS_OUT_CELLTYPE="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/clustering_metrics_by_cell_type_k80.csv"
export METRICS_OUT_DONOR="/mnt/projects/debruinz_project/yu_ting/Tabula_Immune/clustering_metrics_by_donor_id_k80.csv"

echo "Python: $(python -V)"
echo "Running clustering metrics..."

python "$HOME/Tabula_Immune_cell_clustering.py"

echo "Done."
