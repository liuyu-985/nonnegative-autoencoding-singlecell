#!/bin/bash
#SBATCH --job-name=cls_metrics_k80
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

cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR="${SLURM_TMPDIR:-/tmp}"


echo "Python: $(python -V)"
echo "Running Tabula_Immune_cell_classification_2.py from: $(pwd)"
python Tabula_Immune_cell_classification_2.py
echo "Done."
