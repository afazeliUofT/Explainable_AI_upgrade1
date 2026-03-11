#!/bin/bash
#SBATCH --job-name=pusch_reservoir
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# NOTE: Set your account if required by your allocation policy.
# #SBATCH --account=def-youraccount

set -euo pipefail

# If your cluster requires module loads, uncomment and edit:
# module load python/3.10
# module load cuda

source "$HOME/PROJECT/venvs/bin/activate"

# Flush python prints immediately to Slurm stdout
export PYTHONUNBUFFERED=1

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"
# Ensure the repo root is on PYTHONPATH so probe scripts under results/ can `import src`.
# (Running `python results/.../probe.py` sets sys.path[0] to the probe directory, not the repo.)
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

OUTDIR="results/run_${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"

python -u -m src.experiments.run_sim \
  --config configs/example_pusch_reservoir.json \
  --out "$OUTDIR" 2>&1 | tee -a "$OUTDIR/console.log"
