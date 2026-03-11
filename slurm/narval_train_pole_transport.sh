#!/bin/bash
#SBATCH --job-name=pt-train
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/pt-train-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${REPO_DIR:-$REPO_DIR_DEFAULT}"

PY_MODULE="${PY_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
VENV_PATH="${VENV_PATH:-$HOME/PROJECT/venvs}"
CONFIG="${CONFIG:-configs/pole_transport_umino_cfo05_singlelayer.json}"
OUT_DIR="${OUT_DIR:-results/pole_transport_weights}"

module --force purge || true
module load "$PY_MODULE" 2>/dev/null || true
module load "$CUDA_MODULE" 2>/dev/null || true

activate_venv() {
  local base="$1"
  if [ -f "$base/bin/activate" ]; then
    # exact env path
    # shellcheck disable=SC1090
    source "$base/bin/activate"
    return 0
  fi
  if [ -d "$base" ]; then
    local candidate
    candidate=$(find "$base" -maxdepth 2 -type f -path '*/bin/activate' | head -n 1 || true)
    if [ -n "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      return 0
    fi
  fi
  echo "ERROR: Could not find a virtualenv activate script under VENV_PATH=$base"
  echo "Set VENV_PATH to the exact env path, for example:"
  echo "  export VENV_PATH=$HOME/PROJECT/venvs/<your_env_name>"
  exit 1
}

cd "$REPO_DIR"

# Sanity check: this package is an overlay, not a standalone repo.
for req in src/data/pusch_link.py src/receivers/baselines.py src/utils/config.py; do
  if [ ! -f "$req" ]; then
    echo "ERROR: Missing $req under REPO_DIR=$REPO_DIR"
    echo "This package must be merged into the full explainableAI repo before running."
    exit 1
  fi
done

activate_venv "$VENV_PATH"

mkdir -p "$OUT_DIR" slurm
python -m src.experiments.train_pole_transport_policy --config "$CONFIG" --out "$OUT_DIR"
