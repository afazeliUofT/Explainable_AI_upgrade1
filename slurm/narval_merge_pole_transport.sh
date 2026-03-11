#!/bin/bash
#SBATCH --job-name=pt-merge
#SBATCH --account=def-rsadve
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/pt-merge-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${REPO_DIR:-$REPO_DIR_DEFAULT}"

PY_MODULE="${PY_MODULE:-python/3.11}"
VENV_PATH="${VENV_PATH:-$HOME/PROJECT/venvs}"
IN_DIR="${IN_DIR:-results/pole_transport_eval}"

module --force purge || true
module load "$PY_MODULE" 2>/dev/null || true

activate_venv() {
  local base="$1"
  if [ -f "$base/bin/activate" ]; then
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
activate_venv "$VENV_PATH"

python -m src.experiments.merge_pole_transport_results --indir "$IN_DIR"
