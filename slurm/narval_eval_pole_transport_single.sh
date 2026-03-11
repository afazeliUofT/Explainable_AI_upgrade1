#!/bin/bash
#SBATCH --job-name=pt-eval-1gpu
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/rsadve1/PROJECT/pole_transport_receiver_package
#SBATCH --output=/home/rsadve1/PROJECT/pole_transport_receiver_package/slurm/pt-eval-1gpu-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/rsadve1/PROJECT/pole_transport_receiver_package}"
PY_MODULE="${PY_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/PROJECT/venvs}"
CONFIG="${CONFIG:-configs/pole_transport_umino_cfo05_singlelayer.json}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results/pole_transport_eval}"

module --force purge || true
module load "$PY_MODULE" 2>/dev/null || true
module load "$CUDA_MODULE" 2>/dev/null || true

activate_venv() {
  local base="$1"
  if [ -f "$base/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$base/bin/activate"
    return 0
  fi
  if [ -d "$base" ]; then
    local candidates=""
    candidates=$(find "$base" -maxdepth 2 -type f -path '*/bin/activate' | sort || true)
    if [ -n "$candidates" ]; then
      while IFS= read -r cand; do
        if bash -lc "source '$cand' >/dev/null 2>&1 && python - <<'PY'
import tensorflow as tf
import sionna
print(tf.__version__)
print(getattr(sionna, '__version__', 'unknown'))
PY
" >/dev/null 2>&1; then
          # shellcheck disable=SC1090
          source "$cand"
          echo "Activated virtualenv: $(dirname "$(dirname "$cand")")"
          return 0
        fi
      done <<< "$candidates"
      echo "ERROR: Found virtualenvs under $base, but none could import both tensorflow and sionna."
      echo "$candidates"
      exit 1
    fi
  fi
  echo "ERROR: Could not find a usable virtualenv under VENV_PATH=$base"
  echo "Set VENV_PATH to the exact env path, for example:"
  echo "  export VENV_PATH=/home/rsadve1/PROJECT/venvs/<your_env_name>"
  exit 1
}

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

for req in src/data/pusch_link.py src/receivers/baselines.py src/utils/config.py src/experiments/run_pole_transport_sim.py; do
  if [ ! -f "$req" ]; then
    echo "ERROR: Missing $req under REPO_DIR=$REPO_DIR"
    exit 1
  fi
done

activate_venv "$VENV_PATH"

echo "REPO_DIR=$REPO_DIR"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "which python=$(which python)"
python -V

mkdir -p "$OUT_DIR" "$REPO_DIR/slurm"
python -m src.experiments.run_pole_transport_sim \
  --config "$CONFIG" \
  --out "$OUT_DIR"
