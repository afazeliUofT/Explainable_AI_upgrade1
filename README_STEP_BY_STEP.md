# Pole-Transport Receiver Package for `afazeliUofT/explainableAI`

This package is a **drop-in overlay** for your existing repo. It adds a new NR-PUSCH receiver family:

- `pole_static` = paper-style dense-near-unit static poles
- `pole_transport_heuristic` = slot-conditioned pole transport without offline learning
- `pole_transport_learned` = same slot-conditioned receiver, but with a tiny trained pole policy

It also adds:

- a training script for the tiny pole policy,
- a simulation script that compares the new receiver to your repo's existing `lmmse_ls` baseline,
- a merge script for SLURM array jobs,
- Narval job scripts,
- a ready-made stressed UMi + CFO config.

---

## 1. Copy the overlay into your repo

From the directory **above** your repo:

```bash
rsync -av pole_transport_receiver_package/ explainableAI/
```

If you already copied the zip to Narval and extracted it there, run the same command on Narval.

After copying, your repo should now contain these new files:

```text
configs/pole_transport_umino_cfo05_singlelayer.json
configs/pole_transport_quick_smoke.json
slurm/narval_train_pole_transport.sh
slurm/narval_eval_pole_transport_array.sh
slurm/narval_merge_pole_transport.sh
src/receivers/pole_transport_policy.py
src/receivers/pole_transport_detector.py
src/receivers/pole_transport_receiver.py
src/experiments/report_utils.py
src/experiments/run_pole_transport_sim.py
src/experiments/train_pole_transport_policy.py
src/experiments/merge_pole_transport_results.py
```

---

## 2. Create or activate your Python environment

Inside the repo root:

```bash
cd ~/explainableAI
python -m venv ~/venvs/sionna
source ~/venvs/sionna/bin/activate
pip install --upgrade pip
pip install tensorflow sionna matplotlib
```

If your repo already has a working Sionna/TensorFlow environment, use that instead.

---

## 3. Run a smoke test first

This is only to verify imports and file paths.

```bash
cd ~/explainableAI
source ~/venvs/sionna/bin/activate
python -m src.experiments.run_pole_transport_sim \
  --config configs/pole_transport_quick_smoke.json \
  --out results/pole_transport_smoke
```

If that works, you should see these files:

```text
results/pole_transport_smoke/results.json
results/pole_transport_smoke/comparison_table.csv
results/pole_transport_smoke/comparison_report.txt
results/pole_transport_smoke/bler_comparison.png
```

---

## 4. Train the tiny pole policy on Narval

Edit the account line in

```text
slurm/narval_train_pole_transport.sh
```

Replace

```bash
#SBATCH --account=def-YOURPI
```

with your real Narval account.

If needed, also edit:

- `PY_MODULE`
- `CUDA_MODULE`
- `VENV_PATH`
- `REPO_DIR`

Then submit:

```bash
cd ~/explainableAI
sbatch slurm/narval_train_pole_transport.sh
```

This writes the trained tiny-network weights to:

```text
results/pole_transport_weights/pole_policy.weights.h5
```

and logs to:

```text
results/pole_transport_weights/train_log.json
results/pole_transport_weights/train_config_resolved.json
```

---

## 5. Evaluate all Eb/N0 points with a SLURM array job

The provided config uses 6 Eb/N0 points:

```json
[0, 2, 4, 6, 8, 10]
```

So the array script is already set to:

```bash
#SBATCH --array=0-5
```

If you change the list length in the config, also update the array range.

Submit the evaluation array:

```bash
cd ~/explainableAI
sbatch slurm/narval_eval_pole_transport_array.sh
```

Each array task writes one partial file, for example:

```text
results/pole_transport_eval/result_ebno_p0.00.json
results/pole_transport_eval/result_ebno_p2.00.json
...
```

---

## 6. Merge the array outputs into one comparison package

When all array jobs finish, submit the merge job:

```bash
cd ~/explainableAI
sbatch slurm/narval_merge_pole_transport.sh
```

This writes:

```text
results/pole_transport_eval/results.json
results/pole_transport_eval/results_compact.json
results/pole_transport_eval/comparison_table.csv
results/pole_transport_eval/comparison_report.txt
results/pole_transport_eval/bler_comparison.png
```

These are the files I need from you.

---

## 7. The exact files to send back for comparison

Please send me these 3 files first:

```text
results/pole_transport_eval/comparison_report.txt
results/pole_transport_eval/comparison_table.csv
results/pole_transport_eval/results.json
```

The PNG plot is also useful:

```text
results/pole_transport_eval/bler_comparison.png
```

---

## 8. What the comparison report contains

The script automatically writes:

- BLER and BER for every receiver at every Eb/N0
- per-point winner
- approximate dB gain at target BLER levels (when interpolation is possible)

The receivers are:

- `lmmse_ls`
- `pole_static`
- `pole_transport_heuristic`
- `pole_transport_learned` (only if trained weights were found)

If trained weights are missing, `pole_transport_learned` is skipped automatically.

---

## 9. One-command local evaluation without SLURM

If you ever want to run the full evaluation in one process (not recommended for long Narval runs), use:

```bash
cd ~/explainableAI
source ~/venvs/sionna/bin/activate
python -m src.experiments.run_pole_transport_sim \
  --config configs/pole_transport_umino_cfo05_singlelayer.json \
  --out results/pole_transport_eval_local
```

---

## 10. Why this config was chosen

The provided config is intentionally a **stress regime** where the pole-transport idea has a fair chance to beat the baseline:

- 5G NR PUSCH
- UMi channel
- 30 kHz SCS
- 20 PRBs
- 4 RX antennas
- nonzero normalized CFO = 0.05
- moderate mobility = 16.67 m/s
- single-layer uplink

This is meant to create structured frequency coupling (ICI), where a conditional frequency deconvolution receiver should help more than standard interpolation-based reception.

---

## 11. Troubleshooting

### A. `pole_transport_learned` is skipped

That means the file

```text
results/pole_transport_weights/pole_policy.weights.h5
```

was not found from the repo root.

Either:

- run training first, or
- update `pole_transport.policy.weights_path` in the config.

### B. `ModuleNotFoundError: sionna`

Activate the correct environment before running the scripts.

### C. SLURM array length mismatch

If you change the number of Eb/N0 points in the config, also change:

```bash
#SBATCH --array=0-5
```

in `slurm/narval_eval_pole_transport_array.sh`.

### D. You want a harder regime

Edit these in the config:

- `channel.impairments.cfo_normalized`
- `channel.speed_mps`
- `pusch.tb.mcs_index`

Recommended order:

1. increase CFO first,
2. then increase speed,
3. then increase MCS.

---

## 12. Minimal result checklist

After everything is done, confirm that these exist:

```text
results/pole_transport_weights/pole_policy.weights.h5
results/pole_transport_eval/results.json
results/pole_transport_eval/comparison_table.csv
results/pole_transport_eval/comparison_report.txt
```

If they exist, send them to me and I can judge whether the learned pole-transport receiver actually opened a reliable gap over `lmmse_ls` and over `pole_static`.
