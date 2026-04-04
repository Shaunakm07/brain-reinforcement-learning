# Stanford FarmShare — Exact Training + Pipeline Runbook

This is the **authoritative, copy/paste-ready guide** for running this repo on
Stanford FarmShare (`rice.stanford.edu`) with SLURM.

---

## 0) What this runbook covers

- One-time setup on FarmShare
- Data/image upload
- Forward pipeline submission (inference + plots)
- **Training submission for `brain_steer` (new FarmShare wrapper)**
- Monitoring, troubleshooting, and downloading outputs

If you only care about model training, jump to **Section 5**.

---

## 1) Login + copy code to FarmShare

From your local machine:

```bash
ssh <sunetid>@rice.stanford.edu
```

On FarmShare:

```bash
cd "$HOME"
git clone <your-repo-url> brain_optimisation
cd brain_optimisation
```

Or upload an existing local copy via `rsync`:

```bash
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'cache/' \
  . <sunetid>@rice.stanford.edu:~/brain_optimisation/
```

---

## 2) One-time environment setup (required)

Run once per account (or when rebuilding env):

```bash
cd ~/brain_optimisation
bash cluster/setup.sh
```

This script:
- loads FarmShare modules,
- creates conda env `brain_opt`,
- installs dependencies,
- downloads the HCP atlas,
- creates scratch layout at `/scratch/users/$USER/brain_optimisation`,
- writes `cluster/cluster_config.env`.

---

## 3) Upload input images to scratch

From local machine:

```bash
scp /path/to/image.jpg \
  <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/images/
```

On FarmShare, confirm:

```bash
ls /scratch/users/$USER/brain_optimisation/images/
```

---

## 4) Forward pipeline (image → brain response + plots)

From FarmShare:

```bash
cd ~/brain_optimisation
bash cluster/submit_pipeline.sh \
  /scratch/users/$USER/brain_optimisation/images/image.jpg
```

This submits:
1. `inference.sh` (GPU): produces `outputs/<image>.npy`
2. `plots.sh` (CPU): produces region/surface plots

---

## 5) Training on FarmShare (recommended new command)

Use the new wrapper:

```bash
bash cluster/submit_train_farmshare.sh --target V1 V2 V3 --run-name visual
```

### Example A — face-region objective

```bash
bash cluster/submit_train_farmshare.sh \
  --target FFC STSda STSdp \
  --suppress V1 V2 \
  --steps 500 \
  --temperature 0.05 \
  --run-name faces
```

### Example B — lower-memory run

```bash
bash cluster/submit_train_farmshare.sh \
  --target V1 \
  --num-frames 8 \
  --steps 250 \
  --run-name v1_lowmem
```

### Example C — chain training + generation

```bash
bash cluster/submit_train_farmshare.sh \
  --target FFC STSda STSdp \
  --suppress V1 V2 \
  --steps 400 \
  --run-name faces \
  --generate 32 --analyse
```

This submits `train.sh` and then (if `--generate N` is set) submits
`generate.sh` with a dependency on training success.

### Advanced: override FarmShare resources

```bash
bash cluster/submit_train_farmshare.sh \
  --target V1 V2 V3 \
  --run-name visual_long \
  --partition gpu \
  --time 12:00:00 \
  --mem 80G \
  --cpus 8 \
  --gres gpu:1
```

---

## 6) Monitoring jobs

```bash
squeue -u "$USER"
```

Follow logs:

```bash
tail -f /scratch/users/$USER/brain_optimisation/logs/train_<JOBID>.out
```

Check completed state + exit code:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed
```

Cancel a job:

```bash
scancel <JOBID>
```

---

## 7) Output locations

Everything is under:

```text
/scratch/users/<sunetid>/brain_optimisation/
```

Important subfolders:
- `outputs/` → `.npy` brain responses
- `plots/` → surface + ROI figures
- `checkpoints/<run-name>/` → training checkpoints + history
- `generated/<run-name>_generated/` → sampled images (+ optional analysis)
- `logs/` → SLURM stdout/stderr

---

## 8) Troubleshooting

### `conda not found` during setup

Run:

```bash
module avail 2>&1 | grep -i -E 'conda|python|miniconda'
```

Then update module entries in:
- `cluster/setup.sh`
- `cluster/activate_env.sh`

### Job pending for long time

Use:

```bash
squeue -u "$USER" -o '%.10i %.9P %.20j %.8T %.10M %.10l %R'
```

Common reasons:
- `Resources` (cluster busy)
- `Priority` (fairshare wait)
- `Dependency` (waiting for previous job)

### GPU OOM in training

Try:
- `--num-frames 8`
- fewer `--steps` for initial debug
- higher job memory/time override if needed

---

## 9) Quick command index

```bash
# One-time setup
bash cluster/setup.sh

# Full image pipeline
bash cluster/submit_pipeline.sh /scratch/users/$USER/brain_optimisation/images/image.jpg

# Train only (new wrapper)
bash cluster/submit_train_farmshare.sh --target V1 V2 V3 --run-name visual

# Train + generate + analyse
bash cluster/submit_train_farmshare.sh --target FFC STSda STSdp --run-name faces --generate 16 --analyse
```
