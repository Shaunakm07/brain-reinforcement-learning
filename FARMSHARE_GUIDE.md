# Stanford FarmShare — Step-by-Step Guide

This guide walks you through logging into Stanford FarmShare and running the
Brain Optimisation pipeline from scratch, assuming no prior HPC experience.

---

## Prerequisites

- A Stanford SUNet ID and password
- Two-factor authentication (Duo) set up for your SUNet account
- The project files on your laptop (this repository)

---

## Part 1 — Log in to FarmShare

### Open a terminal

- **Mac**: open the Terminal app (`/Applications/Utilities/Terminal.app`)
- **Windows**: use Windows Terminal, PowerShell, or install [Git Bash](https://gitforwindows.org)

### SSH into FarmShare

```bash
ssh <sunetid>@rice.stanford.edu
```

Replace `<sunetid>` with your Stanford username (e.g. `jsmith`).

You will be prompted for your password, then Duo two-factor authentication.
After authenticating you will see a prompt like:

```
[jsmith@rice04 ~]$
```

You are now on a **login node**. Login nodes are for file management and
submitting jobs — do not run heavy computation here.

### First time only — add SSH key to avoid typing your password every time

On your **laptop**, run:

```bash
ssh-keygen -t ed25519 -C "farmshare"   # press Enter at all prompts
ssh-copy-id <sunetid>@rice.stanford.edu
```

After this, `ssh <sunetid>@rice.stanford.edu` will log you in without a
password prompt (Duo still required unless you set up multiplexing below).

### Optional — avoid repeated Duo prompts

Add this to `~/.ssh/config` on your **laptop**:

```
Host farmshare
    HostName rice.stanford.edu
    User <sunetid>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Then `ssh farmshare` will reuse an existing connection for 10 minutes,
skipping Duo on repeat logins.

---

## Part 2 — Copy the project to FarmShare

Run this on your **laptop** (not inside the SSH session):

```bash
cd /Users/shaunak/Desktop/Brain\ Optimisation

# Copy the whole project to your FarmShare home directory
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'cache/' \
    --exclude 'brain_response.npy' \
    . <sunetid>@rice.stanford.edu:~/brain_optimisation/
```

This copies everything except the large model cache (which will be
re-downloaded to scratch on the cluster).

To verify it arrived, from inside the SSH session:

```bash
ls ~/brain_optimisation/
```

You should see `run_image.py`, `plot_brain.py`, `brain_regions.py`, etc.

---

## Part 3 — One-time environment setup

Run this **once** from your SSH session on FarmShare.
You do not need to repeat this for future jobs.

```bash
cd ~/brain_optimisation
bash cluster/setup.sh
```

This will take about 10 minutes. It:
1. Creates a conda environment called `brain_opt`
2. Installs all Python dependencies
3. Downloads the HCP brain atlas (~50 MB)
4. Creates the output folder at `/scratch/users/<sunetid>/brain_optimisation/`

When it finishes you should see:

```
============================================
Setup complete.
Config: cluster/cluster_config.env
Submit a job: bash cluster/submit_pipeline.sh /path/to/image.jpg
============================================
```

### If setup fails with "conda not found"

Find the correct module name and update the script:

```bash
module avail 2>&1 | grep -i -E "conda|python|miniconda"
```

Copy the exact module name shown, then edit `cluster/activate_env.sh`:

```bash
nano cluster/activate_env.sh
# Change the module load lines to match what you found above
# Save with Ctrl+O, exit with Ctrl+X
```

Re-run `bash cluster/setup.sh`.

---

## Part 4 — Copy an image to FarmShare

From your **laptop**, copy the image you want to run the model on:

```bash
scp "/Users/shaunak/Desktop/Brain Optimisation/sample_image_converted.jpg" \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/images/
```

Or copy a whole folder of images:

```bash
scp -r /path/to/your/images/ \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/images/
```

If the image is in AVIF format (from iPhone/Mac), convert it first on your
laptop:

```bash
sips -s format jpeg your_image.avif --out your_image.jpg
```

---

## Part 5 — Run the pipeline

All jobs are submitted from your SSH session on FarmShare.

### Option A — Full pipeline on one image (recommended for first run)

```bash
cd ~/brain_optimisation

bash cluster/submit_pipeline.sh \
    /scratch/users/$USER/brain_optimisation/images/sample_image_converted.jpg
```

This submits two SLURM jobs automatically:
- **Job 1** (`brain-inference`) — runs TRIBE v2, ~30–60 min
- **Job 2** (`brain-plots`) — generates all plots, starts automatically when Job 1 finishes

You will see output like:

```
Submitted inference : job 1234567
Submitted plots     : job 1234568 (after 1234567)

Monitor with:
  squeue -u $USER
```

### Option B — Also train a brain-guided image generator

```bash
bash cluster/submit_pipeline.sh \
    /scratch/users/$USER/brain_optimisation/images/sample_image_converted.jpg \
    --train \
    --target FFC STSda STSdp \
    --suppress V1 V2 \
    --temperature 0.05 \
    --steps 500 \
    --run-name faces
```

This chains four jobs: inference → plots → train (5–8 hrs) → generate

### Option C — Process many images in parallel

```bash
# Create a list of image paths
ls /scratch/users/$USER/brain_optimisation/images/*.jpg > image_list.txt

bash cluster/submit_pipeline.sh --batch image_list.txt
```

---

## Part 6 — Monitor your jobs

### Check job status

```bash
squeue -u $USER
```

Output looks like:

```
  JOBID PARTITION     NAME ST       TIME  NODES REASON
1234567       gpu   brain-i  R       8:23      1 None
1234568       gpu   brain-p PD       0:00      1 Dependency
```

Status codes:
- `PD` — pending (waiting to start)
- `R`  — running
- `CG` — completing
- `F`  — failed

`Dependency` means the job is waiting for the previous one to succeed — this
is normal.

### Watch the output live

```bash
# Replace JOBID with the number from squeue
tail -f /scratch/users/$USER/brain_optimisation/logs/brain-inference_1234567.out
```

Press `Ctrl+C` to stop watching.

### Check if a job succeeded or failed

```bash
sacct -j 1234567 --format=JobID,JobName,State,ExitCode,Elapsed
```

A `State` of `COMPLETED` and `ExitCode` of `0:0` means success.

### Cancel a job

```bash
scancel 1234567          # cancel one job
scancel -u $USER         # cancel all your jobs
```

---

## Part 7 — Get results back to your laptop

Once jobs complete, copy the outputs from FarmShare to your laptop.

From your **laptop**:

```bash
# Copy plots
rsync -avz --progress \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/plots/ \
    ./farmshare_results/plots/

# Copy generated images (if you ran brain_steer)
rsync -avz --progress \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/generated/ \
    ./farmshare_results/generated/

# Copy brain response .npy files
rsync -avz --progress \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/outputs/ \
    ./farmshare_results/outputs/
```

---

## Quick reference card

| Task | Command (run on FarmShare) |
|---|---|
| Check jobs | `squeue -u $USER` |
| Watch logs | `tail -f /scratch/users/$USER/brain_optimisation/logs/*.out` |
| Cancel job | `scancel <JOBID>` |
| Disk usage | `du -sh /scratch/users/$USER/*` |
| Interactive GPU | `salloc -p gpu --gres=gpu:1 --mem=48G --time=2:00:00` |
| Activate env | `source ~/brain_optimisation/cluster/cluster_config.env && conda activate brain_opt` |
| Check GPU (interactive) | `nvidia-smi` |
| See available modules | `module avail` |

---

## Troubleshooting

### "Permission denied (publickey)" when SSHing

Your SSH key may not be set up. Use password login:

```bash
ssh -o PreferredAuthentications=password <sunetid>@rice.stanford.edu
```

### Job immediately fails (State: FAILED, ExitCode: 1:0)

Read the error log:

```bash
cat /scratch/users/$USER/brain_optimisation/logs/brain-inference_<JOBID>.err
```

The most common cause is a module not loading. Check `cluster/activate_env.sh`
has the right module names.

### "No space left on device"

FarmShare scratch has quotas. Check what is using space:

```bash
du -sh /scratch/users/$USER/*/
```

Delete outputs you no longer need, or move completed results to your laptop.

### Jobs stuck in pending for hours

Check the reason:

```bash
squeue -u $USER -o "%.10i %.9P %.8T %R"
```

- `Resources` — all GPU nodes are in use; wait or try late evening/weekends
- `ReqNodeNotAvail` — requested resources unavailable; try removing the
  `--constraint` line from the sbatch script

### HuggingFace download fails inside a job

Pre-download models from the login node (which has internet access):

```bash
source ~/brain_optimisation/cluster/cluster_config.env
conda activate brain_opt
python -c "
from transformers import AutoModel, AutoVideoProcessor
import os
AutoVideoProcessor.from_pretrained(
    'facebook/vjepa2-vitg-fpc64-256',
    cache_dir=os.environ['HF_HOME']
)
AutoModel.from_pretrained(
    'facebook/vjepa2-vitg-fpc64-256',
    cache_dir=os.environ['HF_HOME']
)
print('Models cached successfully.')
"
```
