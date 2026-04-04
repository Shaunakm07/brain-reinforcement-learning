# Running on Stanford FarmShare

FarmShare (`rice.stanford.edu`) is Stanford's general-purpose research cluster.
It runs SLURM and provides GPU nodes suitable for all steps in this pipeline.

---

## First-time setup

SSH in and run the setup script **once**:

```bash
ssh <sunetid>@rice.stanford.edu

# Copy the project to your home directory
cd $HOME
git clone <your-repo-url> brain_optimisation
cd brain_optimisation

# Run setup (≈10 min — creates conda env, installs deps, downloads HCP atlas)
bash cluster/setup.sh
```

### What setup does

1. Loads the conda/Python and CUDA modules available on FarmShare
2. Creates a conda environment called `brain_opt`
3. Installs PyTorch with CUDA 12.1 and all project dependencies
4. Installs the `tribev2` package
5. Pre-downloads the HCP MMP brain atlas via MNE (~50 MB)
6. Creates `cluster/cluster_config.env` — a small file sourced by every job
7. Creates the scratch layout at `/scratch/users/<sunetid>/brain_optimisation/`

### If module names differ

FarmShare periodically updates its software stack. If `setup.sh` prints
`conda not found`, find the correct module name:

```bash
module avail 2>&1 | grep -i -E "conda|python|miniconda"
module avail 2>&1 | grep -i cuda
```

Then edit the module names near the top of `cluster/setup.sh` and
`cluster/activate_env.sh` to match what you see.

---

## Submitting jobs

### Single image — full pipeline (inference + plots)

```bash
bash cluster/submit_pipeline.sh /path/to/image.jpg
```

Submits two chained SLURM jobs:
1. **inference** — runs TRIBE v2, saves `brain_response.npy`
2. **plots** — runs all plotting scripts (depends on job 1)

### Single image + train a generator

```bash
bash cluster/submit_pipeline.sh /path/to/image.jpg \
    --train \
    --target FFC STSda STSdp \
    --suppress V1 V2 \
    --temperature 0.05 \
    --steps 500 \
    --run-name faces
```

Chains four jobs in order:
1. inference → 2. plots → (in parallel) 3. train → 4. generate

### Many images in parallel (array job)

```bash
# Create a file with one image path per line
ls /path/to/images/*.jpg > image_list.txt

bash cluster/submit_pipeline.sh --batch image_list.txt
```

Submits a SLURM array job — up to 5 images processed concurrently.

### Train only (skip inference)

```bash
bash cluster/submit_pipeline.sh --train-only \
    --target V1 V2 V3 \
    --steps 200 \
    --run-name visual
```

---

## Manual job submission

If you need more control, submit individual jobs directly:

```bash
# Inference
sbatch cluster/sbatch/inference.sh /path/to/image.jpg my_image

# Plots (after inference finishes)
sbatch cluster/sbatch/plots.sh \
    /scratch/users/$USER/brain_optimisation/outputs/my_image.npy my_image

# Train generator
sbatch cluster/sbatch/train.sh \
    --target FFC STSda STSdp \
    --suppress V1 V2 \
    --temperature 0.05 \
    --steps 500 \
    --run-name faces

# Generate from checkpoint
sbatch cluster/sbatch/generate.sh \
    /scratch/users/$USER/brain_optimisation/checkpoints/faces/checkpoint_final.pt \
    --n 32 --analyse --run-name faces_out
```

---

## Interactive session (for testing/debugging)

Request a GPU node interactively:

```bash
salloc -p gpu --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=2:00:00
```

Once you have the node:

```bash
source cluster/cluster_config.env
conda activate brain_opt
python run_image.py /path/to/image.jpg
```

---

## Monitoring

```bash
# See your queued and running jobs
squeue -u $USER

# More detail (state, reason, time left)
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.10l %R"

# Live log output
tail -f /scratch/users/$USER/brain_optimisation/logs/brain-train_<JOBID>.out

# Job history and exit codes (did it succeed?)
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed

# Cancel a specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

---

## Output structure

All outputs go to `/scratch/users/<sunetid>/brain_optimisation/`:

```
/scratch/users/<sunetid>/brain_optimisation/
├── outputs/
│   └── my_image.npy                ← predicted brain response
├── plots/
│   └── my_image/
│       ├── mean_activation.png
│       ├── peak_tr.png
│       ├── timesteps.png
│       ├── temporal.png
│       ├── distribution.png
│       ├── summary.png
│       ├── top_regions_bar.png
│       ├── group_comparison.png
│       ├── group_timeseries.png
│       ├── roi_brain_map.png
│       ├── visual_timeseries.png
│       └── region_rankings.txt
├── checkpoints/
│   └── faces/
│       ├── config.json
│       ├── training_history.json
│       ├── checkpoint_final.pt
│       └── images/
│           ├── step_0050.png
│           └── ...
├── generated/
│   └── faces_out/
│       ├── generated_00.png
│       ├── ...
│       └── brain_analysis.json
└── logs/
    ├── brain-inference_<JOB>.out
    ├── brain-plots_<JOB>.out
    ├── brain-train_<JOB>.out
    └── brain-generate_<JOB>.out
```

## Copying results to your laptop

```bash
# From your laptop:
scp -r <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/plots ./results/
scp -r <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/generated ./results/

# For large transfers, rsync is faster:
rsync -avz --progress \
    <sunetid>@rice.stanford.edu:/scratch/users/<sunetid>/brain_optimisation/ \
    ./cluster_results/
```

---

## Resource reference

| Job | Partition | GPU | RAM | Time | Notes |
|---|---|---|---|---|---|
| `inference.sh` | `gpu` | 1 | 48 GB | 1 h | V-JEPA2 needs ~24 GB GPU |
| `plots.sh` | `normal` | — | 16 GB | 30 min | CPU only |
| `train.sh` | `gpu` | 1 | 64 GB | 8 h | Backprop through V-JEPA2 |
| `generate.sh` | `gpu` | 1 | 32 GB | 1 h | |
| `batch_inference.sh` | `gpu` | 1/task | 48 GB | 1.5 h | 5 concurrent by default |

Reduce training memory with `--num-frames 8` (8× less RAM, slightly different features).

---

## Troubleshooting

### `module load` fails
Run `module avail` to see installed software, update the module names in
`cluster/activate_env.sh` and `cluster/setup.sh`.

### Jobs stay in `PD` (pending) state
```bash
squeue -u $USER -o "%.10i %.9P %.8T %R"   # shows the reason
```
- `Resources` — all GPU nodes are busy, wait or try off-peak hours
- `Dependency` — waiting for an upstream job to finish (normal)
- `Priority` — fairshare queue, will run eventually

### Out of GPU memory during training
```bash
sbatch cluster/sbatch/train.sh --target V1 --num-frames 8 --steps 200 --run-name v1_fast
```

### Models fail to download on compute node
Pre-download from a login node (FarmShare login nodes have internet access):
```bash
source cluster/cluster_config.env && conda activate brain_opt
python -c "
from transformers import AutoModel, AutoVideoProcessor
AutoVideoProcessor.from_pretrained('facebook/vjepa2-vitg-fpc64-256', cache_dir='$HF_HOME')
AutoModel.from_pretrained('facebook/vjepa2-vitg-fpc64-256', cache_dir='$HF_HOME')
print('Done')
"
```

### `No space left on device` on /scratch
FarmShare scratch has a quota. Check usage:
```bash
du -sh /scratch/users/$USER/*
```
Delete old outputs or move completed results to `$OAK` (if you have Oak storage).
