# brain_steer — Brain-Guided Image Generation

Trains a small CNN generator to produce images that maximise activation in
target brain regions and suppress others, using the pretrained TRIBE v2 model
as the loss oracle.

---

## Concept

The standard image-to-brain pipeline predicts **what the brain does** given
an image. This module inverts the question: **what image should I show to
maximally activate a specific brain region?**

```
Standard pipeline (run_image.py):
    image  →  V-JEPA2  →  TRIBE v2  →  brain activations

brain_steer (train.py):
    z ~ N(0,I)  →  Generator  →  image  →  V-JEPA2  →  TRIBE v2
                                               ↓
                                       brain activations
                                               ↓
                                    brain_region_loss  →  ∇ Generator weights
```

The generator learns which kinds of images consistently activate the target
regions. After training, sampling new latent vectors produces diverse images
that all tend to activate those regions.

---

## Loss function

The loss is a **softmax cross-entropy over brain regions**.

### Step-by-step

1. Compute mean activation per HCP MMP region (181 regions):

   ```
   scores[i] = mean( brain[vertices_i, :] )
   ```

2. Apply softmax with temperature τ:

   ```
   probs = softmax( scores / τ )
   ```

3. Cross-entropy term — maximise probability mass on target regions:

   ```
   ce_loss = -log( Σ probs[target_rois] )
   ```

4. Optional suppression term — penalise specific competing regions:

   ```
   suppress_loss = Σ probs[suppress_rois]
   ```

5. Total loss:

   ```
   loss = ce_loss + λ · suppress_loss
   ```

### Why this works

The softmax normalisation means pushing probability mass onto target regions
**automatically suppresses all others**. The suppression term gives extra
gradient signal for regions you especially want to avoid.

The temperature τ controls selectivity:
- **Low τ (e.g. 0.05)** — near-winner-takes-all. Good for a single precise
  region (e.g. `--target V1`).
- **High τ (e.g. 0.5)** — soft competition, activates a broader area.
  Good for a functional group (e.g. `--target FFC STSda STSdp`).

---

## File structure

```
brain_steer/
├── generator.py     # CNN generator: z → (3, 224, 224) image
├── brain_loss.py    # Softmax cross-entropy loss over HCP MMP regions
├── pipeline.py      # Differentiable image → V-JEPA2 → TRIBE v2 → brain
├── train.py         # Training loop (main entry point)
└── generate.py      # Inference: generate images from a trained checkpoint
```

---

## Setup

Same dependencies as the parent project:
```bash
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
# Maximise primary visual cortex (V1 + V2 + V3)
python -m brain_steer.train \
    --target V1 V2 V3 \
    --steps 200 \
    --out brain_steer/checkpoints/visual/

# Maximise face areas, suppress early visual cortex
python -m brain_steer.train \
    --target FFC STSda STSdp \
    --suppress V1 V2 \
    --temperature 0.05 \
    --steps 500 \
    --out brain_steer/checkpoints/faces/

# Maximise auditory cortex
python -m brain_steer.train \
    --target A1 LBelt MBelt PBelt \
    --steps 300 \
    --out brain_steer/checkpoints/auditory/

# Low-memory mode (8 frames instead of 64, ~8× less GPU RAM for backprop)
python -m brain_steer.train \
    --target V1 \
    --num-frames 8 \
    --steps 200 \
    --out brain_steer/checkpoints/v1_lowmem/
```

All region names come from the HCP MMP atlas. Run `python brain_regions.py`
to see all 181 options, or check the region groups in `brain_regions.py`:
`visual_core`, `visual_dorsal`, `visual_ventral`, `motion`, `auditory`,
`language`, `default_mode`, `frontal_eye`, `somatosensory`, `motor`, `prefrontal`.

### Generate

```bash
# Sample 8 images from a trained checkpoint
python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/faces/checkpoint_final.pt \
    --n 8 \
    --out brain_steer/generated/faces/

# Also run TRIBE v2 on the generated images to verify brain activation
python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/faces/checkpoint_final.pt \
    --n 4 \
    --analyse \
    --out brain_steer/generated/faces/
```

---

## Training arguments

| Argument | Default | Description |
|---|---|---|
| `--target` | required | HCP MMP region names to maximise |
| `--suppress` | none | Region names to explicitly suppress |
| `--steps` | 500 | Number of gradient steps |
| `--lr` | 1e-4 | Adam learning rate |
| `--temperature` | 0.1 | Softmax temperature (lower = more selective) |
| `--lambda-suppress` | 1.0 | Weight on the suppression loss term |
| `--latent-dim` | 256 | Generator noise dimension |
| `--num-frames` | 64 | Video frames for V-JEPA2 (use 8 for lower memory) |
| `--out` | `brain_steer/checkpoints` | Output directory |
| `--cache` | `./cache` | HuggingFace model cache |
| `--seed` | 42 | Random seed |
| `--save-interval` | 50 | Save image + checkpoint every N steps |

---

## Outputs

After training, `--out` contains:

```
checkpoints/
├── config.json                — training configuration
├── training_history.json      — loss/activation values per step
├── checkpoint_step0050.pt     — periodic checkpoints
├── ...
├── checkpoint_final.pt        — final generator weights
└── images/
    ├── step_0050.png
    ├── step_0100.png
    ...
```

After generation with `--analyse`:

```
generated/
├── generated_00.png
├── generated_01.png
...
└── brain_analysis.json        — top-10 regions per generated image
```

---

## Memory requirements

The most memory-intensive step is backpropagating through V-JEPA2 (ViT-G,
40 transformer blocks).

| Mode | RAM required | Notes |
|---|---|---|
| `--num-frames 64` | ~3–4 GB | Full pipeline, matches training conditions |
| `--num-frames 8`  | ~500 MB | 8× lighter; features differ slightly from 64-frame |
| `--num-frames 1`  | ~100 MB | Smoke-test only; single-frame features |

TRIBE v2 itself is small (~few hundred MB) and adds negligible overhead.

---

## How to interpret results

**Monitoring training:** Watch `target_prob` in the progress bar. This is the
total softmax probability mass assigned to your target regions. A well-trained
generator should push this towards 1.0.

**top3 regions** in the progress bar shows which regions currently have the
highest raw activation score. Early in training these will be random; by the
end they should include your target regions.

**Use `--analyse`** on generated images to confirm the brain effect: the
`← TARGET` tag marks target regions in the top-10.

---

## Example: face-selective generation

To generate images that preferentially activate the fusiform face area (FFC),
superior temporal sulcus (STSda/STSdp), and temporal pole areas — the network
underlying face perception — while suppressing early visual cortex:

```bash
python -m brain_steer.train \
    --target FFC STSda STSdp TE1a TE2a \
    --suppress V1 V2 V3 \
    --temperature 0.05 \
    --lambda-suppress 2.0 \
    --steps 500 \
    --out brain_steer/checkpoints/faces/

python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/faces/checkpoint_final.pt \
    --n 8 \
    --analyse \
    --out brain_steer/generated/faces/
```
