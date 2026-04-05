# Brain Optimisation — End-to-End Guide

This project uses Meta's pretrained [TRIBE v2](https://github.com/facebookresearch/tribev2)
brain encoding model to:

1. **Predict** how a human brain responds to any image (fMRI simulation)
2. **Analyse** which specific brain regions activate and by how much
3. **Generate** images engineered to maximally activate a chosen brain region

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Setup](#2-setup)
3. [Project layout](#3-project-layout)
4. [Step 1 — Predict brain response](#4-step-1--predict-brain-response-run_imagepy)
5. [Step 2 — Whole-brain surface plots](#5-step-2--whole-brain-surface-plots-plot_brainpy)
6. [Step 3 — Brain region analysis](#6-step-3--brain-region-analysis-brain_regionspy)
7. [Step 4 — Region activation plots](#7-step-4--region-activation-plots-plot_regionspy)
8. [Step 5 — Train a brain-guided image generator](#8-step-5--train-a-brain-guided-image-generator-brain_steer)
9. [Step 6 — Generate images from a trained model](#9-step-6--generate-images-from-a-trained-model)
10. [Brain atlas reference](#10-brain-atlas-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What this project does

### The forward problem (Steps 1–4)

TRIBE v2 is a deep learning model trained on human fMRI data. Given any image,
it predicts the blood-oxygen-level-dependent (BOLD) response across ~20,000
points on the cortical surface. This prediction is the same thing a neuroscientist
would measure by putting a person in an MRI scanner and showing them the image.

**The output is a float array of shape `(20484 vertices, 40 TRs)`**, where:
- Each vertex is a point on the `fsaverage5` cortical surface mesh
- Each TR (repetition time) is a 2-second window of predicted BOLD signal
- The first TR corresponds to 5 seconds after stimulus onset (hemodynamic lag)

### The inverse problem (Step 5–6)

`brain_steer/` inverts the pipeline: instead of asking "what does the brain do
given this image?", it asks "what image should I generate to maximally activate
a specific brain region?". A small generator network is trained using the TRIBE v2
prediction as a differentiable loss function.

---

## 2. Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### First-run model downloads

On the first run of any script, two large models are downloaded from HuggingFace
into `./cache/`:

| Model | Size | Used for |
|---|---|---|
| `facebook/tribev2` | ~2 GB | Brain encoder (TRIBE v2) |
| `facebook/vjepa2-vitg-fpc64-256` | ~4 GB | Visual feature extractor (V-JEPA2 ViT-G) |

Subsequent runs use the cached versions.

### Image format

PIL is used to load images. AVIF files (common on newer iPhones and Macs) are
not supported. Convert them first:

```bash
sips -s format jpeg your_image.avif --out your_image.jpg
```

---

## 3. Project layout

```
Brain Optimisation/
│
├── run_image.py          # Step 1: image → brain_response.npy
├── plot_brain.py         # Step 2: brain_response.npy → 6 surface plots
├── brain_regions.py      # Step 3: region analysis utility + CLI
├── plot_regions.py       # Step 4: region-level plots
│
├── brain_steer/          # Step 5–6: brain-guided image generation
│   ├── generator.py      #   CNN generator: z → image
│   ├── brain_loss.py     #   Softmax cross-entropy over brain regions
│   ├── pipeline.py       #   Differentiable image → brain forward pass
│   ├── train.py          #   Training loop CLI
│   └── generate.py       #   Inference CLI
│
├── tribev2/              # TRIBE v2 model code (git submodule, do not edit)
├── requirements.txt
└── plots/                # Generated figures (created automatically)
```

---

## 4. Step 1 — Predict brain response (`run_image.py`)

### What it does

Runs an image through the TRIBE v2 pipeline and saves the predicted fMRI
response as a NumPy array.

The internal pipeline is:
1. Load `facebook/tribev2` (`FmriEncoderModel`)
2. Load `facebook/vjepa2-vitg-fpc64-256` (ViT-G visual backbone)
3. Process the image as a 64-frame static video clip
4. Extract features at 50%, 75%, and 100% of network depth, group-mean
   aggregated to 2 feature vectors of size 1408
5. Forward through the TRIBE v2 Transformer encoder
6. Save predictions as `brain_response.npy`

### Usage

```bash
python run_image.py path/to/image.jpg
```

```bash
# With a specific output path (edit the script or redirect):
python run_image.py path/to/image.jpg
# Output is always saved to brain_response.npy in the current directory
```

### Output

```
brain_response.npy
  dtype:  float32
  shape:  (20484, 40)
            │      └── 40 TRs (time points)
            └──────── 20484 cortical vertices (fsaverage5)

  Value range:  typically -0.35 to +0.30
  Positive values = above-baseline predicted BOLD response
  Negative values = below-baseline (suppression)
```

### Example

```bash
python run_image.py sample_image_converted.jpg
```

```
Feature dims: {'video': (2, 1408), 'audio': (2, ...), 'text': (2, ...)}
Output shape: (20484, 40)
Saved to brain_response.npy
```

---

## 5. Step 2 — Whole-brain surface plots (`plot_brain.py`)

### What it does

Loads `brain_response.npy` and produces six publication-quality figures
showing the predicted brain response on the 3D cortical surface.

### Usage

```bash
python plot_brain.py brain_response.npy plots/
```

Arguments:
- `brain_response.npy` — path to the NumPy array from Step 1 (default: `brain_response.npy`)
- `plots/` — output directory (default: `plots/`)

### Output files

| File | Description |
|---|---|
| `mean_activation.png` | Mean response averaged over all 40 TRs, shown from 5 angles: lateral left/right, medial left/right, dorsal |
| `peak_tr.png` | The single TR with the highest mean activation across vertices — same 5 views |
| `timesteps.png` | Time-resolved strip: lateral-left brain map at every 5th TR |
| `temporal.png` | Three line plots: mean±std, p95+max, and spatial std of activation per TR |
| `distribution.png` | Histogram and cumulative distribution of per-vertex mean activations |
| `summary.png` | All panels combined into one overview figure |

### Reading the surface maps

- **Hot colormap** (black → red → yellow → white): brighter = stronger activation
- **Sulcal shading**: dark grooves = sulci (folds), light ridges = gyri
- The 5 standard views together give complete coverage of the cortical surface

### Example output summary

For an image of a face, you might expect:
- Strong activation in the **temporal lobe** (face perception areas)
- Moderate activation in **occipital cortex** (visual processing)
- The peak TR will be in the 5–15 TR range (10–30 seconds after onset)

---

## 6. Step 3 — Brain region analysis (`brain_regions.py`)

### What it does

Maps the 20,484 cortical vertices onto 181 named brain regions using the
**HCP Multi-Modal Parcellation (MMP)** atlas. Provides both a command-line
tool and a Python module API.

### Command-line usage

```bash
# Print ranked table of top-20 activated regions
python brain_regions.py brain_response.npy

# Print top-50 regions
python brain_regions.py brain_response.npy 50
```

### Example output

```
Loaded brain_response.npy: 20484 vertices × 40 TRs

Top 10 most activated HCP MMP regions (mean across vertices and TRs):
 rank region    score  n_vertices
    1  STSdp  0.1251         136    ← posterior dorsal superior temporal sulcus
    2     A5  0.0933         109    ← auditory association cortex
    3  STSvp  0.0799         109    ← posterior ventral STS
    4  STSda  0.0790          83    ← anterior dorsal STS
    5   STGa  0.0781          44    ← anterior superior temporal gyrus
    6  TPOJ1  0.0649         153    ← temporo-parieto-occipital junction
    7   TE1a  0.0441          86    ← temporal area 1a
    8    55b  0.0379          66    ← premotor/language area
    9    PEF  0.0377          57    ← parietal eye field
   10    SFL  0.0373          92    ← superior frontal language

Region group summaries:
         group    mean    std  peak_tr  n_vertices
      language  0.0579  0.0705      70         502
      auditory  0.0033  0.0622      67         590
   frontal_eye -0.0050  0.0537      45         288
    ...
```

### Python module API

```python
import numpy as np
from brain_regions import (
    list_regions,           # all 181 HCP region names
    get_roi_indices,        # vertex indices for a region
    get_roi_activation,     # scalar mean activation
    get_roi_timeseries,     # (T,) activation per TR
    get_group_timeseries,   # (T,) for a named functional group
    top_regions,            # DataFrame ranked by activation
    region_summary,         # mean/std/max/peak_tr table
    REGION_GROUPS,          # predefined functional groups
)

brain = np.load("brain_response.npy")   # (20484, 40)

# ── List all available regions ─────────────────────────────────────────────
regions = list_regions()
# ['1', '10d', ..., 'v23ab']  (181 total)

# ── Get vertex indices ─────────────────────────────────────────────────────
v1_idx      = get_roi_indices("V1")            # both hemispheres
v1_left     = get_roi_indices("V1", hemi="left")
visual_all  = get_roi_indices("V*")            # wildcard: all V regions
belt_areas  = get_roi_indices("*Belt")         # LBelt, MBelt, PBelt
motion_rois = get_roi_indices(["MT", "MST", "V4t"])  # multiple regions

# ── Scalar activation for a region ────────────────────────────────────────
ffc_act = get_roi_activation(brain, "FFC")     # → float, e.g. 0.031

# ── Per-TR timeseries ──────────────────────────────────────────────────────
v1_ts  = get_roi_timeseries(brain, "V1")       # shape (40,)
ffc_ts = get_roi_timeseries(brain, "FFC")      # shape (40,)

# ── Functional group timeseries ────────────────────────────────────────────
vis_ts  = get_group_timeseries(brain, "visual_core")   # V1+V2+V3+V4
lang_ts = get_group_timeseries(brain, "language")      # 44+45+STGa+...

# ── Ranked table ───────────────────────────────────────────────────────────
df = top_regions(brain, k=10)
#  rank  region    score  n_vertices
#     1   STSdp  0.1251         136
#     2      A5  0.0933         109
#     ...

# ── Summary stats for specific regions ────────────────────────────────────
summary = region_summary(brain, ["A1", "LBelt", "MBelt", "PBelt"])
#  region    mean     std     max  peak_tr  n_vertices
#      A1 -0.0142  0.0305  0.0805       61          42
#   LBelt -0.0250  0.0389  0.1065       67          66
#   MBelt -0.0211  0.0391  0.1126       61          67
#   PBelt -0.0257  0.0400  0.0927       67          82
```

### Wildcard queries

`get_roi_indices` supports prefix and suffix wildcards:

```python
get_roi_indices("V*")      # all regions starting with V: V1, V2, V3, V4, V6...
get_roi_indices("*Belt")   # LBelt, MBelt, PBelt
get_roi_indices("STS*")    # STSda, STSdp, STSva, STSvp
get_roi_indices("TE*")     # TE1a, TE1m, TE1p, TE2a, TE2p
```

### Predefined region groups

| Group | Regions | Function |
|---|---|---|
| `visual_core` | V1, V2, V3, V4 | Primary and secondary visual cortex |
| `visual_dorsal` | V3A, V3B, V6, V6A, V7, IPS1 | Dorsal "where" pathway |
| `visual_ventral` | V8, VVC, FFC, PIT, VMV1–3 | Ventral "what" pathway |
| `motion` | MT, MST, V4t, FST, LO1–3 | Motion perception |
| `auditory` | A1, LBelt, MBelt, PBelt, RI, A4, A5 | Auditory cortex |
| `language` | 44, 45, STGa, STSda, STSdp, TA2 | Language/speech areas |
| `default_mode` | RSC, d23ab, v23ab, POS1, POS2, PCV | Default mode network |
| `frontal_eye` | FEF, PEF, SCEF | Frontal eye fields |
| `somatosensory` | 3a, 3b, 1, 2 | Somatosensory cortex |
| `motor` | 4, 6a, 6d, 6v, 6r | Motor cortex |
| `prefrontal` | 46, 9a–p, 10r/v/d/pp | Prefrontal cortex |

### Vertex layout

```
Vertex indices 0     – 10241   →  left hemisphere
Vertex indices 10242 – 20483   →  right hemisphere
```

`hemi="both"` (default) combines both. `hemi="left"` or `"right"` returns
indices for one hemisphere only (with correct offsets).

---

## 7. Step 4 — Region activation plots (`plot_regions.py`)

### Usage

```bash
python plot_regions.py brain_response.npy plots/
```

### Output files

| File | Description |
|---|---|
| `top_regions_bar.png` | Horizontal bar chart of the 30 most activated HCP regions, colour-coded by functional group |
| `group_comparison.png` | Bar chart comparing mean ± std activation across all 11 functional groups |
| `group_timeseries.png` | 11-panel grid: per-TR timeseries for each group, mean ± std shaded |
| `roi_brain_map.png` | Cortical surface map with the top-10 active regions labelled by name |
| `visual_timeseries.png` | Visual hierarchy timeseries: V1 → V2 → MT → FFC → VVC on one axis |

---

## 8. Step 5 — Train a brain-guided image generator (`brain_steer/`)

### Concept

The generator is a small CNN that maps a random noise vector to a 224×224 RGB
image. It is trained with TRIBE v2 as the loss function: at each step, the
generated image is passed through the brain encoder and the loss measures how
much the image fails to activate the target brain regions.

```
Training loop (one step):

  z ~ N(0, I)  [shape: 256]
       ↓
  Generator (CNN, trainable)
       ↓
  image  [3 × 224 × 224, values in [0,1]]
       ↓
  V-JEPA2 ViT-G  [frozen — no weight updates, gradients flow through]
       ↓
  features  [2 × 1408]
       ↓
  TRIBE v2  [frozen — no weight updates, gradients flow through]
       ↓
  brain activations  [20484 vertices × 40 TRs]
       ↓
  brain_region_loss  [scalar]
       ↓
  backprop → update Generator only
```

### Loss function in detail

```
scores[i]  = mean activation over all vertices in HCP region i
probs      = softmax( scores / temperature )
ce_loss    = -log( sum( probs[target_rois] ) )
supp_loss  = sum( probs[suppress_rois] )          ← optional
loss       = ce_loss  +  λ · supp_loss
```

The softmax normalisation means pushing probability mass onto target regions
automatically suppresses all non-target regions.

**Temperature guide:**
- `--temperature 0.05` — very selective, strong gradient signal, good for a single region
- `--temperature 0.1`  — default, good balance
- `--temperature 0.5`  — diffuse, targets a broad functional area

### Usage

```bash
# Maximise primary visual cortex
python -m brain_steer.train \
    --target V1 V2 V3 \
    --steps 200 \
    --out brain_steer/checkpoints/visual/

# Maximise face areas, suppress early visual cortex
python -m brain_steer.train \
    --target FFC STSda STSdp TE1a TE2a \
    --suppress V1 V2 \
    --temperature 0.05 \
    --lambda-suppress 2.0 \
    --steps 500 \
    --out brain_steer/checkpoints/faces/

# Maximise auditory cortex
python -m brain_steer.train \
    --target A1 LBelt MBelt PBelt \
    --steps 300 \
    --out brain_steer/checkpoints/auditory/

# Low-memory mode (8 frames instead of 64, ~8× less RAM needed for backprop)
python -m brain_steer.train \
    --target V1 \
    --num-frames 8 \
    --steps 200 \
    --out brain_steer/checkpoints/v1_fast/
```

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--target` | required | HCP MMP region names to maximise |
| `--suppress` | none | Region names to explicitly suppress |
| `--steps` | 500 | Number of gradient steps |
| `--lr` | 1e-4 | Adam learning rate |
| `--temperature` | 0.1 | Softmax temperature (lower = more selective) |
| `--lambda-suppress` | 1.0 | Weight on the explicit suppression term |
| `--latent-dim` | 256 | Generator noise vector dimension |
| `--num-frames` | 64 | Video frames for V-JEPA2 (8 for low-memory) |
| `--out` | `brain_steer/checkpoints` | Output directory |
| `--cache` | `./cache` | HuggingFace model cache directory |
| `--seed` | 42 | Random seed |
| `--log-interval` | 10 | Print stats every N steps |
| `--save-interval` | 50 | Save image + checkpoint every N steps |

### Progress bar explained

```
Training: 45%|████████         | 225/500 [12:03, loss=3.2104, p_tgt=0.0821, top3=FFC=0.041,STSda=0.038,V1=0.021]
```

| Field | Meaning |
|---|---|
| `loss` | Total loss (lower = better; starts ~5, should decrease) |
| `p_tgt` | Softmax probability mass on target regions (higher = better; aim for >0.3) |
| `top3` | The 3 regions currently receiving the most activation — these should shift towards your target |

### Output files

```
checkpoints/
├── config.json                 ← training configuration (targets, lr, etc.)
├── training_history.json       ← loss, target_prob, top5 per step
├── checkpoint_step0050.pt      ← periodic checkpoint (every --save-interval steps)
├── checkpoint_step0100.pt
├── ...
├── checkpoint_final.pt         ← final trained generator weights
└── images/
    ├── step_0050.png            ← sample image at each checkpoint
    ├── step_0100.png
    ...
```

### Memory requirements

| `--num-frames` | RAM needed | Notes |
|---|---|---|
| 64 (default) | ~3–4 GB | Full pipeline, most faithful to training |
| 8 | ~500 MB | 8× lighter, slightly different features |
| 1 | ~100 MB | Smoke-test only |

---

## 9. Step 6 — Generate images from a trained model (`brain_steer/generate.py`)

### Usage

```bash
# Sample 8 images
python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/faces/checkpoint_final.pt \
    --n 8 \
    --out brain_steer/generated/faces/

# Sample + run TRIBE v2 on each image to verify which regions activated
python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/faces/checkpoint_final.pt \
    --n 4 \
    --analyse \
    --out brain_steer/generated/faces/
```

### `--analyse` output

For each generated image, prints the top-10 activated regions:

```
Image 00 — top-10 activated regions:
   1. FFC          +0.04821  ← TARGET
   2. STSda        +0.04103  ← TARGET
   3. PIT          +0.03891
   4. STSdp        +0.03712  ← TARGET
   5. TE1a         +0.03540  ← TARGET
   ...
```

Also saves `brain_analysis.json` with the full results.

### All generate arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to a `.pt` checkpoint from `train.py` |
| `--n` | 8 | Number of images to generate |
| `--out` | `brain_steer/generated` | Output directory |
| `--analyse` | off | Run TRIBE v2 to verify brain activation |
| `--num-frames` | 64 | Frames for V-JEPA2 during analysis |
| `--cache` | `./cache` | HuggingFace model cache |
| `--seed` | 0 | RNG seed |

---

## 10. Brain atlas reference

### HCP Multi-Modal Parcellation (MMP)

The HCP MMP atlas defines **181 bilateral cortical areas** on the fsaverage5
surface, based on combining cortical architecture, function, connectivity, and
topography from hundreds of human subjects.

The 181 region names include well-known areas:

```
Visual:       V1  V2  V3  V3A V3B V4  V6  V6A V7  V8  MT  MST LO1 LO2 FFC VVC
Auditory:     A1  A4  A5  LBelt MBelt PBelt RI
Language:     44  45  STGa STSda STSdp STSva STSvp TA2
Motor:        4   6a  6d  6v  6r  FEF  PEF
Parietal:     AIP LIPd LIPv VIP IP0 IP1 IP2 IPS1 7AL 7Am 7PC 7PL
Prefrontal:   46  9a  9m  9p  10r 10v 10d 10pp 11l 47l 47m 47s
Temporal:     TE1a TE1m TE1p TE2a TE2p TGd TGv PHA1 PHA2 PHA3
Default mode: RSC  d23ab v23ab POS1 POS2 PCV
... and 100+ more
```

Run `python brain_regions.py` to see all 181 with their current activation
values, or call `list_regions()` in Python.

### How the atlas is loaded

The atlas is fetched via MNE's `fetch_hcp_mmp_parcellation()` and aligned to
the fsaverage5 mesh (10,242 vertices per hemisphere). Results are cached after
the first call.

---

## 11. Troubleshooting

### `PIL.UnidentifiedImageError`

Your image file format is not supported. Common cause: AVIF files from modern
cameras/phones. Fix:

```bash
sips -s format jpeg your_image.avif --out your_image.jpg
python run_image.py your_image.jpg
```

### `No such file or directory: '/Users/.../Brain'`

The path contains spaces and was not quoted. Always quote paths with spaces:

```bash
python run_image.py "/Users/shaunak/Desktop/Brain Optimisation/image.jpg"
# or from inside the project directory:
python run_image.py image.jpg
```

### `IndexError: index 10331 is out of bounds` in surface plots

This is a known bug in `tribev2/plotting/cortical.py`'s `annotate_rois` method
for right-hemisphere views. `plot_regions.py` already works around it by only
annotating left-hemisphere views.

### `CUDA out of memory` / MPS out of memory during training

Reduce the number of video frames:

```bash
python -m brain_steer.train --target V1 --num-frames 8 --steps 200
```

Or use a CPU-only run (slow but will complete):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m brain_steer.train --target V1 --num-frames 4 --steps 50
```

### Training loss not decreasing

- Try a lower temperature: `--temperature 0.02`
- Try a higher learning rate: `--lr 5e-4`
- Ensure the target regions exist: run `python brain_regions.py` and check region names
- Check that the target is not naturally suppressed for all images — try `brain_regions.py` on a few real images first to see if the region activates at all

### Models are not downloading

Check your internet connection and HuggingFace status. The models require
`huggingface_hub` to be installed and may require authentication for gated
models. Run:

```bash
huggingface-cli login
```

---

## Quick-start summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Convert image if needed
sips -s format jpeg photo.avif --out photo.jpg

# 3. Predict brain response
python run_image.py photo.jpg

# 4. Whole-brain surface plots
python plot_brain.py brain_response.npy plots/

# 5. See which regions activated most
python brain_regions.py brain_response.npy

# 6. Region-level plots
python plot_regions.py brain_response.npy plots/

# 7. Train generator targeting visual cortex
python -m brain_steer.train \
    --target V1 V2 V3 \
    --steps 200 \
    --num-frames 8 \
    --out brain_steer/checkpoints/visual/

# 8. Generate images from trained model
python -m brain_steer.generate \
    --checkpoint brain_steer/checkpoints/visual/checkpoint_final.pt \
    --n 8 \
    --analyse \
    --out brain_steer/generated/visual/
```
