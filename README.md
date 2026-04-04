# Brain Optimisation — Image-to-Brain Pipeline

A minimal pipeline to predict fMRI brain responses to a static image using the pretrained [TRIBE v2](https://github.com/facebookresearch/tribev2) model from Meta, with detailed cortical surface visualisations and brain region analysis.

## What it does

**`run_image.py`** takes a single image, runs it through the TRIBE v2 brain encoder, and saves predicted fMRI activity across the cortical surface as `brain_response.npy` — shape `(~20484 vertices, 40 TRs)` on the fsaverage5 mesh.

**`plot_brain.py`** loads that output and produces six publication-ready whole-brain surface figures.

**`brain_regions.py`** provides utilities for extracting and analysing specific brain regions using the HCP Multi-Modal Parcellation (MMP) atlas.

**`plot_regions.py`** produces five region-level plots: ranked bar charts, group comparisons, timeseries per region group, a labelled brain map, and a visual hierarchy timeseries.

## Inference pipeline (`run_image.py`)

Four steps:

1. **Load TRIBE v2** — downloads `facebook/tribev2` from HuggingFace and loads the `FmriEncoderModel` directly, bypassing the high-level `TribeModel` abstractions.

2. **Extract V-JEPA2 features** — loads `facebook/vjepa2-vitg-fpc64-256` (ViT-G, the actual visual backbone used in training) and processes the image as a 64-frame static clip. Hidden states are extracted at layers `[0.5, 0.75, 1.0]` of network depth and reduced to 2 feature vectors via group-mean aggregation, then mean-pooled over all spatial tokens. This exactly replicates the `neuralset` `HuggingFaceVideo` extractor used during training.

3. **Build a minimal batch** — the feature vector is replicated across all 40 output timesteps and placed into a batch dict keyed by modality. Audio and text are omitted entirely (not zeroed through projectors), so `FmriEncoderModel.aggregate_features` treats them as absent and fills them with hard zeros — identical to how the model handles missing modalities during inference.

4. **Forward pass** — the `FmriEncoderModel` Transformer encoder runs on the features and produces predictions of shape `(1, n_vertices, 40)`. The result is saved to `brain_response.npy`.

## Plotting (`plot_brain.py`)

Loads `brain_response.npy` and saves six figures to `plots/`:

| File | Description |
|---|---|
| `mean_activation.png` | Mean response across all 40 TRs — lateral L/R, medial L/R, dorsal views |
| `peak_tr.png` | The TR with highest mean activation — same 5 views |
| `timesteps.png` | Time-resolved strip of lateral-left brain maps, one panel every 5 TRs |
| `temporal.png` | Mean±std, p95+max, and spatial std plotted per TR |
| `distribution.png` | Histogram and CDF of per-vertex mean activations |
| `summary.png` | All panels combined into one overview figure |

## Brain region analysis (`brain_regions.py`)

Uses the **HCP Multi-Modal Parcellation (MMP)** atlas to map the ~20 484 fsaverage5 vertices onto 181 named cortical areas. Can be used as a command-line tool or imported as a module.

### Command-line

```bash
# Print ranked table of top-20 activated regions
python brain_regions.py brain_response.npy

# Print top-50 regions
python brain_regions.py brain_response.npy 50
```

### Module API

```python
from brain_regions import (
    list_regions,           # → list of all 181 region names
    get_roi_indices,        # → vertex indices for one or more regions
    get_roi_activation,     # → scalar mean activation
    get_roi_timeseries,     # → (T,) timeseries for a region
    get_group_timeseries,   # → (T,) timeseries for a named group
    top_regions,            # → DataFrame ranked by activation
    region_summary,         # → mean/std/max/peak_tr table
    REGION_GROUPS,          # → dict of predefined functional groups
)
import numpy as np

brain = np.load("brain_response.npy")  # (n_vertices, 40)

# Vertex indices for V1 (both hemispheres)
v1_idx = get_roi_indices("V1")

# Mean activation at each TR for V1
v1_ts = get_roi_timeseries(brain, "V1")

# All visual areas matching "V*"
visual_idx = get_roi_indices("V*")

# Top-10 most activated regions
df = top_regions(brain, k=10)
print(df)

# Timeseries for the full visual cortex group
vis_ts = get_group_timeseries(brain, "visual_core")

# Summary stats for auditory cortex
summary = region_summary(brain, ["A1", "LBelt", "MBelt", "PBelt"])
```

### Predefined region groups

| Group | Regions |
|---|---|
| `visual_core` | V1, V2, V3, V4 |
| `visual_dorsal` | V3A, V3B, V6, V6A, V7, IPS1 |
| `visual_ventral` | V8, VVC, FFC, PIT, VMV1–3 |
| `motion` | MT, MST, V4t, FST, LO1–3 |
| `auditory` | A1, LBelt, MBelt, PBelt, RI, A4, A5 |
| `language` | 44, 45, STGa, STSda, STSdp, TA2 |
| `default_mode` | RSC, d23ab, v23ab, POS1, POS2, PCV |
| `frontal_eye` | FEF, PEF, SCEF |
| `somatosensory` | 3a, 3b, 1, 2 |
| `motor` | 4, 6a, 6d, 6v, 6r |
| `prefrontal` | 46, 9a, 9m, 9p, 10r, 10v, 10d, 10pp |

### Wildcard region queries

`get_roi_indices` supports prefix (`"V*"`) and suffix (`"*Belt"`) wildcards:

```python
# All visual areas
v_idx = get_roi_indices("V*")

# All auditory belt areas
belt_idx = get_roi_indices("*Belt")

# Specific list of regions
idx = get_roi_indices(["MT", "MST", "V4t"])
```

## Region plots (`plot_regions.py`)

```bash
python plot_regions.py brain_response.npy plots/
```

| File | Description |
|---|---|
| `top_regions_bar.png` | Horizontal bar chart, top-30 regions colour-coded by group |
| `group_comparison.png` | Mean ± std activation per region group |
| `group_timeseries.png` | Per-TR timeseries for each of the 11 functional groups |
| `roi_brain_map.png` | Surface map with top-10 regions labelled on the cortex |
| `visual_timeseries.png` | Visual hierarchy (V1 → MT → FFC) response over time |

## Equivalence to the demo

This pipeline is equivalent to running the official TRIBE v2 demo on a video consisting of the input image repeated as a static clip, with no audio or text. It is **not** equivalent to the full demo on a real video, because:

- Audio and language context are absent (the demo uses all three modalities)
- All timesteps see identical visual features — there is no temporal variation for the Transformer to attend over

## Setup

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Full usage

**Step 1 — predict brain response:**
```bash
python run_image.py path/to/image.jpg
```

**Step 2 — whole-brain surface plots:**
```bash
python plot_brain.py brain_response.npy plots/
```

**Step 3 — brain region analysis:**
```bash
python brain_regions.py brain_response.npy
```

**Step 4 — region activation plots:**
```bash
python plot_regions.py brain_response.npy plots/
```

## Output

| Field | Value |
|---|---|
| Mesh | fsaverage5 |
| Vertices | ~20484 (10242 per hemisphere) |
| Timesteps | 40 TRs |
| TR offset | 5 seconds (hemodynamic lag baked in) |
| Format | NumPy `.npy`, float32 |
| Atlas | HCP MMP (181 bilateral regions via MNE) |
