# Brain Optimisation — Image-to-Brain Pipeline

A minimal pipeline to predict fMRI brain responses to a static image using the pretrained [TRIBE v2](https://github.com/facebookresearch/tribev2) model from Meta, with detailed cortical surface visualisations and brain region analysis.

---

## Overview

| Script | Input | Output |
|---|---|---|
| `run_image.py` | image file | `brain_response.npy` — predicted fMRI (20484 vertices × 40 TRs) |
| `plot_brain.py` | `brain_response.npy` | 6 whole-brain surface plots |
| `brain_regions.py` | `brain_response.npy` | ranked region table (stdout) |
| `plot_regions.py` | `brain_response.npy` | 5 region-level plots |

---

## Setup

```bash
pip install -r requirements.txt
```

> On first run, `run_image.py` downloads `facebook/tribev2` (~2 GB) and
> `facebook/vjepa2-vitg-fpc64-256` (~4 GB) from HuggingFace into `./cache/`.

### Image format

PIL is used to load images. AVIF files (common on newer iPhones/Macs) must be
converted first — PIL does not support AVIF:

```bash
sips -s format jpeg your_image.avif --out your_image.jpg
```

---

## Step 1 — Predict brain response (`run_image.py`)

```bash
python run_image.py path/to/image.jpg
```

Saves `brain_response.npy` in the current directory.

**What the script does:**

1. Loads the pretrained `FmriEncoderModel` from `facebook/tribev2`.
2. Loads `facebook/vjepa2-vitg-fpc64-256` (ViT-G) and processes the image as a
   64-frame static video clip.
3. Extracts features at layers `[0.5, 0.75, 1.0]` of network depth, group-mean
   aggregated to 2 feature vectors of size 1408 — exactly replicating the
   `neuralset` `HuggingFaceVideo` extractor used during training.
4. Omits audio and text entirely (absent modalities are zero-filled by
   `aggregate_features`, not passed through projectors).
5. Runs the Transformer encoder and saves predictions of shape `(n_vertices, 40)`.

**Output shape:**

```
brain_response.npy  →  float32 array, shape (20484, 40)
                        axis 0: cortical vertices (fsaverage5)
                        axis 1: predicted fMRI TRs (TR 0 = 5 s after stimulus onset)
```

---

## Step 2 — Whole-brain surface plots (`plot_brain.py`)

```bash
python plot_brain.py brain_response.npy plots/
```

Produces 6 figures in `plots/`:

| File | Description |
|---|---|
| `mean_activation.png` | Mean predicted response across all TRs — lateral L/R, medial L/R, dorsal |
| `peak_tr.png` | The single TR with highest mean activation — same 5 views |
| `timesteps.png` | Time-resolved strip: lateral-left brain map every 5 TRs |
| `temporal.png` | Mean ± std, p95 + max, and spatial std per TR |
| `distribution.png` | Histogram and CDF of per-vertex mean activations |
| `summary.png` | All panels combined into one overview figure |

---

## Step 3 — Brain region analysis (`brain_regions.py`)

### Command-line

```bash
# Top-20 activated regions (default)
python brain_regions.py brain_response.npy

# Top-50
python brain_regions.py brain_response.npy 50
```

**Example output:**

```
Loaded brain_response.npy: 20484 vertices × 40 TRs

Top 10 most activated HCP MMP regions (mean across vertices and TRs):
 rank region  score  n_vertices
    1  STSdp 0.1251         136
    2     A5 0.0933         109
    3  STSvp 0.0799         109
    4  STSda 0.0790          83
    5   STGa 0.0781          44
    6  TPOJ1 0.0649         153
    7   TE1a 0.0441          86
    8    55b 0.0379          66
    9    PEF 0.0377          57
   10    SFL 0.0373          92

Region group summaries:
         group    mean    std  peak_tr  n_vertices
      language  0.0579 0.0705       70         502
      auditory  0.0033 0.0622       67         590
   frontal_eye -0.0050 0.0537       45         288
    prefrontal -0.0133 0.0434       84         778
 somatosensory -0.0159 0.0516       45        1162
         motor -0.0294 0.0449       45        1045
  default_mode -0.0536 0.0466        1         608
   visual_core -0.0663 0.0621       97        1328
visual_ventral -0.0675 0.0617       95         400
        motion -0.0967 0.0630       81         279
 visual_dorsal -0.1036 0.0602       97         343
```

### Module API

```python
import numpy as np
from brain_regions import (
    list_regions,
    get_roi_indices,
    get_roi_activation,
    get_roi_timeseries,
    get_group_timeseries,
    top_regions,
    region_summary,
    REGION_GROUPS,
)

brain = np.load("brain_response.npy")  # (20484, 40)

# ── List all 181 available region names ───────────────────────────────────────
regions = list_regions()
# ['1', '10d', '10pp', ..., 'v23ab']  (181 total)

# ── Get vertex indices for a region ───────────────────────────────────────────
v1_idx = get_roi_indices("V1")               # both hemispheres
v1_left = get_roi_indices("V1", hemi="left") # left only

# Wildcard queries
v_idx     = get_roi_indices("V*")      # all regions starting with V
belt_idx  = get_roi_indices("*Belt")   # LBelt, MBelt, PBelt

# Multiple regions at once
motion_idx = get_roi_indices(["MT", "MST", "V4t"])

# ── Scalar activation for a region ────────────────────────────────────────────
act = get_roi_activation(brain, "FFC")   # mean over vertices and TRs
# 0.0312

# ── Per-TR timeseries for a region ────────────────────────────────────────────
v1_ts = get_roi_timeseries(brain, "V1")      # shape (40,)
ffc_ts = get_roi_timeseries(brain, "FFC")    # shape (40,)

# ── Timeseries for a named functional group ───────────────────────────────────
vis_ts  = get_group_timeseries(brain, "visual_core")   # shape (40,)
lang_ts = get_group_timeseries(brain, "language")      # shape (40,)

# ── Ranked table of top-k regions ─────────────────────────────────────────────
df = top_regions(brain, k=10)
#  rank region    score  n_vertices
#     1  STSdp  0.1251         136
#     2     A5  0.0933         109
#     ...

# ── Summary stats for a set of regions ────────────────────────────────────────
summary = region_summary(brain, ["A1", "LBelt", "MBelt", "PBelt"])
# region    mean    std     max  peak_tr  n_vertices
#     A1 -0.0142 0.0305  0.0805       61          42
#  LBelt -0.0250 0.0389  0.1065       67          66
#  MBelt -0.0211 0.0391  0.1126       61          67
#  PBelt -0.0257 0.0400  0.0927       67          82
```

### Predefined region groups (`REGION_GROUPS`)

| Group key | Regions |
|---|---|
| `visual_core` | V1, V2, V3, V4 |
| `visual_dorsal` | V3A, V3B, V6, V6A, V7, IPS1 |
| `visual_ventral` | V8, VVC, FFC, PIT, VMV1, VMV2, VMV3 |
| `motion` | MT, MST, V4t, FST, LO1, LO2, LO3 |
| `auditory` | A1, LBelt, MBelt, PBelt, RI, A4, A5 |
| `language` | 44, 45, STGa, STSda, STSdp, TA2 |
| `default_mode` | RSC, d23ab, v23ab, POS1, POS2, PCV |
| `frontal_eye` | FEF, PEF, SCEF |
| `somatosensory` | 3a, 3b, 1, 2 |
| `motor` | 4, 6a, 6d, 6v, 6r |
| `prefrontal` | 46, 9a, 9m, 9p, 10r, 10v, 10d, 10pp |

### Vertex layout

The `brain_response.npy` array has shape `(20484, T)`:
- Vertices `0–10241`: left hemisphere
- Vertices `10242–20483`: right hemisphere

`hemi="both"` (default) returns indices spanning both halves.
`hemi="left"` / `"right"` returns only that hemisphere's indices.

---

## Step 4 — Region activation plots (`plot_regions.py`)

```bash
python plot_regions.py brain_response.npy plots/
```

Produces 5 figures in `plots/`:

| File | Description |
|---|---|
| `top_regions_bar.png` | Horizontal bar chart of top-30 regions, colour-coded by functional group |
| `group_comparison.png` | Mean ± std activation per region group (bar chart) |
| `group_timeseries.png` | Per-TR timeseries for each of the 11 functional groups |
| `roi_brain_map.png` | Cortical surface map with the top-10 active regions labelled |
| `visual_timeseries.png` | Visual hierarchy timeseries: V1 → V2 → MT → FFC → VVC |

---

## Atlas reference (HCP MMP)

The **HCP Multi-Modal Parcellation** defines 181 bilateral cortical areas on
fsaverage5, accessed via MNE. The full list of region names:

```
1, 10d, 10pp, 10r, 10v, 11l, 13l, 2, 23c, 23d, 24dd, 24dv, 25, 31a,
31pd, 31pv, 33pr, 3a, 3b, 4, 43, 44, 45, 46, 47l, 47m, 47s, 52, 55b,
5L, 5m, 5mv, 6a, 6d, 6ma, 6mp, 6r, 6v, 7AL, 7Am, 7PC, 7PL, 7Pm, 7m,
8Ad, 8Av, 8BL, 8BM, 8C, 9-46d, 9a, 9m, 9p, A1, A4, A5, AAIC, AIP,
AVI, DVT, EC, FEF, FFC, FOP1-5, FST, H, IFJa, IFJp, IFSa, IFSp, IP0,
IP1, IP2, IPS1, Ig, LBelt, LIPd, LIPv, LO1, LO2, LO3, MBelt, MI, MIP,
MST, MT, OFC, OP1-4, PBelt, PCV, PEF, PF, PFcm, PFm, PFop, PFt, PGi,
PGp, PGs, PH, PHA1-3, PHT, PI, PIT, POS1, POS2, PSL, PeEc, Pir, PoI1,
PoI2, PreS, ProS, RI, RSC, SCEF, SFL, STGa, STSda, STSdp, STSva, STSvp,
STV, TA2, TE1a, TE1m, TE1p, TE2a, TE2p, TF, TGd, TGv, TPOJ1-3, V1, V2,
V3, V3A, V3B, V3CD, V4, V4t, V6, V6A, V7, V8, VIP, VMV1-3, VVC, a10p,
a24, a24pr, a32pr, a47r, a9-46v, d23ab, d32, i6-8, p10p, p24, p24pr,
p32, p32pr, p47r, p9-46v, pOFC, s32, s6-8, v23ab
```

---

## Equivalence to the official demo

This pipeline is equivalent to running the TRIBE v2 demo on a video of the input
image repeated as a static clip, with no audio or text.

It is **not** equivalent to the full demo on a real video because:
- Audio and language context are absent (the demo uses all three modalities)
- All timesteps see identical visual features — no temporal variation for the
  Transformer to attend over

---

## Output reference

| Field | Value |
|---|---|
| Mesh | fsaverage5 |
| Vertices | 20484 (10242 per hemisphere) |
| Timesteps | 40 TRs |
| TR offset | 5 s (hemodynamic lag baked in) |
| Format | NumPy `.npy`, float32 |
| Atlas | HCP MMP — 181 bilateral cortical areas (via MNE) |
