# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repo contains **TRIBE v2** — Meta's deep multimodal brain encoding model that predicts fMRI responses to naturalistic stimuli (video, audio, text). The model maps multimodal Transformer representations onto the cortical surface (fsaverage5 mesh, ~20k vertices).

The main codebase lives in `tribev2/tribev2/`. The outer `tribev2/` directory is a git submodule cloned from `facebookresearch/tribev2`.

## Installation

```bash
cd tribev2

# Inference only
pip install -e .

# With brain visualization
pip install -e ".[plotting]"

# With training (PyTorch Lightning, W&B)
pip install -e ".[training]"
```

## Running

**Quick local test:**
```bash
python -m tribev2.grids.test_run
```

**Grid search on Slurm:**
```bash
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

**Required environment variables for training:**
```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
# Optional
export SLURM_PARTITION="..."
export WANDB_ENTITY="..."
```

**Inference from pretrained weights:**
```python
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)  # (n_timesteps, n_vertices)
```

## Architecture

### Core data flow
1. **`studies/`** — Dataset definitions (Algonauts2025, Lahner2024, LeBel2023, Wen2017). Each registers study-specific loaders via `neuralset`.
2. **`main.py`** — `Data` (configures extractors → DataLoaders) and `TribeExperiment` (orchestrates training/eval via `exca.TaskInfra`).
3. **`model.py`** — `FmriEncoder` (`BaseModelConfig`): projects multimodal features → Transformer encoder → subject-specific predictor → fMRI vertices.
4. **`pl_module.py`** — PyTorch Lightning `BrainModule` wrapping `FmriEncoder`.
5. **`demo_utils.py`** — `TribeModel`: high-level inference class; handles video/audio/text → events → predictions.

### Key architectural details
- **`FmriEncoder`** uses per-modality `projector` MLPs, an optional `combiner`, a `TransformerEncoder`, and `SubjectLayers` (per-subject linear head). Features from text/audio/video/image extractors are concatenated (`extractor_aggregation="cat"`) before the encoder.
- Predictions are offset by **5 seconds** to compensate for hemodynamic lag.
- Multi-GPU training uses FSDP; batch size is automatically divided by `gpus_per_node`.
- Default backbone models: `meta-llama/Llama-3.2-3B` (text), `facebook/dinov2-large` (image), `facebook/vjepa2-vitg-fpc64-256` (video), `Wav2VecBert` (audio).

### Config system
All experiment configs are Pydantic models. Default hyperparameters live in `grids/defaults.py`. Grid sweeps in `grids/run_cortical.py` / `run_subcortical.py` override these defaults via `ConfDict`.

### Registration pattern
`main.py` uses wildcard imports (`from .model import *`, `from .studies import *`, etc.) to register subclasses with `neuraltrain`/`neuralset` registries. New models/studies/transforms must be importable from their respective modules.

## Pipeline scripts (root of repo)

| Script | Purpose |
|---|---|
| `run_image.py` | Run TRIBE v2 on a single image → `brain_response.npy` (n_vertices, 40) |
| `plot_brain.py` | Whole-brain surface plots from `brain_response.npy` |
| `brain_regions.py` | HCP MMP region utilities: extract vertices, compute per-ROI stats |
| `plot_regions.py` | Region-level activation bar charts, timeseries, and labelled surface maps |

**Typical two-step workflow:**
```bash
python run_image.py path/to/image.jpg          # → brain_response.npy
python plot_brain.py brain_response.npy plots/ # → 6 surface plots
python brain_regions.py brain_response.npy     # → ranked region table (stdout)
python plot_regions.py brain_response.npy plots/ # → 5 region plots
```

## Brain region atlas (HCP MMP)

`brain_regions.py` and `tribev2/utils.py` expose the **HCP Multi-Modal Parcellation** atlas via MNE, providing 181 bilateral cortical regions on fsaverage5.

### Key functions (in `tribev2/utils.py`)

| Function | What it does |
|---|---|
| `get_hcp_labels(mesh, hemi)` | Returns `dict[roi_name → vertex_indices]` for all 181 HCP regions |
| `get_hcp_roi_indices(roi, hemi, mesh)` | Vertex indices for a named ROI; supports `"V*"` / `"*Belt"` wildcards |
| `get_hcp_vertex_labels(mesh)` | Per-vertex string label for the entire surface (length 2×N) |
| `summarize_by_roi(data, hemi, mesh)` | Mean activation per ROI from a 1-D vertex array |
| `get_topk_rois(data, hemi, mesh, k)` | Top-k ROI names ranked by mean activation |

### Convenience functions (in `brain_regions.py`)

| Function | What it does |
|---|---|
| `list_regions()` | Sorted list of all 181 region names |
| `get_roi_indices(roi, hemi)` | Vertex indices (wraps `get_hcp_roi_indices`) |
| `get_roi_activation(brain, roi)` | Scalar mean activation for a region |
| `get_roi_timeseries(brain, roi)` | Shape-(T,) mean response per TR |
| `get_group_timeseries(brain, group)` | Same, but for a named group (e.g. `"visual_core"`) |
| `top_regions(brain, k)` | DataFrame of top-k regions sorted by mean activation |
| `region_summary(brain, rois)` | mean/std/max/peak_tr table for a list of regions |

### Predefined region groups (`REGION_GROUPS` in `brain_regions.py`)

`visual_core`, `visual_dorsal`, `visual_ventral`, `motion`, `auditory`, `language`,
`default_mode`, `frontal_eye`, `somatosensory`, `motor`, `prefrontal`

### Vertex layout (fsaverage5)
- Total: 20 484 vertices (indices 0–20 483)
- Left hemisphere: indices 0–10 241
- Right hemisphere: indices 10 242–20 483
- `hemi="both"` returns indices from both halves; `hemi="left"` / `"right"` returns
  only the respective half with the index offset already applied.

## Custom Agent

A `tribev2-architect` subagent (`.claude/agents/tribev2-architect.md`) is available for implementing or extending TriBeV2-specific components. Use it via the Agent tool when writing new model components, attention layers, or architecture extensions.
