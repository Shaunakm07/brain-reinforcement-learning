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

## Custom Agent

A `tribev2-architect` subagent (`.claude/agents/tribev2-architect.md`) is available for implementing or extending TriBeV2-specific components. Use it via the Agent tool when writing new model components, attention layers, or architecture extensions.
