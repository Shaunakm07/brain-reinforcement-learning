# Brain Optimisation — Image-to-Brain Pipeline

A minimal pipeline to predict fMRI brain responses to a static image using the pretrained [TRIBE v2](https://github.com/facebookresearch/tribev2) model from Meta.

## What it does

`run_image.py` takes a single image and predicts what fMRI activity it would evoke across the cortical surface, producing a `(~20484 vertices, 40 TRs)` array on the fsaverage5 mesh.

The pipeline has four steps:

1. **Load TRIBE v2** — downloads `facebook/tribev2` from HuggingFace and loads the `FmriEncoderModel` directly, bypassing the high-level `TribeModel` abstractions.

2. **Extract V-JEPA2 features** — loads `facebook/vjepa2-vitg-fpc64-256` (ViT-G, the actual visual backbone used in training) and processes the image as a 64-frame static clip. Hidden states are extracted at layers `[0.5, 0.75, 1.0]` of network depth and reduced to 2 feature vectors via group-mean aggregation, then mean-pooled over all spatial tokens. This exactly replicates the `neuralset` `HuggingFaceVideo` extractor used during training.

3. **Build a minimal batch** — the feature vector is replicated across all 40 output timesteps and placed into a batch dict keyed by modality. Audio and text are omitted entirely (not zeroed through projectors), so `FmriEncoderModel.aggregate_features` treats them as absent and fills them with hard zeros — identical to how the model handles missing modalities during inference.

4. **Forward pass** — the `FmriEncoderModel` Transformer encoder runs on the features and produces predictions of shape `(1, n_vertices, 40)`. The result is saved to `brain_response.npy`.

## Equivalence to the demo

This pipeline is equivalent to running the official TRIBE v2 demo on a video consisting of the input image repeated as a static clip, with no audio or text. It is **not** equivalent to the full demo on a real video, because:

- Audio and language context are absent (the demo uses all three modalities)
- All timesteps see identical visual features — there is no temporal variation for the Transformer to attend over

## Setup

Install TRIBE v2 from the submodule:

```bash
cd tribev2
pip install -e ".[training]"
```

## Usage

```bash
cd "/Users/shaunak/Desktop/Brain Optimisation"
python run_image.py path/to/image.jpg
```

Output is saved to `brain_response.npy` — shape `(20484, 40)`: cortical vertices × TR timesteps.

## Output

| Field | Value |
|---|---|
| Mesh | fsaverage5 |
| Vertices | ~20484 |
| Timesteps | 40 TRs |
| TR offset | 5 seconds (hemodynamic lag baked in) |
| Format | NumPy `.npy`, float32 |
