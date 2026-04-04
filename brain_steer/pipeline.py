"""
Differentiable forward pass: RGB image tensor → TRIBE v2 brain activations.

This module reimplements the run_image.py feature extraction pipeline in pure
PyTorch so that gradients can flow from the brain activation loss all the way
back to the input image (and through it, back to a generator network).

Pipeline
--------

    image_01  (3, H, W),  values in [0, 1]
        │
        ├─ Resize to 224×224 (bilinear, differentiable)
        ├─ Normalise with ImageNet μ/σ (differentiable)
        └─ Replicate to 64-frame video  (1, 3, 64, 224, 224)
              │
              ▼
         V-JEPA2  (frozen weights, gradients flow through)
         hidden_states: 41 × (1, n_tokens, 1408)
              │
              ├─ Mean over tokens  →  (41, 1408)
              └─ Group-mean [0.5, 0.75, 1.0]  →  (2, 1408)
                   │
                   ▼
              TRIBE v2  (frozen weights, gradients flow through)
              batch.data = {"video": (1, 2, 1408, T)}
                   │
                   ▼
              brain  (n_vertices, T)   ←  loss computed here

Gradient flow
-------------
Even though V-JEPA2 and TRIBE v2 weights have requires_grad=False, PyTorch
autograd still propagates gradients THROUGH their operations when the input
descends from a learnable tensor (e.g. the generator). This is the standard
frozen-backbone training pattern.

Memory note
-----------
Backpropagating through V-JEPA2 (ViT-G, 40 blocks, 12545 tokens per frame)
is memory-intensive. Each layer's activation is ~70 MB in float32.
Use --num-frames 8 (train.py flag) to reduce token count 8×, at the cost
of slightly different features than the 64-frame training pipeline.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "tribev2"))

# ImageNet normalisation constants (used by AutoVideoProcessor with do_rescale=True)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

NUM_FRAMES_DEFAULT = 64


def image_to_video_input(
    image_01: torch.Tensor,
    num_frames: int = NUM_FRAMES_DEFAULT,
) -> torch.Tensor:
    """
    Convert a [0, 1] RGB image to a V-JEPA2 video input tensor.

    Replicates the AutoVideoProcessor(do_rescale=True) pipeline in
    differentiable PyTorch, so gradients flow back to `image_01`.

    Args:
        image_01: (3, H, W) float tensor with values in [0, 1]
        num_frames: number of identical frames to replicate (default 64,
                    matching the vjepa2-vitg-fpc64-256 training config)

    Returns:
        pixel_values_videos: (1, num_frames, 3, 256, 256) normalised float tensor.
        V-JEPA2 expects shape (B, T, C, H, W) with 256×256 resolution and the
        key name "pixel_values_videos" (changed from "pixel_values" in newer
        transformers versions).
    """
    # V-JEPA2's native resolution is 256×256 (not 224×224)
    img = F.interpolate(
        image_01.unsqueeze(0),          # (1, 3, H, W)
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, 256, 256)

    # ImageNet normalisation (matches AutoVideoProcessor behaviour)
    mean = torch.tensor(_MEAN, device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_STD,  device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
    img = (img - mean) / std  # (1, 3, 256, 256)

    # Replicate the single frame into a static video clip.
    # V-JEPA2 expects (B, T, C, H, W) — time dimension is axis 1.
    # expand() is lazy (no copy); contiguous() materialises it for V-JEPA2.
    pixel_values_videos = img.unsqueeze(1).expand(1, num_frames, 3, 256, 256).contiguous()
    return pixel_values_videos  # (1, T, 3, 256, 256)


def extract_vjepa_features(vjepa, pixel_values_videos: torch.Tensor) -> torch.Tensor:
    """
    Run V-JEPA2 and extract (2, 1408) features matching the neuralset
    HuggingFaceVideo extractor exactly:

        layers             = [0.5, 0.75, 1.0]
        layer_aggregation  = "group_mean"   → 2 output vectors
        token_aggregation  = "mean"         → pool over all spatial tokens

    Gradient flow is maintained through all operations.

    Args:
        vjepa: V-JEPA2 model (frozen weights, eval mode)
        pixel_values_videos: (1, T, 3, 256, 256) normalised float tensor

    Returns:
        feats: (2, 1408) feature tensor with gradient path back through vjepa
    """
    out = vjepa(pixel_values_videos=pixel_values_videos, output_hidden_states=True)

    # Stack all hidden states: (1, n_layers, n_tokens, 1408)
    states = torch.stack(list(out.hidden_states), dim=1)

    # Token aggregation: mean over all spatial+temporal tokens
    # → (n_layers, 1408)
    states = states[0].mean(dim=1)

    # Layer aggregation: group_mean matching neuralset's _aggregate_layers
    n = states.shape[0]
    fracs = [0.5, 0.75, 1.0]
    idxs  = [int(f * (n - 1)) for f in fracs]
    idxs[-1] += 1  # make the last index exclusive (inclusive last layer)

    feats = torch.stack([
        states[i1:i2].mean(0)
        for i1, i2 in zip(idxs[:-1], idxs[1:])
    ])  # (2, 1408)

    return feats


def features_to_brain(tribe_model, feats: torch.Tensor) -> torch.Tensor:
    """
    Forward the (2, 1408) feature vector through TRIBE v2 to get brain
    activations of shape (n_vertices, T).

    Replicates the batch construction in run_image.py.
    Gradient flow is maintained.

    Args:
        tribe_model: frozen TRIBE v2 FmriEncoderModel
        feats: (2, 1408) feature tensor

    Returns:
        brain: (n_vertices, T) activation tensor
    """
    T = tribe_model.n_output_timesteps
    n_layers, feat_dim = tribe_model.feature_dims["video"]  # (2, 1408)

    # Expand to (1, 2, 1408, T) — same video features at every TR
    video_feats = (
        feats
        .unsqueeze(0)   # (1, 2, 1408)
        .unsqueeze(-1)  # (1, 2, 1408, 1)
        .expand(1, n_layers, feat_dim, T)
        .contiguous()
    )

    batch = SimpleNamespace(data={"video": video_feats})
    preds = tribe_model(batch)  # (1, n_vertices, T)
    return preds[0]             # (n_vertices, T)


def image_to_brain(
    image_01: torch.Tensor,
    vjepa,
    tribe_model,
    num_frames: int = NUM_FRAMES_DEFAULT,
) -> torch.Tensor:
    """
    Full differentiable forward pass: image → brain activations.

    Args:
        image_01: (3, H, W) float tensor in [0, 1].
                  Should trace back to the generator for gradients to flow.
        vjepa: V-JEPA2 model (frozen, eval mode)
        tribe_model: TRIBE v2 FmriEncoderModel (frozen, eval mode)
        num_frames: number of frames for the video clip (default 64)

    Returns:
        brain: (n_vertices, T) with full gradient path back to image_01
    """
    pixel_values_videos = image_to_video_input(image_01, num_frames=num_frames)
    feats               = extract_vjepa_features(vjepa, pixel_values_videos)
    brain               = features_to_brain(tribe_model, feats)
    return brain
