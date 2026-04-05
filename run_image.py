"""
Minimal pipeline: run the TRIBE v2 brain encoder on a single image.

Matches the training feature extraction pipeline exactly:
- Backbone: facebook/vjepa2-vitg-fpc64-256 (the actual video_feature backbone)
- 64 frames per clip (static image repeated), clip_duration=4s
- layers=[0.5, 0.75, 1.0] + layer_aggregation="group_mean" → 2 output layers
- token_aggregation="mean" over all spatial/temporal tokens

Usage:
    python run_image.py path/to/image.jpg
"""

import sys
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace
from transformers import AutoVideoProcessor, AutoModel

from tribev2.demo_utils import TribeModel  # registers tribev2 components

# ── 1. Load pretrained model ───────────────────────────────────────────────
xp = TribeModel.from_pretrained(r"facebook/tribev2", cache_folder="./cache")
model = xp._model  # FmriEncoderModel (nn.Module), eval + on device
device = model.device

print("Feature dims:", model.feature_dims)
# Expected: {'video': (2, 1408), 'audio': (2, ...), 'text': (2, ...)}

# ── 2. Extract V-JEPA2 features ────────────────────────────────────────────
# neuralset's HuggingFaceVideo/_HFVideoModel uses:
#   - AutoVideoProcessor with do_rescale=True
#   - input field "videos" as a list of uint8 numpy frames (H, W, 3)
#   - 64 frames per clip (vjepa2 default)
#
# Layer extraction (from HuggingFaceMixin._aggregate_layers):
#   layer_indices = [int(f * (n_total_layers - 1)) for f in [0.5, 0.75, 1.0]]
#   layer_aggregation = "group_mean" → groups [idx0:idx1] and [idx1:idx2+1]
#   → 2 output layers (not 3)
#
# Token aggregation (HuggingFaceMixin._aggregate_tokens, token_aggregation="mean"):
#   mean over all tokens (spatial + temporal flattened together)

VJEPA_MODEL = "facebook/vjepa2-vitg-fpc64-256"
NUM_FRAMES = 64  # fpc64

processor = AutoVideoProcessor.from_pretrained(VJEPA_MODEL, do_rescale=True)
vjepa = AutoModel.from_pretrained(VJEPA_MODEL, output_hidden_states=True).to(device).eval()

image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
image_np = np.array(Image.open(image_path).convert("RGB"))  # (H, W, 3), uint8
frames = list(np.stack([image_np] * NUM_FRAMES))  # list of 64 (H, W, 3) arrays

inputs = processor(videos=frames, return_tensors="pt").to(device)

with torch.inference_mode():
    out = vjepa(**inputs)
    # hidden_states: tuple of len n_total_layers, each (B, n_tokens, hidden_size)
    states = torch.cat([x.unsqueeze(1) for x in out.hidden_states], dim=1)
    # → (B=1, n_total_layers, n_tokens, hidden_size)

    # Token aggregation: mean over all tokens (matches neuralset token_aggregation="mean")
    states = states[0].mean(dim=1)  # (n_total_layers, hidden_size)

    # Layer aggregation: group_mean matching neuralset's _aggregate_layers logic
    n = states.shape[0]
    layer_fractions = [0.5, 0.75, 1.0]
    indices = [int(f * (n - 1)) for f in layer_fractions]  # e.g. [20, 30, 40] for n=41
    indices[-1] += 1  # make last index exclusive for slicing
    feats = torch.stack([
        states[i1:i2].mean(0) for i1, i2 in zip(indices[:-1], indices[1:])
    ])  # (2, hidden_size)

# ── 3. Build minimal batch ─────────────────────────────────────────────────
# FmriEncoderModel.forward reads batch.data: dict[modality → (B, L, D, T)]
T = model.n_output_timesteps
n_layers, feat_dim = model.feature_dims["video"]  # (2, 1408)

video_feats = feats.unsqueeze(0).unsqueeze(-1).expand(1, n_layers, feat_dim, T).contiguous()

# Only include video — omit audio/text entirely.
# aggregate_features creates hard zeros (bypassing projectors) for absent modalities,
# which matches what the demo does when there are no Audio/Word events.
batch_data = {"video": video_feats}

batch = SimpleNamespace(data=batch_data)

# ── 4. Forward pass ────────────────────────────────────────────────────────
with torch.inference_mode():
    preds = model(batch)  # (1, n_vertices, T)

brain_response = preds[0].cpu().numpy()  # (n_vertices ≈ 20484, T=40)
print("Output shape:", brain_response.shape)

np.save("brain_response.npy", brain_response)
print("Saved to brain_response.npy")
