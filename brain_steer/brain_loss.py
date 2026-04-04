"""
Differentiable brain activation loss using HCP MMP region statistics.

Loss formulation
----------------
Given the TRIBE v2 output `brain` of shape (n_vertices, T):

  1. Compute mean activation per HCP MMP region:
         scores[i]  =  mean( brain[vertices_i, :] )     for each region i

  2. Apply softmax to turn scores into a probability distribution over regions:
         probs  =  softmax( scores / temperature )

  3. Cross-entropy loss — push probability mass onto the target regions:
         ce_loss  =  -log( sum( probs[target_rois] ) )

  4. Optional explicit suppression term — penalise any residual probability
     on regions we want to suppress:
         suppress_loss  =  sum( probs[suppress_rois] )

  5. Total loss:
         loss  =  ce_loss  +  lambda_suppress * suppress_loss

Why softmax cross-entropy?
  The softmax normalises all regions so their probabilities sum to 1.
  Minimising -log(p_target) is equivalent to maximising p_target while
  the normalisation automatically suppresses all other regions — even ones
  not listed in `suppress_rois`. Explicit suppression (`lambda_suppress`)
  gives additional gradient signal to push down specific competing regions.

Temperature (τ)
  • Low  τ (e.g. 0.05): very selective — hard winner-takes-all.
    Good for activating a single precise region.
  • High τ (e.g. 0.5):  soft competition — spread across several regions.
    Good for activating a broader functional area.

Example usage
-------------
    from brain_steer.brain_loss import brain_region_loss

    brain = tribe_model(batch)            # (n_vertices, T), requires_grad
    loss, info = brain_region_loss(
        brain,
        target_rois=["FFC"],             # maximise fusiform face area
        suppress_rois=["V1", "V2"],      # suppress early visual cortex
        temperature=0.05,
    )
    loss.backward()
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "tribev2"))
from tribev2.utils import get_hcp_labels

MESH = "fsaverage5"


@lru_cache(maxsize=4)
def _roi_index_cache(hemi: str = "both") -> tuple[list[str], list[np.ndarray]]:
    """
    Cache the HCP MMP vertex arrays to avoid re-fetching on every forward pass.
    Returns (roi_names, vertex_index_arrays).
    """
    labels = get_hcp_labels(mesh=MESH, combine=False, hemi=hemi)
    names = list(labels.keys())
    vertices = [labels[n] for n in names]
    return names, vertices


def compute_roi_scores(
    brain: torch.Tensor,
    hemi: str = "both",
) -> tuple[list[str], torch.Tensor]:
    """
    Compute mean brain activation per HCP MMP region.

    Args:
        brain: (n_vertices, T) TRIBE v2 output, requires_grad should be True
               for backprop
        hemi: which hemisphere(s) to include ("both", "left", "right")

    Returns:
        roi_names: list of region names, length n_rois
        scores: (n_rois,) tensor of mean activations, gradient-attached
    """
    roi_names, vertex_arrays = _roi_index_cache(hemi)
    mean_act = brain.mean(dim=1)  # (n_vertices,)  average over TRs

    scores = torch.stack([
        mean_act[
            torch.from_numpy(verts.astype(np.int64)).to(brain.device)
        ].mean()
        if len(verts) > 0
        else torch.tensor(0.0, device=brain.device, dtype=brain.dtype)
        for verts in vertex_arrays
    ])  # (n_rois,)

    return roi_names, scores


def brain_region_loss(
    brain: torch.Tensor,
    target_rois: list[str],
    suppress_rois: Optional[list[str]] = None,
    temperature: float = 0.1,
    lambda_suppress: float = 1.0,
    hemi: str = "both",
) -> tuple[torch.Tensor, dict]:
    """
    Softmax cross-entropy loss over HCP MMP brain regions.

    Args:
        brain: (n_vertices, T) TRIBE v2 predicted fMRI activations.
               Must be part of a computation graph (i.e. requires_grad=True
               or descend from a tensor that does).
        target_rois: list of HCP MMP region names to maximise.
                     Use brain_regions.list_regions() to see all 181 options.
        suppress_rois: optional list of regions to explicitly penalise.
                       The cross-entropy term already suppresses non-target
                       regions implicitly; this term gives extra signal for
                       regions you especially want to suppress.
        temperature: softmax temperature (float > 0).
                     Lower → sharper competition. Start at 0.1 and tune.
        lambda_suppress: weight on the explicit suppression term (default 1.0).
        hemi: hemisphere(s) to consider when computing ROI scores.

    Returns:
        loss: scalar tensor (backpropagatable)
        info: dict with logging values:
              - loss, ce_loss, suppress_loss (floats)
              - target_prob: combined probability mass on target regions
              - target_mean_activation: raw mean of target region scores
              - top5_regions: [(name, score), ...] top-5 by raw activation
    """
    roi_names, scores = compute_roi_scores(brain, hemi=hemi)

    # Softmax over all regions
    probs = F.softmax(scores / temperature, dim=0)

    # ── Cross-entropy: maximise target probability ─────────────────────────
    target_mask = torch.tensor(
        [name in target_rois for name in roi_names],
        dtype=torch.bool, device=brain.device,
    )
    if not target_mask.any():
        raise ValueError(
            f"None of the target_rois {target_rois} were found in the HCP MMP atlas. "
            "Use brain_regions.list_regions() to check available names."
        )

    target_prob = probs[target_mask].sum().clamp(min=1e-8)
    ce_loss = -torch.log(target_prob)

    # ── Explicit suppression term ──────────────────────────────────────────
    suppress_loss = torch.zeros(1, device=brain.device, dtype=brain.dtype).squeeze()
    if suppress_rois:
        suppress_mask = torch.tensor(
            [name in suppress_rois for name in roi_names],
            dtype=torch.bool, device=brain.device,
        )
        if suppress_mask.any():
            suppress_loss = probs[suppress_mask].sum()

    loss = ce_loss + lambda_suppress * suppress_loss

    # ── Logging info ───────────────────────────────────────────────────────
    with torch.no_grad():
        top5_idx = scores.topk(5).indices.cpu().tolist()
        info = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "suppress_loss": suppress_loss.item(),
            "target_prob": target_prob.item(),
            "target_mean_activation": scores[target_mask].mean().item(),
            "top5_regions": [(roi_names[i], round(scores[i].item(), 5))
                             for i in top5_idx],
        }

    return loss, info
