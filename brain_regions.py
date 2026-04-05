"""
Brain region analysis using the HCP Multi-Modal Parcellation (MMP) atlas.

The HCP MMP atlas defines 180 cortical areas per hemisphere (360 total) on the
fsaverage surface. These tools let you:
  - List all available brain regions
  - Extract vertex indices for named regions (with wildcard support)
  - Compute per-region mean activations
  - Rank regions by activation

Atlas notes:
  - Regions are bilateral: each name (e.g. "V1", "MT") covers both hemispheres
    unless you specify hemi="left" or hemi="right".
  - Vertex layout: indices 0..N-1 are left hemisphere, N..2N-1 are right hemisphere,
    where N = FSAVERAGE_SIZES["fsaverage5"] = 10242.
  - ROI names support prefix wildcards ("V*") and suffix wildcards ("*Belt").

Common region groups:
  VISUAL_CORE   = ["V1", "V2", "V3", "V4"]
  VISUAL_DORSAL = ["V3A", "V3B", "V6", "V6A", "V7", "IPS1"]
  VISUAL_VENTRAL= ["V8", "VVC", "FFC", "PIT", "VMV1", "VMV2", "VMV3"]
  MOTION        = ["MT", "MST", "V4t", "FST", "LO1", "LO2", "LO3"]
  AUDITORY      = ["A1", "LBelt", "MBelt", "PBelt", "RI", "A4", "A5"]
  LANGUAGE      = ["44", "45", "STGa", "STSda", "STSdp", "TA2"]
  DEFAULT_MODE  = ["PCC", "RSC", "d23ab", "v23ab", "POS1", "POS2"]
  FRONTAL_EYE   = ["FEF", "PEF", "SCEF"]

Usage (command-line, prints top-20 ROIs):
    python brain_regions.py [brain_response.npy]

Usage (as a module):
    from brain_regions import get_roi_activation, list_regions, top_regions

    # All vertex indices for V1 (both hemispheres)
    idx = get_roi_indices("V1")

    # Mean activation at each TR for V1
    v1_timeseries = get_roi_timeseries(brain, "V1")

    # DataFrame ranking all regions by mean activation
    df = top_regions(brain, k=20)
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "tribev2"))
from tribev2.utils import (
    get_hcp_labels,
    get_hcp_roi_indices,
    get_hcp_vertex_labels,
    get_topk_rois,
    summarize_by_roi,
)

MESH = "fsaverage5"


# ── Named region groups ────────────────────────────────────────────────────────

REGION_GROUPS = {
    "visual_core":    ["V1", "V2", "V3", "V4"],
    "visual_dorsal":  ["V3A", "V3B", "V6", "V6A", "V7", "IPS1"],
    "visual_ventral": ["V8", "VVC", "FFC", "PIT", "VMV1", "VMV2", "VMV3"],
    "motion":         ["MT", "MST", "V4t", "FST", "LO1", "LO2", "LO3"],
    "auditory":       ["A1", "LBelt", "MBelt", "PBelt", "RI", "A4", "A5"],
    "language":       ["44", "45", "STGa", "STSda", "STSdp", "TA2"],
    "default_mode":   ["RSC", "d23ab", "v23ab", "POS1", "POS2", "PCV"],
    "frontal_eye":    ["FEF", "PEF", "SCEF"],
    "somatosensory":  ["3a", "3b", "1", "2"],
    "motor":          ["4", "6a", "6d", "6v", "6r"],
    "prefrontal":     ["46", "9a", "9m", "9p", "10r", "10v", "10d", "10pp"],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def list_regions(hemi: str = "both", combine: bool = False) -> list[str]:
    """Return all HCP MMP region names available on the given hemisphere(s).

    Parameters
    ----------
    hemi : {"both", "left", "right"}
    combine : bool
        If True, uses the coarser 22-region combined atlas instead of the
        full 180-region atlas.

    Returns
    -------
    list[str]
        Sorted list of region names.
    """
    labels = get_hcp_labels(mesh=MESH, combine=combine, hemi=hemi)
    return sorted(labels.keys())


def get_roi_indices(
    roi: Union[str, list[str]],
    hemi: str = "both",
) -> np.ndarray:
    """Return the fsaverage5 vertex indices belonging to one or more named ROIs.

    Supports wildcards:
      "V*"     → all regions whose name starts with "V"
      "*Belt"  → all regions whose name ends with "Belt"

    Parameters
    ----------
    roi : str or list[str]
        Region name(s). Wildcards "*" are supported at the start or end.
    hemi : {"both", "left", "right"}
        Which hemisphere(s) to include.

    Returns
    -------
    np.ndarray
        1-D integer array of vertex indices (0-based, into the full
        left+right surface array of shape (2*N_vertices,)).
    """
    return get_hcp_roi_indices(roi, hemi=hemi, mesh=MESH)


def get_roi_activation(
    brain: np.ndarray,
    roi: Union[str, list[str]],
    hemi: str = "both",
    reduction: str = "mean",
) -> float:
    """Compute the scalar activation of an ROI averaged over all TRs and vertices.

    Parameters
    ----------
    brain : np.ndarray, shape (n_vertices, T)
        Brain response as returned by run_image.py (loaded from brain_response.npy).
    roi : str or list[str]
        Region name(s). Wildcards supported.
    hemi : {"both", "left", "right"}
    reduction : {"mean", "median", "max", "std"}
        How to reduce across the vertex × TR matrix.

    Returns
    -------
    float
    """
    idx = get_roi_indices(roi, hemi=hemi)
    data = brain[idx, :]
    fn = {"mean": np.mean, "median": np.median, "max": np.max, "std": np.std}[reduction]
    return float(fn(data))


def get_roi_timeseries(
    brain: np.ndarray,
    roi: Union[str, list[str]],
    hemi: str = "both",
    reduction: str = "mean",
) -> np.ndarray:
    """Return the activation timeseries for a brain region, one value per TR.

    Parameters
    ----------
    brain : np.ndarray, shape (n_vertices, T)
    roi : str or list[str]
    hemi : {"both", "left", "right"}
    reduction : {"mean", "median", "max", "std"}
        How to reduce across vertices at each TR.

    Returns
    -------
    np.ndarray, shape (T,)
    """
    idx = get_roi_indices(roi, hemi=hemi)
    data = brain[idx, :]  # (n_roi_vertices, T)
    fn = {"mean": np.mean, "median": np.median, "max": np.max, "std": np.std}[reduction]
    return fn(data, axis=0)


def get_group_timeseries(
    brain: np.ndarray,
    group: str,
    hemi: str = "both",
    reduction: str = "mean",
) -> np.ndarray:
    """Return the timeseries for a named region group (see REGION_GROUPS).

    Parameters
    ----------
    brain : np.ndarray, shape (n_vertices, T)
    group : str
        Key from REGION_GROUPS (e.g. "visual_core", "auditory").
    hemi : {"both", "left", "right"}
    reduction : {"mean", "median", "max", "std"}

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if group not in REGION_GROUPS:
        raise ValueError(
            f"Unknown group '{group}'. Available: {list(REGION_GROUPS.keys())}"
        )
    rois = REGION_GROUPS[group]
    return get_roi_timeseries(brain, rois, hemi=hemi, reduction=reduction)


def top_regions(
    brain: np.ndarray,
    k: int = 20,
    hemi: str = "both",
    reduction: str = "mean",
) -> pd.DataFrame:
    """Rank all HCP MMP regions by activation strength.

    Parameters
    ----------
    brain : np.ndarray, shape (n_vertices, T)
    k : int
        Number of top regions to return (use k=-1 for all 181 regions).
    hemi : {"both", "left", "right"}
    reduction : {"mean", "max", "std"}
        Statistic used to score each region. "mean" aggregates over both
        vertices and TRs; "max" takes the vertex-maximum then TR-mean.

    Returns
    -------
    pd.DataFrame with columns [rank, region, score, n_vertices]
        Sorted descending by score.
    """
    mean_brain = brain.mean(axis=1)  # (n_vertices,)
    labels = get_hcp_labels(mesh=MESH, combine=False, hemi=hemi)
    rows = []
    for roi_name, vertices in labels.items():
        verts_in_brain = vertices[vertices < brain.shape[0]]
        if len(verts_in_brain) == 0:
            continue
        data = brain[verts_in_brain, :]  # (n_roi_verts, T)
        if reduction == "mean":
            score = float(data.mean())
        elif reduction == "max":
            score = float(data.max(axis=0).mean())
        elif reduction == "std":
            score = float(data.std())
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")
        rows.append({
            "region": roi_name,
            "score": score,
            "n_vertices": len(verts_in_brain),
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    if k > 0:
        df = df.head(k)
    return df


def region_summary(
    brain: np.ndarray,
    rois: Union[str, list[str]],
    hemi: str = "both",
) -> pd.DataFrame:
    """Return a summary table for a specific set of regions.

    Parameters
    ----------
    brain : np.ndarray, shape (n_vertices, T)
    rois : str or list[str]
        Region names (wildcards supported).
    hemi : {"both", "left", "right"}

    Returns
    -------
    pd.DataFrame with columns [region, mean, std, max, n_vertices]
    """
    if isinstance(rois, str):
        rois = [rois]
    rows = []
    for roi in rois:
        idx = get_roi_indices(roi, hemi=hemi)
        data = brain[idx, :]
        rows.append({
            "region": roi,
            "mean": float(data.mean()),
            "std": float(data.std()),
            "max": float(data.max()),
            "peak_tr": int(data.mean(axis=0).argmax()),
            "n_vertices": len(idx),
        })
    return pd.DataFrame(rows)


# ── Command-line entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    npy_path = sys.argv[1] if len(sys.argv) > 1 else "brain_response.npy"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    brain = np.load(npy_path)
    print(f"Loaded {npy_path}: {brain.shape[0]} vertices × {brain.shape[1]} TRs")
    print()

    df = top_regions(brain, k=k)
    print(f"Top {k} most activated HCP MMP regions (mean across vertices and TRs):")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print()

    print("Region group summaries:")
    rows = []
    for group_name, rois in REGION_GROUPS.items():
        valid_rois = [r for r in rois if r in list_regions()]
        if not valid_rois:
            continue
        idx = get_roi_indices(valid_rois)
        data = brain[idx, :]
        rows.append({
            "group": group_name,
            "mean": float(data.mean()),
            "std": float(data.std()),
            "peak_tr": int(data.mean(axis=0).argmax()),
            "n_vertices": len(idx),
        })
    group_df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(group_df.to_string(index=False, float_format="{:.4f}".format))
