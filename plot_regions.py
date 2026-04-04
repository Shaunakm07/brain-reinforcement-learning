"""
Brain region activation plots from TRIBE v2 output.

Requires the same dependencies as plot_brain.py:
    pip install -e "tribev2/.[plotting]"

Usage:
    python plot_regions.py [brain_response.npy] [output_dir]

Outputs (saved to output_dir, default: ./plots/):
    top_regions_bar.png     — horizontal bar chart of top-30 HCP regions by mean activation
    group_comparison.png    — grouped bar chart comparing region-group activations
    group_timeseries.png    — per-TR timeseries for each named region group
    roi_brain_map.png       — cortical surface map with top regions labelled
    visual_timeseries.png   — detailed timeseries for visual hierarchy (V1→MT→higher)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "tribev2"))

from brain_regions import (
    REGION_GROUPS,
    get_roi_indices,
    get_roi_timeseries,
    get_group_timeseries,
    list_regions,
    top_regions,
)
from tribev2.plotting.cortical import PlotBrainNilearn

# ── Load data ──────────────────────────────────────────────────────────────────
npy_path = sys.argv[1] if len(sys.argv) > 1 else "brain_response.npy"
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "plots")
out_dir.mkdir(parents=True, exist_ok=True)

brain = np.load(npy_path)  # (n_vertices, T)
n_vertices, T = brain.shape
trs = np.arange(T)
print(f"Loaded {npy_path}: {n_vertices} vertices × {T} TRs")

all_regions = list_regions()

plotter = PlotBrainNilearn(mesh="fsaverage5", inflate="half", bg_map="sulcal")

# ── Palette ────────────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "visual_core":    "#e63946",
    "visual_dorsal":  "#f4a261",
    "visual_ventral": "#e9c46a",
    "motion":         "#2a9d8f",
    "auditory":       "#457b9d",
    "language":       "#6a4c93",
    "default_mode":   "#a8dadc",
    "frontal_eye":    "#606c38",
    "somatosensory":  "#bc6c25",
    "motor":          "#8ecae6",
    "prefrontal":     "#95d5b2",
}


# ── 1. Top-30 regions bar chart ────────────────────────────────────────────────
print("Plotting top regions bar chart...")
df = top_regions(brain, k=30)

# Assign group colours to each region
region_to_group = {}
for grp, rois in REGION_GROUPS.items():
    for roi in rois:
        region_to_group[roi] = grp

bar_colors = [
    GROUP_COLORS.get(region_to_group.get(r, ""), "#bbbbbb") for r in df["region"]
]

fig, ax = plt.subplots(figsize=(9, 10))
bars = ax.barh(df["region"][::-1], df["score"][::-1], color=bar_colors[::-1], edgecolor="none")
ax.set_xlabel("Mean predicted fMRI response", fontsize=11)
ax.set_title("Top 30 most activated HCP MMP brain regions", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for group colours
legend_patches = [
    mpatches.Patch(color=c, label=g.replace("_", " ").title())
    for g, c in GROUP_COLORS.items()
]
legend_patches.append(mpatches.Patch(color="#bbbbbb", label="Other"))
ax.legend(
    handles=legend_patches, fontsize=7, loc="lower right",
    frameon=False, ncol=2,
)
plt.tight_layout()
fig.savefig(out_dir / "top_regions_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 2. Region group comparison bar chart ──────────────────────────────────────
print("Plotting group comparison...")
group_means, group_stds, group_names = [], [], []
for grp, rois in REGION_GROUPS.items():
    valid = [r for r in rois if r in all_regions]
    if not valid:
        continue
    idx = get_roi_indices(valid)
    data = brain[idx, :]
    group_means.append(data.mean())
    group_stds.append(data.std())
    group_names.append(grp.replace("_", "\n"))

order = np.argsort(group_means)[::-1]
group_means = np.array(group_means)[order]
group_stds = np.array(group_stds)[order]
group_names = np.array(group_names)[order]
colors_ord = [list(GROUP_COLORS.values())[i] for i in order]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(group_names))
ax.bar(x, group_means, yerr=group_stds, color=colors_ord, edgecolor="none",
       error_kw=dict(elinewidth=1, ecolor="gray", capsize=3))
ax.set_xticks(x)
ax.set_xticklabels(group_names, fontsize=9)
ax.set_ylabel("Mean predicted fMRI response")
ax.set_title("Mean activation by brain region group", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / "group_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 3. Region group timeseries ─────────────────────────────────────────────────
print("Plotting group timeseries...")
fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
axes_flat = axes.flatten()

for i, (grp, rois) in enumerate(REGION_GROUPS.items()):
    ax = axes_flat[i]
    valid = [r for r in rois if r in all_regions]
    if not valid:
        ax.set_visible(False)
        continue
    ts = get_group_timeseries(brain, grp)
    idx = get_roi_indices(valid)
    per_vertex = brain[idx, :]
    ts_std = per_vertex.std(axis=0)

    color = GROUP_COLORS.get(grp, "steelblue")
    ax.plot(trs, ts, color=color, linewidth=1.5)
    ax.fill_between(trs, ts - ts_std, ts + ts_std, color=color, alpha=0.2)
    ax.set_title(grp.replace("_", " ").title(), fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i >= len(REGION_GROUPS) - 3:
        ax.set_xlabel("TR")

# Hide unused subplots
for j in range(len(REGION_GROUPS), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Predicted fMRI Response by Region Group (mean ± std over vertices)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(out_dir / "group_timeseries.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 4. Brain map labelled with top regions ────────────────────────────────────
print("Plotting labelled brain map...")
mean_act = brain.mean(axis=1)

VIEWS_5 = ["left", "right", "medial_left", "medial_right", "dorsal"]
fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 3),
    subplot_kw={"projection": "3d"},
    gridspec_kw={"wspace": 0, "hspace": 0},
)
top10 = list(top_regions(brain, k=10)["region"])
valid_top = [r for r in top10 if r in all_regions]
for ax, view in zip(axes, VIEWS_5):
    plotter.plot_surf(mean_act, axes=[ax], views=[view], cmap="hot")
    # annotate_rois in tribev2 has an index-offset bug for right-hemisphere views;
    # only annotate left-hemisphere views where it works correctly.
    if view in ("left", "medial_left"):
        plotter.annotate_rois(ax, valid_top[:5], hemi="left", fontsize=5, color="white")

fig.suptitle("Mean Response with Top-10 Active Regions Labelled",
             fontweight="bold", y=1.02)
fig.savefig(out_dir / "roi_brain_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 5. Visual hierarchy timeseries ────────────────────────────────────────────
print("Plotting visual hierarchy timeseries...")
VISUAL_ROIS = [
    ("V1",         "Primary visual",      "#d62728"),
    ("V2",         "V2",                  "#ff7f0e"),
    ("V3",         "V3",                  "#ffd700"),
    ("V4",         "V4",                  "#2ca02c"),
    ("MT",         "MT (motion)",         "#1f77b4"),
    ("MST",        "MST",                 "#9467bd"),
    ("FFC",        "FFC (fusiform)",      "#8c564b"),
    ("VVC",        "VVC (ventral visual)","#e377c2"),
]

valid_visual = [(roi, label, color) for roi, label, color in VISUAL_ROIS
                if roi in all_regions]

fig, ax = plt.subplots(figsize=(11, 5))
for roi, label, color in valid_visual:
    ts = get_roi_timeseries(brain, roi)
    ax.plot(trs, ts, label=label, color=color, linewidth=1.5)

ax.set_xlabel("TR")
ax.set_ylabel("Mean predicted fMRI response")
ax.set_title("Visual hierarchy: predicted response per TR", fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / "visual_timeseries.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── Done ───────────────────────────────────────────────────────────────────────
print(f"\nAll region plots saved to {out_dir}/")
print("  top_regions_bar.png   — top-30 regions ranked by mean activation")
print("  group_comparison.png  — mean activation per region group (bar)")
print("  group_timeseries.png  — per-TR timeseries for each group")
print("  roi_brain_map.png     — brain surface map with top regions labelled")
print("  visual_timeseries.png — visual hierarchy timeseries")
