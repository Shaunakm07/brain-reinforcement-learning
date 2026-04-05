"""
Detailed plots of the brain response output from run_image.py.

Requires the plotting extras:
    pip install -e "tribev2/.[plotting]"

Usage:
    python plot_brain.py [brain_response.npy] [output_dir]

Outputs (saved to output_dir, default: ./plots/):
    mean_activation.png   — mean response across TRs, 5 cortical views
    peak_tr.png           — the single TR with highest mean activation
    timesteps.png         — time-resolved strip of brain maps (every 5 TRs)
    temporal.png          — mean / std / max activation per TR
    distribution.png      — histogram of per-vertex mean activations
    summary.png           — all panels combined into one figure
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
npy_path = sys.argv[1] if len(sys.argv) > 1 else "brain_response.npy"
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "plots")
out_dir.mkdir(parents=True, exist_ok=True)

brain = np.load(npy_path)  # (n_vertices, T)
n_vertices, T = brain.shape
print(f"Loaded {npy_path}: {n_vertices} vertices × {T} TRs")

# PlotBrainNilearn expects (n_timesteps, n_vertices) for plot_timesteps
brain_t = brain.T  # (T, n_vertices)

# ── Setup plotter ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "tribev2"))
from tribev2.plotting.cortical import PlotBrainNilearn

plotter = PlotBrainNilearn(mesh="fsaverage5", inflate="half", bg_map="sulcal")

VIEWS_5 = ["left", "right", "medial_left", "medial_right", "dorsal"]
CMAP = "hot"


# ── 1. Mean activation across all TRs ─────────────────────────────────────
print("Plotting mean activation...")
mean_act = brain.mean(axis=1)  # (n_vertices,)

fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 3),
    subplot_kw={"projection": "3d"},
    gridspec_kw={"wspace": 0, "hspace": 0},
)
for ax, view in zip(axes, VIEWS_5):
    plotter.plot_surf(mean_act, axes=[ax], views=[view], cmap=CMAP)
fig.suptitle("Mean Predicted fMRI Response (averaged over all TRs)", fontweight="bold", y=1.02)
fig.savefig(out_dir / "mean_activation.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 2. Peak TR ─────────────────────────────────────────────────────────────
print("Plotting peak TR...")
peak_tr = int(brain.mean(axis=0).argmax())
peak_act = brain[:, peak_tr]  # (n_vertices,)

fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 3),
    subplot_kw={"projection": "3d"},
    gridspec_kw={"wspace": 0, "hspace": 0},
)
for ax, view in zip(axes, VIEWS_5):
    plotter.plot_surf(peak_act, axes=[ax], views=[view], cmap=CMAP)
fig.suptitle(f"Peak TR Response (TR {peak_tr}, highest mean activation)", fontweight="bold", y=1.02)
fig.savefig(out_dir / "peak_tr.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 3. Time-resolved strip (lateral left, every 5 TRs) ────────────────────
# Note: tribev2's plot_timesteps creates axes without projection='3d', which
# breaks nilearn. We build the strip manually with proper 3D axes instead.
print("Plotting time-resolved strip...")
step = 5  # T=40, step=5 → 8 panels
n_panels = T // step
fig, axes = plt.subplots(
    1, n_panels,
    figsize=(2.5 * n_panels, 2.5),
    subplot_kw={"projection": "3d"},
    gridspec_kw={"wspace": 0, "hspace": 0},
)
for i, ax in enumerate(axes):
    tr_idx = i * step
    plotter.plot_surf(brain[:, tr_idx], axes=[ax], views=["left"], cmap=CMAP, norm_percentile=99)
    ax.set_title(f"TR {tr_idx}", fontsize=8, pad=0)
fig.suptitle("Predicted Response Over Time (lateral left, every 5 TRs)", fontweight="bold")
fig.savefig(out_dir / "timesteps.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ── 4. Temporal summary ────────────────────────────────────────────────────
print("Plotting temporal summary...")
tr_mean = brain.mean(axis=0)      # (T,)
tr_std = brain.std(axis=0)        # (T,)
tr_max = brain.max(axis=0)        # (T,)
tr_p95 = np.percentile(brain, 95, axis=0)  # (T,)
trs = np.arange(T)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(trs, tr_mean, color="steelblue", linewidth=1.5, label="mean")
axes[0].fill_between(trs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.25, color="steelblue", label="±1 std")
axes[0].set_ylabel("Predicted response")
axes[0].set_title("Mean activation ± std across vertices")
axes[0].legend(frameon=False)

axes[1].plot(trs, tr_p95, color="coral", linewidth=1.5, label="p95")
axes[1].plot(trs, tr_max, color="firebrick", linewidth=1.5, linestyle="--", label="max")
axes[1].set_ylabel("Predicted response")
axes[1].set_title("Upper tail activation (p95 and max) across vertices")
axes[1].legend(frameon=False)

axes[2].plot(trs, tr_std, color="mediumseagreen", linewidth=1.5)
axes[2].set_ylabel("Std")
axes[2].set_xlabel("TR")
axes[2].set_title("Spatial variability per TR (std across vertices)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(out_dir / "temporal.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 5. Distribution of per-vertex mean activations ────────────────────────
print("Plotting distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(mean_act, bins=120, color="steelblue", edgecolor="none", alpha=0.85)
axes[0].axvline(np.percentile(mean_act, 95), color="firebrick", linestyle="--", label="p95")
axes[0].axvline(mean_act.mean(), color="k", linestyle="-", label="mean")
axes[0].set_xlabel("Mean predicted response")
axes[0].set_ylabel("Number of vertices")
axes[0].set_title("Distribution of mean vertex activations")
axes[0].legend(frameon=False)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Cumulative distribution
sorted_act = np.sort(mean_act)
cdf = np.arange(1, n_vertices + 1) / n_vertices
axes[1].plot(sorted_act, cdf, color="steelblue", linewidth=1.5)
axes[1].axhline(0.95, color="firebrick", linestyle="--", linewidth=0.8, label="p95")
axes[1].set_xlabel("Mean predicted response")
axes[1].set_ylabel("Cumulative fraction of vertices")
axes[1].set_title("Cumulative distribution of vertex activations")
axes[1].legend(frameon=False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(out_dir / "distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ── 6. Summary figure ─────────────────────────────────────────────────────
print("Plotting summary figure...")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 5, hspace=0.45, wspace=0.05)

# Row 0: mean activation (5 views)
mean_axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(5)]
for ax, view in zip(mean_axes, VIEWS_5):
    plotter.plot_surf(mean_act, axes=[ax], views=[view], cmap=CMAP)
fig.text(0.01, 0.88, "Mean\nResponse", va="center", ha="left", fontsize=9, color="gray")

# Row 1: peak TR (5 views)
peak_axes = [fig.add_subplot(gs[1, i], projection="3d") for i in range(5)]
for ax, view in zip(peak_axes, VIEWS_5):
    plotter.plot_surf(peak_act, axes=[ax], views=[view], cmap=CMAP)
fig.text(0.01, 0.55, f"Peak\nTR {peak_tr}", va="center", ha="left", fontsize=9, color="gray")

# Row 2, left: temporal mean
ax_temp = fig.add_subplot(gs[2, :3])
ax_temp.plot(trs, tr_mean, color="steelblue", linewidth=1.5)
ax_temp.fill_between(trs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.25, color="steelblue")
ax_temp.set_xlabel("TR")
ax_temp.set_ylabel("Mean response")
ax_temp.set_title("Mean activation over time (±1 std)", fontsize=10)
ax_temp.spines["top"].set_visible(False)
ax_temp.spines["right"].set_visible(False)

# Row 2, right: histogram
ax_hist = fig.add_subplot(gs[2, 3:])
ax_hist.hist(mean_act, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
ax_hist.axvline(np.percentile(mean_act, 95), color="firebrick", linestyle="--", linewidth=0.8)
ax_hist.set_xlabel("Mean response")
ax_hist.set_ylabel("Vertices")
ax_hist.set_title("Vertex activation distribution", fontsize=10)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

fig.suptitle("TRIBE v2 — Predicted fMRI Brain Response to Static Image", fontsize=14, fontweight="bold", y=0.98)
fig.savefig(out_dir / "summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nAll plots saved to {out_dir}/")
print(f"  mean_activation.png  — mean response, 5 views")
print(f"  peak_tr.png          — TR {peak_tr} (peak mean activation), 5 views")
print(f"  timesteps.png        — time-resolved strip, every 5 TRs")
print(f"  temporal.png         — mean/std/max/p95 per TR")
print(f"  distribution.png     — vertex activation histogram + CDF")
print(f"  summary.png          — combined overview")
