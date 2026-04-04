"""
Train a small CNN generator to produce images that maximise activation in
target brain regions and suppress others, using TRIBE v2 as the loss oracle.

How training works
------------------
Each step:
  1. Sample z ~ N(0, I)  of shape (1, latent_dim)
  2. Generate image:  img = generator(z)  →  (1, 3, 224, 224), values [-1,1]
  3. Convert to [0, 1] and build a 64-frame video clip
  4. Extract V-JEPA2 features  →  (2, 1408)
  5. Forward TRIBE v2  →  brain activations  (n_vertices, T)
  6. Compute brain_region_loss (softmax cross-entropy over HCP MMP regions)
  7. Backpropagate gradients through TRIBE v2, V-JEPA2, and into the generator
  8. Update generator weights via Adam

V-JEPA2 and TRIBE v2 are frozen throughout — their weights never change.
Only the generator is trained.

Memory note
-----------
Backprop through V-JEPA2 stores intermediate activations for all 40
transformer blocks. This requires ~3 GB of RAM in float32 for 64 frames.
Use --num-frames 8 to reduce this by 8×, or --num-frames 1 for a quick
smoke test (features will differ from the 64-frame training pipeline).

Usage
-----
  # Maximise primary visual cortex
  python -m brain_steer.train --target V1 V2 V3 --steps 200 --out checkpoints/v1/

  # Maximise face areas, suppress early visual cortex
  python -m brain_steer.train \\
      --target FFC STSda STSdp \\
      --suppress V1 V2 \\
      --temperature 0.05 \\
      --steps 500 \\
      --out checkpoints/faces/

  # Maximise auditory cortex
  python -m brain_steer.train --target A1 LBelt MBelt PBelt --steps 300

  # Low-memory mode (fewer frames, faster per step)
  python -m brain_steer.train --target V1 --num-frames 8 --steps 200
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tribev2"))

from brain_steer.generator import BrainGenerator
from brain_steer.brain_loss import brain_region_loss
from brain_steer.pipeline import image_to_brain


# ── Device selection ──────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_frozen_models(cache_dir: str = "../cache", device: torch.device = None):
    """
    Load V-JEPA2 and TRIBE v2, freeze all weights, and move to device.

    Both models are kept in eval mode (no dropout, deterministic outputs).
    Weights are frozen (requires_grad=False) so they are not updated during
    generator training, but gradients still flow THROUGH their operations.

    Returns:
        tribe: FmriEncoderModel
        vjepa: V-JEPA2 AutoModel
        device: torch.device
    """
    if device is None:
        device = _get_device()

    # ── TRIBE v2 ──────────────────────────────────────────────────────────────
    print("Loading TRIBE v2 (facebook/tribev2)...")
    from tribev2.demo_utils import TribeModel
    xp    = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir)
    tribe = xp._model.to(device).eval()
    for p in tribe.parameters():
        p.requires_grad_(False)

    # ── V-JEPA2 ───────────────────────────────────────────────────────────────
    VJEPA_MODEL = "facebook/vjepa2-vitg-fpc64-256"
    print(f"Loading V-JEPA2 ({VJEPA_MODEL})...")
    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(
        VJEPA_MODEL,
        output_hidden_states=True,
        cache_dir=cache_dir,
    ).to(device).eval()
    for p in vjepa.parameters():
        p.requires_grad_(False)

    print(f"Models loaded on {device}.")
    return tribe, vjepa, device


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    target_rois: list[str],
    suppress_rois: list[str] | None = None,
    n_steps: int = 500,
    latent_dim: int = 256,
    lr: float = 1e-4,
    temperature: float = 0.1,
    lambda_suppress: float = 1.0,
    num_frames: int = 64,
    log_interval: int = 10,
    save_interval: int = 50,
    out_dir: str = "brain_steer/checkpoints",
    cache_dir: str = "./cache",
    seed: int = 42,
):
    """
    Train the BrainGenerator to produce images that activate target_rois.

    Args:
        target_rois: HCP MMP region names to maximise (e.g. ["V1", "V2"])
        suppress_rois: regions to explicitly suppress (optional)
        n_steps: number of gradient steps
        latent_dim: generator noise dimension
        lr: Adam learning rate
        temperature: softmax temperature for the loss (lower = more selective)
        lambda_suppress: weight on the explicit suppression term
        num_frames: video frames for V-JEPA2 (64=full pipeline, 8=low-memory)
        log_interval: print stats every N steps
        save_interval: save image + checkpoint every N steps
        out_dir: directory for checkpoints and sample images
        cache_dir: HuggingFace model cache
        seed: random seed for reproducibility
    """
    torch.manual_seed(seed)
    out_path   = Path(out_dir)
    imgs_path  = out_path / "images"
    out_path.mkdir(parents=True, exist_ok=True)
    imgs_path.mkdir(exist_ok=True)

    tribe, vjepa, device = load_frozen_models(cache_dir=cache_dir)

    print(f"\nTarget ROIs  : {target_rois}")
    print(f"Suppress ROIs: {suppress_rois or 'none'}")
    print(f"Temperature  : {temperature}  |  Steps: {n_steps}  |  Frames: {num_frames}")
    print(f"Output dir   : {out_path}\n")

    # ── Generator ─────────────────────────────────────────────────────────────
    generator = BrainGenerator(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Save config alongside outputs
    config = dict(
        target_rois=target_rois,
        suppress_rois=suppress_rois,
        n_steps=n_steps,
        latent_dim=latent_dim,
        lr=lr,
        temperature=temperature,
        lambda_suppress=lambda_suppress,
        num_frames=num_frames,
    )
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history: list[dict] = []
    generator.train()

    pbar = tqdm(range(1, n_steps + 1), desc="Training")
    for step in pbar:
        optimizer.zero_grad()

        # ── Forward pass ──────────────────────────────────────────────────────
        z           = torch.randn(1, latent_dim, device=device)
        img_tanh    = generator(z)               # (1, 3, 224, 224) in [-1, 1]
        img_01      = (img_tanh + 1) / 2         # remap to [0, 1] for pipeline

        # Differentiable: image → V-JEPA2 features → TRIBE v2 → brain
        brain = image_to_brain(
            img_01[0], vjepa, tribe, num_frames=num_frames
        )  # (n_vertices, T), gradient path through vjepa + tribe → generator

        # ── Loss ──────────────────────────────────────────────────────────────
        loss, info = brain_region_loss(
            brain,
            target_rois=target_rois,
            suppress_rois=suppress_rois,
            temperature=temperature,
            lambda_suppress=lambda_suppress,
        )

        loss.backward()
        # Clip gradients to prevent exploding values early in training
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history.append({"step": step, **info})

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_interval == 0:
            top3 = ", ".join(f"{r}={s:.4f}" for r, s in info["top5_regions"][:3])
            pbar.set_postfix({
                "loss":   f"{info['loss']:.4f}",
                "p_tgt":  f"{info['target_prob']:.4f}",
                "top3":   top3,
            })

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % save_interval == 0:
            save_image(img_01, imgs_path / f"step_{step:04d}.png")
            torch.save({
                "step": step,
                "generator_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss_history": history,
            }, out_path / f"checkpoint_step{step:04d}.pt")

    # ── Final outputs ─────────────────────────────────────────────────────────
    torch.save({
        "step": n_steps,
        "generator_state_dict": generator.state_dict(),
        "config": config,
        "loss_history": history,
    }, out_path / "checkpoint_final.pt")

    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Outputs saved to {out_path}/")
    print(f"  images/          — sample images every {save_interval} steps")
    print(f"  checkpoint_final.pt — final generator weights")
    print(f"  training_history.json — loss/activation log")
    return generator, history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target",     nargs="+", required=True,
                        help="HCP MMP regions to maximise, e.g. V1 V2 FFC")
    parser.add_argument("--suppress",   nargs="*", default=None,
                        help="HCP MMP regions to explicitly suppress")
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--temperature",type=float, default=0.1,
                        help="Softmax temperature (lower = more selective)")
    parser.add_argument("--lambda-suppress", type=float, default=1.0,
                        help="Weight on explicit suppression term")
    parser.add_argument("--latent-dim", type=int,   default=256)
    parser.add_argument("--num-frames", type=int,   default=64,
                        help="Video frames for V-JEPA2. Use 8 for lower memory.")
    parser.add_argument("--out",   default="brain_steer/checkpoints")
    parser.add_argument("--cache", default="./cache",
                        help="HuggingFace model cache directory")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--log-interval",  type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    train(
        target_rois=args.target,
        suppress_rois=args.suppress,
        n_steps=args.steps,
        latent_dim=args.latent_dim,
        lr=args.lr,
        temperature=args.temperature,
        lambda_suppress=args.lambda_suppress,
        num_frames=args.num_frames,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        out_dir=args.out,
        cache_dir=args.cache,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
