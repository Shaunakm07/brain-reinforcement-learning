"""
Generate images from a trained BrainGenerator checkpoint and (optionally)
show which brain regions they activate.

Usage
-----
  # Generate 8 images from the final checkpoint
  python -m brain_steer.generate \\
      --checkpoint brain_steer/checkpoints/checkpoint_final.pt \\
      --n 8 \\
      --out brain_steer/generated/

  # Also compute brain activations for each image and print top regions
  python -m brain_steer.generate \\
      --checkpoint brain_steer/checkpoints/checkpoint_final.pt \\
      --n 4 \\
      --analyse \\
      --out brain_steer/generated/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tribev2"))

from brain_steer.generator import BrainGenerator
from brain_steer.pipeline import image_to_brain
from brain_steer.brain_loss import compute_roi_scores


def load_generator(checkpoint_path: str, device: torch.device) -> tuple[BrainGenerator, dict]:
    """
    Load a BrainGenerator from a checkpoint.

    Returns:
        generator: loaded BrainGenerator in eval mode
        config: training config dict from the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    generator = BrainGenerator(latent_dim=config["latent_dim"]).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    print(f"Loaded checkpoint: step {ckpt['step']}")
    print(f"  Target ROIs  : {config['target_rois']}")
    print(f"  Suppress ROIs: {config.get('suppress_rois') or 'none'}")
    return generator, config


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate(
    checkpoint_path: str,
    n: int = 8,
    out_dir: str = "brain_steer/generated",
    analyse: bool = False,
    num_frames: int = 64,
    cache_dir: str = "./cache",
    seed: int = 0,
):
    """
    Sample images from a trained generator.

    Args:
        checkpoint_path: path to a .pt checkpoint from train.py
        n: number of images to generate
        out_dir: directory to save generated images
        analyse: if True, also run TRIBE v2 on each image and print
                 the top-10 activated brain regions
        num_frames: video frames to use when analysing (matches training)
        cache_dir: HuggingFace model cache
        seed: RNG seed for reproducibility
    """
    torch.manual_seed(seed)
    device = _get_device()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generator, config = load_generator(checkpoint_path, device)

    # ── Sample images ─────────────────────────────────────────────────────────
    with torch.no_grad():
        images_01 = generator.sample(n, device=device)  # (n, 3, 224, 224) in [0,1]

    for i, img in enumerate(images_01):
        save_image(img, out_path / f"generated_{i:02d}.png")

    print(f"\nSaved {n} images to {out_path}/")

    # ── Optional brain analysis ────────────────────────────────────────────────
    if analyse:
        print("\nLoading models for brain analysis...")
        from tribev2.demo_utils import TribeModel
        from transformers import AutoModel

        xp    = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir)
        tribe = xp._model.to(device).eval()
        vjepa = AutoModel.from_pretrained(
            "facebook/vjepa2-vitg-fpc64-256",
            output_hidden_states=True,
            cache_dir=cache_dir,
        ).to(device).eval()

        all_results = []
        for i, img in enumerate(images_01):
            with torch.no_grad():
                brain = image_to_brain(img, vjepa, tribe, num_frames=num_frames)
            roi_names, scores = compute_roi_scores(brain)
            top10_idx = scores.topk(10).indices.tolist()
            top10 = [(roi_names[j], round(scores[j].item(), 5)) for j in top10_idx]
            print(f"\nImage {i:02d} — top-10 activated regions:")
            for rank, (name, score) in enumerate(top10, 1):
                tag = " ← TARGET" if name in config["target_rois"] else ""
                print(f"  {rank:2d}. {name:<12s}  {score:+.5f}{tag}")
            all_results.append({"image": f"generated_{i:02d}.png", "top10": top10})

        with open(out_path / "brain_analysis.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBrain analysis saved to {out_path}/brain_analysis.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt checkpoint from train.py")
    parser.add_argument("--n",    type=int, default=8,
                        help="Number of images to generate")
    parser.add_argument("--out",  default="brain_steer/generated")
    parser.add_argument("--analyse", action="store_true",
                        help="Run TRIBE v2 on generated images and print top regions")
    parser.add_argument("--num-frames", type=int, default=64,
                        help="V-JEPA2 frame count for brain analysis")
    parser.add_argument("--cache", default="./cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        checkpoint_path=args.checkpoint,
        n=args.n,
        out_dir=args.out,
        analyse=args.analyse,
        num_frames=args.num_frames,
        cache_dir=args.cache,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
