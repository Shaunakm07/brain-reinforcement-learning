"""
Run single-image inference through the official TRIBE v2 demo path.

Why this exists
---------------
`run_image.py` performs a simplified direct forward pass that expands one
video feature vector across all TRs. This script instead routes inference
through `TribeModel.get_events_dataframe(...)` + `TribeModel.predict(...)`,
which matches the demo/event pipeline behavior.

Usage:
    python run_image_demo.py path/to/image.jpg

Outputs:
    - brain_response_demo.npy        (n_kept_segments, n_vertices)
    - brain_response_demo_segments.csv
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tribev2.demo_utils import TribeModel

# Keep fps * duration = 64 frames to align with vjepa2-vitg-fpc64-256.
FPS = 16
DURATION_SEC = 4


def write_static_video(image_path: Path, out_mp4: Path, fps: int = FPS, duration: int = DURATION_SEC) -> None:
    """Write an MP4 where every frame is the provided image."""
    import imageio.v2 as imageio

    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    n_frames = fps * duration

    writer = imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for _ in range(n_frames):
            writer.append_data(image)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image inference via TRIBE demo pipeline")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"), help="Cache folder for TribeModel")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("brain_response_demo.npy"),
        help="Output .npy path for predictions",
    )
    parser.add_argument(
        "--segments-out",
        type=Path,
        default=Path("brain_response_demo_segments.csv"),
        help="Output CSV path for segment metadata",
    )
    parser.add_argument(
        "--keep-temp-video",
        action="store_true",
        help="If set, keep the generated temporary MP4 next to output files",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(args.cache_dir))

    with tempfile.TemporaryDirectory(prefix="tribe_demo_image_") as tmpdir:
        tmp_video = Path(tmpdir) / f"{image_path.stem}_static.mp4"
        write_static_video(image_path, tmp_video)

        events = model.get_events_dataframe(video_path=str(tmp_video))
        preds, segments = model.predict(events, verbose=True)

        np.save(args.out, preds)
        print(f"Saved predictions to {args.out} with shape {preds.shape}")

        with open(args.segments_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["i", "timeline", "offset", "duration", "n_events"],
            )
            writer.writeheader()
            for i, seg in enumerate(segments):
                writer.writerow(
                    {
                        "i": i,
                        "timeline": getattr(seg, "timeline", ""),
                        "offset": float(getattr(seg, "offset", np.nan)),
                        "duration": float(getattr(seg, "duration", np.nan)),
                        "n_events": len(getattr(seg, "ns_events", [])),
                    }
                )
        print(f"Saved segment metadata to {args.segments_out}")

        if args.keep_temp_video:
            keep_path = args.out.with_suffix(".input_static.mp4")
            keep_path.write_bytes(tmp_video.read_bytes())
            print(f"Kept generated static video at {keep_path}")


if __name__ == "__main__":
    main()
