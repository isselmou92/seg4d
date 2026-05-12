"""Generate gridded reference frames and a starter ``seed_boxes.json``.

The reference frames are PNG/JPEG images of frame 0 of each dynamic sequence
overlaid with a pixel-coordinate grid; the user reads coordinates off them
to fill in ``seed_boxes.json`` manually (or via the interactive picker).
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image, ImageDraw

from ..io import load_metadata
from .schema import load_seed_file, save_seed_file
from .visualize import render_seed_previews


def generate_seed_template(data_dir: str) -> None:
    """Render gridded reference images and write a starter seed file.

    Parameters
    ----------
    data_dir
        Patient data directory (must already have run the convert step).

    Side effects
    ------------
    * Writes ``<data_dir>/seed_templates/<seq>_reference.jpg`` for every
      dynamic sequence found in the metadata.
    * Creates ``<data_dir>/seed_boxes.json`` with all-zero placeholders if
      it does not already exist.
    * If ``seed_boxes.json`` exists, also renders preview images that
      overlay the configured boxes/points on frame 0.
    """
    print("\n" + "=" * 60)
    print("GENERATE SEED TEMPLATE")
    print("=" * 60)

    metadata = load_metadata(data_dir)

    seed_dir = os.path.join(data_dir, "seed_templates")
    os.makedirs(seed_dir, exist_ok=True)

    template: dict[str, dict[str, Any]] = {}

    for seq_name, seq_meta in metadata.get("dynamic", {}).items():
        jpeg_dir = seq_meta["jpeg_dir"]
        geom = seq_meta["geometry"]
        rows, cols = int(geom["Rows"]), int(geom["Columns"])

        first_frame_path = os.path.join(jpeg_dir, "00000.jpg")
        if not os.path.exists(first_frame_path):
            print(f"  WARNING: No JPEG frames for {seq_name}, skipping")
            continue

        img = Image.open(first_frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        grid_step = 50
        for x in range(0, cols, grid_step):
            draw.line([(x, 0), (x, rows - 1)], fill=(40, 40, 40), width=1)
            draw.text((x + 2, 2), str(x), fill=(100, 200, 100))
        for y in range(0, rows, grid_step):
            draw.line([(0, y), (cols - 1, y)], fill=(40, 40, 40), width=1)
            draw.text((2, y + 2), str(y), fill=(100, 200, 100))

        draw.text(
            (10, rows - 20),
            f"{seq_name} ({cols}x{rows}) - frame 0",
            fill=(255, 255, 0),
        )

        ref_path = os.path.join(seed_dir, f"{seq_name}_reference.jpg")
        img.save(ref_path, quality=95)
        print(f"  Saved reference image: {ref_path}")

        template[seq_name] = {
            "liver": [0, 0, 0, 0],
            "kidney_right": [0, 0, 0, 0],
            "kidney_left": [0, 0, 0, 0],
        }

    template_path = os.path.join(data_dir, "seed_boxes.json")
    if not os.path.exists(template_path):
        save_seed_file(template_path, template)
        print(f"\n  Template saved: {template_path}")
    else:
        print(f"\n  seed_boxes.json already exists, not overwriting.")

    if os.path.exists(template_path):
        existing_seeds = load_seed_file(template_path)
        render_seed_previews(existing_seeds, metadata, seed_dir)

    print(f"  Reference images saved in: {seed_dir}/")
    print()
    print("  HOW TO USE:")
    print("  1. Open the reference images to see pixel coordinates (grid overlay)")
    print("  2. Edit seed_boxes.json -- extended format per organ:")
    print('     {"box": [x1,y1,x2,y2], "positive": [[x,y],...], "negative": [[x,y],...]}')
    print("     or legacy format: [x1, y1, x2, y2]  (box only)")
    print("  3. Set [0,0,0,0] for organs not visible in a sequence")
    print("  4. Run: python -m seg4d run --data_dir ... --steps dynamic --model medsam2 \\")
    print("          --seed_mode boxes --seed_frames 0,250,500")
