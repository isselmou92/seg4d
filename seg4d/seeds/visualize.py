"""Render preview images of seed prompts for visual inspection."""

from __future__ import annotations

import glob
import os
from typing import Any

from PIL import Image, ImageDraw

from .schema import parse_organ_seed


ORGAN_COLORS_RGB = {
    "liver": (255, 80, 80),
    "kidney_right": (200, 80, 255),
    "kidney_left": (80, 80, 255),
}


def render_seed_previews(
    seed_data: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    seed_dir: str,
) -> None:
    """Save one preview image per dynamic sequence overlaying boxes + points.

    Used by both the template generator and the interactive picker.
    """
    for seq_name, organ_configs in seed_data.items():
        if seq_name not in metadata.get("dynamic", {}):
            continue
        jpeg_dir = metadata["dynamic"][seq_name]["jpeg_dir"]
        first_frame = os.path.join(jpeg_dir, "00000.jpg")
        if not os.path.exists(first_frame):
            continue

        img = Image.open(first_frame).convert("RGB")
        draw = ImageDraw.Draw(img)

        for organ_name, organ_cfg in organ_configs.items():
            seed = parse_organ_seed(organ_cfg)
            color = ORGAN_COLORS_RGB.get(organ_name, (200, 200, 200))

            if seed.box:
                draw.rectangle(seed.box, outline=color, width=2)
                draw.text((seed.box[0], seed.box[1] - 12), organ_name, fill=color)

            if seed.positive_points:
                for pt in seed.positive_points:
                    x, y = pt
                    r = 4
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 255, 0))
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255))

            if seed.negative_points:
                for pt in seed.negative_points:
                    x, y = pt
                    r = 4
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
                    draw.line([(x - r, y - r), (x + r, y + r)],
                              fill=(255, 255, 255), width=2)
                    draw.line([(x - r, y + r), (x + r, y - r)],
                              fill=(255, 255, 255), width=2)

        out = os.path.join(seed_dir, f"{seq_name}_boxes.jpg")
        img.save(out, quality=95)
        print(f"  Saved preview: {out}")


def visualize_seed_points(
    jpeg_dir: str,
    seq_name: str,
    organ_points: dict[str, tuple[list[list[int]], list[list[int]]]],
    output_dir: str,
) -> None:
    """Save a debug PNG marking positive/negative points on frame 0.

    ``organ_points`` is the in-memory representation used by the SAM3 backend:
    ``{organ_name: (positive_points, negative_points)}``.
    """
    frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpg")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpeg")))
    if not frame_paths:
        return

    img = Image.open(frame_paths[0]).convert("RGB")
    draw = ImageDraw.Draw(img)
    radius = 6

    point_colors = {
        "liver": (0, 200, 0),
        "kidney_right": (0, 100, 255),
        "kidney_left": (255, 165, 0),
    }

    for organ_name, (pos_pts, neg_pts) in organ_points.items():
        color = point_colors.get(organ_name, (255, 255, 255))
        for pt in pos_pts:
            x, y = pt
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=color, outline=(255, 255, 255), width=2)
            draw.text((x + radius + 3, y - 8), f"+{organ_name}", fill=color)
        for pt in neg_pts:
            x, y = pt
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=(255, 0, 0), outline=(255, 255, 255), width=2)
            r2 = radius - 2
            draw.line([x - r2, y - r2, x + r2, y + r2], fill=(255, 255, 255), width=2)
            draw.line([x - r2, y + r2, x + r2, y - r2], fill=(255, 255, 255), width=2)
            draw.text((x + radius + 3, y - 8), f"-{organ_name}", fill=(255, 80, 80))

    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, f"{seq_name}_seed_points.png")
    img.save(vis_path)
    print(f"    Seed point visualization: {vis_path}")
