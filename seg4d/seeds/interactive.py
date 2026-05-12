"""Interactive matplotlib GUI for placing seed points on frame 0.

Controls:
  * Left-click   - add positive point for the active organ
  * Right-click  - add negative point for the active organ
  * 1 / 2 / 3    - switch active organ (liver / kidney_right / kidney_left)
  * Z            - undo last point
  * N            - next sequence (saves current and moves on)
  * Q            - quit (saves current and exits)

When the user prompts only ``60_COR_b`` / ``60_SAG_b``, the points are
automatically copied to the matching ``FB_COR_b`` / ``FB_SAG_b`` sequence
(both share the same imaging plane).
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
from PIL import Image


_ORGAN_LIST = ("liver", "kidney_right", "kidney_left")
_ORGAN_COLORS = {
    "liver": "#00C800",
    "kidney_right": "#0064FF",
    "kidney_left": "#FFA500",
}
_ORGAN_KEYS = {"1": "liver", "2": "kidney_right", "3": "kidney_left"}
_PAIR_MAP = {
    "60_COR_b": "FB_COR_b",
    "60_SAG_b": "FB_SAG_b",
}


def interactive_point_picker(
    metadata: dict[str, Any],
    existing_seed_data: dict[str, Any],
) -> dict[str, Any]:
    """Run the GUI loop and return the updated ``seed_data`` mapping.

    Parameters
    ----------
    metadata
        Loaded ``conversion_metadata.json``.
    existing_seed_data
        Existing seed dict (may be empty); new entries are merged into it.

    Returns
    -------
    dict
        Updated seed mapping in the same shape as ``seed_boxes.json``.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    seed_data = dict(existing_seed_data)

    all_sequences = list(metadata.get("dynamic", {}).keys())
    if not all_sequences:
        print("  No dynamic sequences found.")
        return seed_data

    sequences_to_prompt = [s for s in all_sequences if s in _PAIR_MAP] or list(all_sequences)

    for seq_name in sequences_to_prompt:
        seq_meta = metadata["dynamic"][seq_name]
        jpeg_dir = seq_meta["jpeg_dir"]
        frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpg")))
        if not frame_paths:
            frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpeg")))
        if not frame_paths:
            print(f"  No JPEG frames for {seq_name}, skipping.")
            continue

        frame0 = np.array(Image.open(frame_paths[0]).convert("RGB"))

        points: dict[str, dict[str, list[list[int]]]] = {
            organ: {"positive": [], "negative": []} for organ in _ORGAN_LIST
        }
        state = {"current_organ": "liver", "history": [], "done": False}

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.canvas.manager.set_window_title(f"Seed Points: {seq_name}")

        def redraw():
            ax.clear()
            ax.imshow(frame0)
            ax.set_title(
                f"{seq_name}  |  Active organ: {state['current_organ']}\n"
                "Left=positive  Right=negative  |  1/2/3=organ  Z=undo  N=next  Q=quit",
                fontsize=10,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            for organ in _ORGAN_LIST:
                color = _ORGAN_COLORS[organ]
                for pt in points[organ]["positive"]:
                    ax.plot(pt[0], pt[1], "o", color=color, markersize=10,
                            markeredgecolor="white", markeredgewidth=1.5)
                    ax.annotate(f"+{organ}", (pt[0] + 8, pt[1]),
                                color=color, fontsize=7, fontweight="bold")
                for pt in points[organ]["negative"]:
                    ax.plot(pt[0], pt[1], "x", color="red", markersize=12,
                            markeredgewidth=2.5)
                    ax.plot(pt[0], pt[1], "o", color="red", markersize=10,
                            markeredgecolor="white", markeredgewidth=1.0,
                            fillstyle="none")
                    ax.annotate(f"-{organ}", (pt[0] + 8, pt[1]),
                                color="#FF5050", fontsize=7)

            highlight_color = _ORGAN_COLORS[state["current_organ"]]
            for spine_pos in ("top", "bottom", "left", "right"):
                ax.spines[spine_pos].set_color(highlight_color)
                ax.spines[spine_pos].set_linewidth(3)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            organ = state["current_organ"]
            if event.button == 1:
                points[organ]["positive"].append([x, y])
                state["history"].append((organ, "positive", len(points[organ]["positive"]) - 1))
                print(f"    + {organ} positive: ({x}, {y})")
            elif event.button == 3:
                points[organ]["negative"].append([x, y])
                state["history"].append((organ, "negative", len(points[organ]["negative"]) - 1))
                print(f"    - {organ} negative: ({x}, {y})")
            redraw()

        def on_key(event):
            if event.key in _ORGAN_KEYS:
                state["current_organ"] = _ORGAN_KEYS[event.key]
                print(f"    Switched to: {state['current_organ']}")
                redraw()
            elif event.key == "z":
                if state["history"]:
                    organ, ptype, idx = state["history"].pop()
                    removed = points[organ][ptype].pop(idx)
                    print(f"    Undo {ptype} {organ}: {removed}")
                    redraw()
            elif event.key in ("n", "q"):
                state["done"] = True
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()

        seq_seed: dict[str, Any] = {}
        for organ in _ORGAN_LIST:
            pos = points[organ]["positive"]
            neg = points[organ]["negative"]
            if not pos and not neg:
                seq_seed[organ] = None
            else:
                seq_seed[organ] = {"positive": pos, "negative": neg}
        seed_data[seq_name] = seq_seed
        print(f"  Saved points for {seq_name}")

        if seq_name in _PAIR_MAP and _PAIR_MAP[seq_name] in all_sequences:
            paired = _PAIR_MAP[seq_name]
            seed_data[paired] = dict(seq_seed)
            print(f"  Copied points to paired sequence: {paired}")

    return seed_data
