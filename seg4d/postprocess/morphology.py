"""Morphological post-processing for organ masks.

TotalSegmentator can leave small internal cavities and isolated stray
components, especially on thick-slab MR. We close those gaps and keep only
the largest connected component per organ.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def postprocess_organ_mask(mask_3d: np.ndarray, organ_name: str) -> np.ndarray:
    """Clean up a binary organ mask.

    The behaviour is organ-aware:

      * **liver** - aggressive 3D closing (``iterations=3``, full 26-connected
        structure) followed by hole filling. The liver is large and the most
        common failure mode is internal voids near vessels.
      * **other organs** - lighter closing (``iterations=2``,
        6-connected structure). Hole filling is skipped to avoid bridging
        across the renal collecting system.

    After the morphology stage, only the largest connected component is
    retained.

    Parameters
    ----------
    mask_3d
        Binary (0/1) volume of one organ.
    organ_name
        Used to select the organ-specific morphology recipe.

    Returns
    -------
    numpy.ndarray
        Cleaned ``uint8`` mask of the same shape as ``mask_3d``.
    """
    if mask_3d.sum() == 0:
        return mask_3d.astype(np.uint8)

    if organ_name == "liver":
        struct = ndimage.generate_binary_structure(3, 2)
        closed = ndimage.binary_closing(mask_3d, structure=struct, iterations=3)
        filled = ndimage.binary_fill_holes(closed)
    else:
        struct = ndimage.generate_binary_structure(3, 1)
        filled = ndimage.binary_closing(mask_3d, structure=struct, iterations=2)

    labeled, num_features = ndimage.label(filled)
    if num_features > 1:
        component_sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
        largest = int(np.argmax(component_sizes)) + 1
        filled = labeled == largest

    return filled.astype(np.uint8)
