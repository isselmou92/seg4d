"""TotalSegmentator backend for the volumetric step.

Wraps the ``totalsegmentator`` Python API. Compared to a bare call we add:

  1. Optional B-spline resampling to near-isotropic spacing for thick-slab
     MR sequences (recovers 3D context the network needs).
  2. Nearest-neighbour resampling of the segmentation back onto the
     original grid.
  3. Per-organ morphological clean-up via
     :func:`seg4d.postprocess.morphology.postprocess_organ_mask`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from ..constants import TOTALSEG_LABEL_MAP
from ..geometry import resample_seg_to_original, resample_to_isotropic
from ..postprocess import postprocess_organ_mask


class TotalSegmentatorBackend:
    """Per-timepoint volumetric segmenter."""

    def __init__(
        self,
        device: str = "gpu",
        task: str = "total_mr",
        label_map: dict[str, int] = TOTALSEG_LABEL_MAP,
    ) -> None:
        self.device = device
        self.task = task
        self.label_map = label_map

    def segment_volume(
        self,
        nifti_path: str,
        seg_output: str,
    ) -> np.ndarray:
        """Run TotalSegmentator on a single 3D volume and refine the labels.

        Parameters
        ----------
        nifti_path
            Input volumetric NIfTI for one timepoint.
        seg_output
            Where to save the refined multilabel segmentation NIfTI.

        Returns
        -------
        numpy.ndarray
            The refined ``seg_data_refined`` label map (same grid as the
            input volume).
        """
        from totalsegmentator.python_api import totalsegmentator  # type: ignore

        iso_path = resample_to_isotropic(nifti_path)
        iso_seg_output = (
            seg_output.replace("_seg.nii.gz", "_seg_iso.nii.gz")
            if iso_path != nifti_path
            else seg_output
        )

        totalsegmentator(
            input=iso_path,
            output=iso_seg_output,
            ml=True,
            task=self.task,
            device=self.device,
            quiet=True,
        )

        if iso_path != nifti_path:
            resample_seg_to_original(iso_seg_output, nifti_path)
            seg_img = nib.load(iso_seg_output)
            seg_data = np.asanyarray(seg_img.dataobj)
            nib.save(nib.Nifti1Image(seg_data, seg_img.affine, seg_img.header), seg_output)
            os.remove(iso_seg_output)
            os.remove(iso_path)

        seg_img = nib.load(seg_output)
        seg_data_raw = np.asanyarray(seg_img.dataobj)
        seg_data_refined = np.zeros_like(seg_data_raw)

        results_dir = os.path.dirname(seg_output)
        tp_name = Path(seg_output).stem.replace(".nii", "").replace("_seg", "")

        for organ_name, label_id in self.label_map.items():
            raw_mask = (seg_data_raw == label_id).astype(np.uint8)
            refined_mask = postprocess_organ_mask(raw_mask, organ_name)
            seg_data_refined[refined_mask > 0] = label_id

            organ_path = os.path.join(results_dir, f"{tp_name}_{organ_name}.nii.gz")
            nib.save(nib.Nifti1Image(refined_mask, seg_img.affine), organ_path)

        nib.save(nib.Nifti1Image(seg_data_refined, seg_img.affine, seg_img.header), seg_output)

        voxels_raw = {k: int((seg_data_raw == v).sum()) for k, v in self.label_map.items()}
        voxels_ref = {k: int((seg_data_refined == v).sum()) for k, v in self.label_map.items()}
        print(f"    -> {seg_output}")
        print(f"       Raw voxels:      {voxels_raw}")
        print(f"       Refined voxels:  {voxels_ref}")

        return seg_data_refined

    def segment_timepoints(
        self,
        nifti_paths: Iterable[str],
        results_base: str,
    ) -> list[str]:
        """Segment multiple volumetric timepoints in sequence.

        Parameters
        ----------
        nifti_paths
            Input NIfTIs (one per timepoint).
        results_base
            Output directory (created if missing).

        Returns
        -------
        list[str]
            Paths to the per-timepoint segmentation NIfTIs in input order.
            Existing outputs are skipped.
        """
        os.makedirs(results_base, exist_ok=True)
        nifti_list = list(nifti_paths)
        seg_paths: list[str] = []

        for i, nifti_path in enumerate(nifti_list):
            tp_name = Path(nifti_path).stem.replace(".nii", "")
            seg_output = os.path.join(results_base, f"{tp_name}_seg.nii.gz")

            if os.path.exists(seg_output):
                print(f"  [{i + 1}/{len(nifti_list)}] {tp_name} -- already exists, skipping")
                seg_paths.append(seg_output)
                continue

            print(f"  [{i + 1}/{len(nifti_list)}] Segmenting {tp_name}...")
            self.segment_volume(nifti_path, seg_output)
            seg_paths.append(seg_output)

        print(f"\n  Completed TotalSegmentator on {len(seg_paths)} timepoints")
        return seg_paths
