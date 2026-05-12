"""Image resampling helpers.

Volumetric MR sequences acquired with thick slabs (e.g. 20mm sagittal slices,
10 slices per timepoint) starve TotalSegmentator of 3D context. We therefore
resample to near-isotropic voxels before segmentation and resample the
resulting label map back onto the original grid afterwards.
"""

from __future__ import annotations

import SimpleITK as sitk

from ..constants import TARGET_ISOTROPIC_SPACING_MM


def resample_to_isotropic(
    input_path: str,
    target_spacing_mm: float = TARGET_ISOTROPIC_SPACING_MM,
) -> str:
    """Resample a NIfTI to near-isotropic spacing using B-spline interpolation.

    If the input is already close to isotropic (the largest spacing is at
    most ``1.5 * target_spacing_mm``), the original path is returned
    unchanged.

    Parameters
    ----------
    input_path
        Path to the input NIfTI volume.
    target_spacing_mm
        Desired upper bound on voxel spacing (mm) along each axis.

    Returns
    -------
    str
        Path to the resampled volume. Either the original ``input_path`` or
        a sibling file ending in ``_isotropic.nii.gz``.
    """
    img = sitk.ReadImage(input_path)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    if max(original_spacing) <= target_spacing_mm * 1.5:
        return input_path

    new_spacing = [min(s, target_spacing_mm) for s in original_spacing]
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled = resampler.Execute(img)

    resampled_path = input_path.replace(".nii.gz", "_isotropic.nii.gz")
    sitk.WriteImage(resampled, resampled_path)

    print(f"    Resampled: {original_size} @ {[round(s, 2) for s in original_spacing]}mm "
          f"-> {new_size} @ {[round(s, 2) for s in new_spacing]}mm")
    return resampled_path


def resample_seg_to_original(seg_path: str, reference_path: str) -> str:
    """Resample a segmentation back to the grid of ``reference_path``.

    Uses nearest-neighbour interpolation to preserve label values. When the
    segmentation already matches the reference geometry it is returned
    unchanged.

    The segmentation is overwritten in place and ``seg_path`` is returned.
    """
    seg_img = sitk.ReadImage(seg_path)
    ref_img = sitk.ReadImage(reference_path)

    if seg_img.GetSize() == ref_img.GetSize() and seg_img.GetSpacing() == ref_img.GetSpacing():
        return seg_path

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampled = resampler.Execute(seg_img)

    sitk.WriteImage(resampled, seg_path)
    return seg_path
