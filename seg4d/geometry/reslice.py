"""Cross-plane reslicing of a 3D segmentation onto a 2D dynamic image grid.

This is the geometric heart of MedSAM2's "reslice" seeding mode: given a
volumetric organ label map and the DICOM geometry of a 2D dynamic frame, it
samples the volume at the location of every dynamic pixel and returns
per-organ 2D boolean masks that can be fed to the video predictor as a seed.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from ..constants import TOTALSEG_LABEL_MAP


def reslice_volume_at_dynamic_plane(
    seg_path: str,
    dynamic_geometry: dict,
    label_map: dict[str, int] = TOTALSEG_LABEL_MAP,
) -> dict[str, np.ndarray]:
    """Reslice a 3D label map at the plane of a 2D dynamic frame.

    Works for any relative orientation between the volumetric and dynamic
    acquisitions (sagittal-to-sagittal, sagittal-to-coronal, ...).

    For each pixel in the dynamic frame, the corresponding patient-space
    coordinate is computed from the DICOM tags, converted from LPS to RAS
    (NIfTI convention), and mapped through the inverse affine of the
    volumetric NIfTI. ``map_coordinates`` then samples the volume with
    nearest-neighbour interpolation.

    Parameters
    ----------
    seg_path
        Path to the volumetric segmentation NIfTI (typically the
        TotalSegmentator output of one timepoint).
    dynamic_geometry
        Geometry dict produced by
        :func:`seg4d.io.dicom_loader.parse_2d_dynamic_dicoms`. Required
        keys: ``Rows``, ``Columns``, ``ImagePositionPatient``,
        ``ImageOrientationPatient``, ``PixelSpacing``.
    label_map
        Mapping ``{organ_name: integer_label}`` of organs to extract.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{organ_name: 2D bool array of shape (Rows, Columns)}``.
    """
    seg_img = nib.load(seg_path)
    seg_data = np.asanyarray(seg_img.dataobj)
    vol_inv_affine = np.linalg.inv(seg_img.affine)

    target_rows = int(dynamic_geometry["Rows"])
    target_cols = int(dynamic_geometry["Columns"])
    dyn_pos = np.array(dynamic_geometry["ImagePositionPatient"], dtype=np.float64)
    dyn_iop = np.array(dynamic_geometry["ImageOrientationPatient"], dtype=np.float64)
    dyn_ps = np.array(dynamic_geometry["PixelSpacing"], dtype=np.float64)

    dyn_row_dir = dyn_iop[:3]
    dyn_col_dir = dyn_iop[3:]

    c_indices = np.arange(target_cols)
    r_indices = np.arange(target_rows)
    cc, rr = np.meshgrid(c_indices, r_indices)

    # DICOM convention: pixel (r, c) -> patient_coord =
    #   origin + c * row_dir * col_spacing + r * col_dir * row_spacing
    patient_coords_lps = (
        dyn_pos[np.newaxis, np.newaxis, :]
        + cc[:, :, np.newaxis] * dyn_row_dir[np.newaxis, np.newaxis, :] * dyn_ps[1]
        + rr[:, :, np.newaxis] * dyn_col_dir[np.newaxis, np.newaxis, :] * dyn_ps[0]
    )

    patient_coords_ras = patient_coords_lps.copy()
    patient_coords_ras[:, :, 0] *= -1  # L -> R
    patient_coords_ras[:, :, 1] *= -1  # P -> A

    flat_coords = patient_coords_ras.reshape(-1, 3)
    homogeneous = np.column_stack([flat_coords, np.ones(flat_coords.shape[0])])
    voxel_coords = (vol_inv_affine @ homogeneous.T)[:3, :]

    print(f"      Reslice diagnostics:")
    print(f"        Seg volume shape: {seg_data.shape}")
    for ax in range(3):
        lo, hi = float(voxel_coords[ax].min()), float(voxel_coords[ax].max())
        print(f"        Voxel axis {ax}: [{lo:.2f}, {hi:.2f}]  "
              f"(valid: 0..{seg_data.shape[ax] - 1})")

    masks: dict[str, np.ndarray] = {}
    for organ_name, label_id in label_map.items():
        organ_vol = (seg_data == label_id).astype(np.float32)
        sampled = map_coordinates(organ_vol, voxel_coords, order=0, mode="nearest")
        masks[organ_name] = sampled.reshape(target_rows, target_cols) > 0.5

    return masks
