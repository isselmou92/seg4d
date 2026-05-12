"""High-level pipeline orchestration.

The CLI in :mod:`seg4d.cli` is a thin layer over :func:`run_pipeline`. The
function returns the partial state collected across the requested steps so
it can also be used programmatically from a notebook.
"""

from __future__ import annotations

from typing import Any

from .steps import run_convert, run_dynamic, run_reassemble, run_volumetric


VALID_STEPS: tuple[str, ...] = ("convert", "volumetric", "dynamic", "reassemble")


def _expand_steps(steps: list[str]) -> list[str]:
    """Resolve ``["all"]`` to the canonical step list and validate input."""
    if "all" in steps:
        return list(VALID_STEPS)
    unknown = [s for s in steps if s not in VALID_STEPS]
    if unknown:
        raise ValueError(
            f"Unknown step(s): {unknown}. Valid: {list(VALID_STEPS)} or 'all'."
        )
    return list(steps)


def run_pipeline(
    data_dir: str,
    steps: list[str],
    *,
    model: str = "sam3",
    device: str = "gpu",
    seed_mode: str = "boxes",
    seed_file: str | None = None,
    seed_frames: list[int] | None = None,
    interactive: bool = False,
    backend_kwargs: dict[str, Any] | None = None,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Run a sequence of pipeline steps end-to-end.

    Parameters
    ----------
    data_dir
        Patient data directory (must contain a ``dicom/`` subfolder).
    steps
        Subset of ``["convert", "volumetric", "dynamic", "reassemble"]``
        or the literal ``["all"]``.
    model
        Backend used by the dynamic step (``"medsam2"`` / ``"sam3"``).
    device
        Device for TotalSegmentator (``"gpu"`` / ``"cpu"``).
    seed_mode, seed_file, seed_frames, interactive, backend_kwargs
        Forwarded to :func:`seg4d.steps.dynamic.run_dynamic`.
    results_dir
        Output directory for dynamic + reassemble results, relative to
        ``data_dir`` or absolute. Defaults to ``"results"``. Use a unique
        name per model when benchmarking, e.g. ``"results_sam3"``.
        The volumetric (TotalSegmentator) outputs always land in ``"results"``
        regardless of this setting — they are shared across model runs.

    Returns
    -------
    dict
        ``{step_name: step_result}`` for every step that ran.
    """
    expanded = _expand_steps(steps)
    state: dict[str, Any] = {}

    if "convert" in expanded:
        state["convert"] = run_convert(data_dir)

    if "volumetric" in expanded:
        state["volumetric"] = run_volumetric(data_dir, device=device)

    if "dynamic" in expanded:
        state["dynamic"] = run_dynamic(
            data_dir,
            model=model,
            seed_mode=seed_mode,
            seed_file=seed_file,
            seed_frames=seed_frames,
            interactive=interactive,
            backend_kwargs=backend_kwargs,
            results_dir=results_dir,
        )

    if "reassemble" in expanded:
        state["reassemble"] = run_reassemble(data_dir, results_dir=results_dir)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    return state
