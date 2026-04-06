"""
Convert EmbodiedBench EB-Nav trajectory JSON to stable_worldmodel HDF5 (.h5) for LeWM.

Dataset: https://huggingface.co/datasets/EmbodiedBench/EB-Nav_trajectory_dataset

Example (from ``ELEN`` root, with ``h5py`` / ``Pillow`` / ``huggingface_hub`` installed)::

    python -m src.env.navigation.dataset \\
        --json /path/to/EB-Nav_trajectory_dataset/eb-nav_dataset_multi_step.json \\
        --images-root /path/to/EB-Nav_trajectory_dataset \\
        --output eb_nav_train.h5

Place ``eb_nav_train.h5`` under ``$STABLEWM_HOME/datasets/`` (default ``~/.stable-wm/datasets``),
then train LeWM from ``ELEN/lewm``::

    python train.py data=eb_nav
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Iterable

import h5py

try:
    import hdf5plugin  # noqa: F401  # register HDF5 compression filters (compat with swm)
except ImportError:
    pass
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Same order as ELEN/vagen/envs/navigation/benchmark.py VALID_ACTIONS
EB_NAV_ACTION_NAMES: tuple[str, ...] = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
ACTION_DIM = len(EB_NAV_ACTION_NAMES)
_NAME_TO_IDX = {n: i for i, n in enumerate(EB_NAV_ACTION_NAMES)}


def download_eb_nav_repo(local_dir: str | Path) -> Path:
    """Download the full HF dataset repo to ``local_dir`` (JSON + images.zip, etc.)."""
    from huggingface_hub import snapshot_download

    path = Path(local_dir).resolve()
    snapshot_download(
        repo_id="EmbodiedBench/EB-Nav_trajectory_dataset",
        repo_type="dataset",
        local_dir=str(path),
        local_dir_use_symlinks=False,
    )
    return path


def _normalize_action_name(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        raw = raw[1]
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip().lower()
    # common variants
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "moveforward": "moveahead",
        "forward": "moveahead",
        "movebackward": "moveback",
        "backward": "moveback",
        "turnright": "rotateright",
        "turnleft": "rotateleft",
    }
    if s in aliases:
        s = aliases[s]
    return s if s in _NAME_TO_IDX else None


def _resolve_image_path(images_root: Path, p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    if path.is_file():
        return path.resolve()
    cand = (images_root / p).resolve()
    if cand.is_file():
        return cand
    # strip leading segments if JSON stores paths like "images/..."
    parts = path.parts
    if len(parts) > 1:
        cand2 = (images_root / Path(*parts[-2:])).resolve()
        if cand2.is_file():
            return cand2
    return None


def _load_image(
    path: Path,
    resize: tuple[int, int] | None,
) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if resize is not None:
                im = im.resize(resize, Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
    except OSError as e:
        LOGGER.warning("Failed to load image %s: %s", path, e)
        return None
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    return arr


def _one_hot(index: int) -> np.ndarray:
    v = np.zeros((ACTION_DIM,), dtype=np.float32)
    v[index] = 1.0
    return v


def _iter_episode_dicts(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for ep in data:
            if isinstance(ep, dict):
                yield ep
    elif isinstance(data, dict):
        # single dict with "episodes" key, etc.
        if "episodes" in data and isinstance(data["episodes"], list):
            yield from _iter_episode_dicts(data["episodes"])
        else:
            yield data


def _extract_steps_from_trajectory_item(
    item: dict[str, Any],
    images_root: Path,
    resize: tuple[int, int] | None,
    require_action_success: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (pixel_hwc_uint8, action_onehot) for one trajectory segment."""
    plan = item.get("executable_plan")
    if not isinstance(plan, list) or len(plan) == 0:
        return []

    inp = item.get("input_image_path") or ""
    rows: list[tuple[np.ndarray, np.ndarray]] = []

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            continue
        if require_action_success and float(step.get("action_success", 1.0)) < 0.5:
            continue

        act_raw = step.get("action")
        name = _normalize_action_name(act_raw)
        if name is None:
            LOGGER.debug("Skipping step with unknown action: %s", act_raw)
            continue

        if i == 0:
            rel = inp
        else:
            prev = plan[i - 1]
            rel = prev.get("img_path") if isinstance(prev, dict) else None

        if not rel:
            continue
        p = _resolve_image_path(images_root, str(rel))
        if p is None:
            LOGGER.debug("Missing image for %s", rel)
            continue

        pix = _load_image(p, resize)
        if pix is None:
            continue

        rows.append((pix, _one_hot(_NAME_TO_IDX[name])))

    return rows


def _episode_to_steps(
    episode: dict[str, Any],
    images_root: Path,
    resize: tuple[int, int] | None,
    only_success: bool,
    require_action_success: bool,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    if only_success and float(episode.get("success", 0.0)) < 0.5:
        return None

    traj = episode.get("trajectory")
    if not isinstance(traj, list) or len(traj) == 0:
        return None

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for item in traj:
        if not isinstance(item, dict):
            continue
        out.extend(
            _extract_steps_from_trajectory_item(
                item, images_root, resize, require_action_success
            )
        )
    return out if out else None


def _flatten_episodes_single_step(
    episodes: list[dict[str, Any]],
    images_root: Path,
    resize: tuple[int, int] | None,
    only_success: bool,
    require_action_success: bool,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Parse single-step JSON: each episode may mirror multi-step with trajectory list."""
    per_ep: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for ep in episodes:
        steps = _episode_to_steps(ep, images_root, resize, only_success, require_action_success)
        if steps:
            per_ep.append(steps)
    return per_ep


def convert_eb_nav_json_to_hdf5(
    json_path: str | Path,
    images_root: str | Path,
    output_h5_path: str | Path,
    *,
    variant: str = "multi_step",
    only_success: bool = False,
    require_action_success: bool = False,
    resize: tuple[int, int] | None = None,
) -> Path:
    """
    Read EB-Nav JSON and write an HDF5 file compatible with ``stable_worldmodel.HDF5Dataset``.

    Args:
        json_path: Path to ``eb-nav_dataset_multi_step.json`` or ``eb-nav_dataset_single_step.json``.
        images_root: Root directory for resolving relative image paths (unzipped ``images/``).
        output_h5_path: Destination ``.h5`` file path.
        variant: ``\"multi_step\"`` or ``\"single_step\"`` (same parser if structure matches).
        only_success: If True, drop episodes with ``success < 0.5``.
        require_action_success: If True, skip steps with ``action_success < 0.5``.
        resize: Optional ``(width, height)`` for PIL resize; default keeps native resolution.
    """
    json_path = Path(json_path).resolve()
    images_root = Path(images_root).resolve()
    out_path = Path(output_h5_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    episodes = list(_iter_episode_dicts(data))
    if not episodes:
        raise ValueError(f"No episodes found in {json_path}")

    if variant not in ("multi_step", "single_step"):
        warnings.warn(f"Unknown variant {variant!r}, using multi_step parsing.", stacklevel=2)

    per_ep = _flatten_episodes_single_step(
        episodes, images_root, resize, only_success, require_action_success
    )

    if not per_ep:
        raise ValueError(
            "No valid episodes after conversion (check images_root and JSON content)."
        )

    ep_lens = np.array([len(steps) for steps in per_ep], dtype=np.int64)
    ep_offset = np.cumsum(np.concatenate([[0], ep_lens[:-1]])).astype(np.int64)

    total = int(ep_lens.sum())
    pixels = np.empty((total, *per_ep[0][0][0].shape), dtype=np.uint8)
    actions = np.empty((total, ACTION_DIM), dtype=np.float32)

    i = 0
    for steps in per_ep:
        for pix, act in steps:
            pix_i = pixels[i]
            if pix.shape != pix_i.shape:
                raise ValueError(
                    f"Inconsistent image shape {pix.shape} vs {pix_i.shape}; "
                    "use --resize W H or preprocess to a fixed size."
                )
            pixels[i] = pix
            actions[i] = act
            i += 1

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=4)
        f.create_dataset("action", data=actions, compression="gzip", compression_opts=4)

    LOGGER.info(
        "Wrote %s (%d episodes, %d timesteps) to %s",
        out_path.name,
        len(ep_lens),
        total,
        out_path,
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EB-Nav trajectory JSON to LeWM / stable_worldmodel HDF5."
    )
    p.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to eb-nav_dataset_multi_step.json or single_step.json",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Directory used to resolve image paths (folder containing images/ or paths).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .h5 path (or basename under --cache-dir/datasets if no suffix).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="If set and --output has no .h5, write to cache-dir/datasets/<name>.h5",
    )
    p.add_argument(
        "--variant",
        choices=("multi_step", "single_step"),
        default="multi_step",
        help="Which JSON variant (parsing is the same if schema matches).",
    )
    p.add_argument("--only-success", action="store_true", help="Keep only episodes with success.")
    p.add_argument(
        "--require-action-success",
        action="store_true",
        help="Skip per-step rows with action_success < 0.5.",
    )
    p.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Optional resize width and height (two integers).",
    )
    p.add_argument(
        "--download",
        type=Path,
        default=None,
        help="If set, download HF dataset repo to this directory first, then exit.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.download is not None:
        p = download_eb_nav_repo(args.download)
        print(f"Downloaded to {p}")
        return

    out = args.output
    if out.suffix.lower() != ".h5" and args.cache_dir is not None:
        datasets_dir = Path(args.cache_dir).resolve() / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        out = datasets_dir / f"{out.name}.h5"

    resize = tuple(args.resize) if args.resize is not None else None
    convert_eb_nav_json_to_hdf5(
        json_path=args.json,
        images_root=args.images_root,
        output_h5_path=out,
        variant=args.variant,
        only_success=args.only_success,
        require_action_success=args.require_action_success,
        resize=resize,
    )


if __name__ == "__main__":
    main()
