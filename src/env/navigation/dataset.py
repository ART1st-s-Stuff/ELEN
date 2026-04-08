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

The same conversion also writes ``instruction`` (UTF-8 per row) and ``is_end`` (float32 0/1).
``is_end`` is ``1`` only on the **last timestep of successful episodes** (``success >= 0.5``); else ``0``.
Re-run conversion after upgrading; old HDF5 files without these datasets cannot train the terminal scorer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import h5py

try:
    import hdf5plugin  # noqa: F401  # register HDF5 compression filters (compat with swm)
except ImportError:
    pass
import numpy as np
from PIL import Image
from tqdm import tqdm

# Pillow >= 9.1: Image.Resampling; older versions use Image.BILINEAR
try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE_BILINEAR = Image.BILINEAR

LOGGER = logging.getLogger(__name__)


def _default_episode_workers() -> int:
    """Parallel episode loaders (I/O + decode); HDF5 writes stay single-threaded."""
    return min(8, max(1, (os.cpu_count() or 4)))


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

# Episode-level instruction keys (EB-Nav / EmbodiedBench JSON)
_INSTRUCTION_EPISODE_KEYS: tuple[str, ...] = (
    "instruction",
    "task_instruction",
    "language_instruction",
    "task",
    "goal",
)
_INSTRUCTION_TRAJECTORY_KEYS: tuple[str, ...] = (
    "instruction",
    "task_instruction",
    "language_instruction",
)


def _episode_instruction(episode: dict[str, Any]) -> str:
    """Best-effort natural-language instruction string for an episode."""
    for key in _INSTRUCTION_EPISODE_KEYS:
        v = episode.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    traj = episode.get("trajectory")
    if isinstance(traj, list) and traj:
        first = traj[0]
        if isinstance(first, dict):
            for key in _INSTRUCTION_TRAJECTORY_KEYS:
                v = first.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


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
    """Map EB-Nav ``action`` field to one of ``EB_NAV_ACTION_NAMES``.

    The released JSON uses ``[action_id, "Move forward by 0.25"]`` (IDs 0–7 per
    dataset README); the free-text name alone does not normalize to our tokens.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
        first = raw[0]
        # EB-Nav uses integer action id 0..7 (bool is a subclass of int in Python)
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            if float(first).is_integer():
                idx = int(first)
                if 0 <= idx < ACTION_DIM:
                    return EB_NAV_ACTION_NAMES[idx]
        if len(raw) >= 2:
            raw = raw[1]
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip().lower()
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
                im = im.resize(resize, _RESAMPLE_BILINEAR)
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
) -> list[tuple[np.ndarray, np.ndarray, str, float]] | None:
    if only_success and float(episode.get("success", 0.0)) < 0.5:
        return None

    traj = episode.get("trajectory")
    if not isinstance(traj, list) or len(traj) == 0:
        return None

    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for item in traj:
        if not isinstance(item, dict):
            continue
        raw.extend(
            _extract_steps_from_trajectory_item(
                item, images_root, resize, require_action_success
            )
        )
    if not raw:
        return None

    instruction = _episode_instruction(episode)
    success = float(episode.get("success", 0.0)) >= 0.5
    n = len(raw)
    out: list[tuple[np.ndarray, np.ndarray, str, float]] = []
    for i, (pix, act) in enumerate(raw):
        is_end = 1.0 if (success and i == n - 1) else 0.0
        out.append((pix, act, instruction, is_end))
    return out


def _flatten_episodes_single_step(
    episodes: list[dict[str, Any]],
    images_root: Path,
    resize: tuple[int, int] | None,
    only_success: bool,
    require_action_success: bool,
) -> list[list[tuple[np.ndarray, np.ndarray, str, float]]]:
    """Parse single-step JSON: each episode may mirror multi-step with trajectory list."""
    per_ep: list[list[tuple[np.ndarray, np.ndarray, str, float]]] = []
    for ep in episodes:
        steps = _episode_to_steps(ep, images_root, resize, only_success, require_action_success)
        if steps:
            per_ep.append(steps)
    return per_ep


def _split_episodes_train_test(
    episodes: list[dict[str, Any]],
    *,
    test_ratio: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically assign episodes to train/test by fixed-seed shuffle."""
    n = len(episodes)
    if n == 0:
        return [], []
    ratio = float(test_ratio)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"test_ratio must be in [0, 1], got {test_ratio!r}")
    n_test = int(round(n * ratio))
    n_test = max(0, min(n, n_test))
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    test_idx = set(perm[:n_test].tolist())
    train_eps = [episodes[i] for i in range(n) if i not in test_idx]
    test_eps = [episodes[i] for i in range(n) if i in test_idx]
    return train_eps, test_eps


def _convert_episode_list_to_hdf5(
    episodes: list[dict[str, Any]],
    images_root: Path,
    out_path: Path,
    *,
    resize: tuple[int, int] | None,
    only_success: bool,
    require_action_success: bool,
    show_progress: bool,
    num_workers: int | None,
    progress_desc: str,
) -> Path:
    """Write pre-selected episodes to one HDF5 file (see ``convert_eb_nav_json_to_hdf5``)."""
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ep_lens_list: list[int] = []
    total = 0
    img_shape: tuple[int, ...] | None = None
    pix_dset: h5py.Dataset | None = None
    act_dset: h5py.Dataset | None = None
    instr_dset: h5py.Dataset | None = None
    is_end_dset: h5py.Dataset | None = None
    str_dtype = h5py.string_dtype(encoding="utf-8")

    nw = num_workers if num_workers is not None else _default_episode_workers()
    nw = max(1, int(nw))

    def _load_one(ep: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, str, float]] | None:
        return _episode_to_steps(
            ep, images_root, resize, only_success, require_action_success
        )

    def _consume_steps(steps: list[tuple[np.ndarray, np.ndarray, str, float]] | None) -> None:
        nonlocal total, img_shape, pix_dset, act_dset, instr_dset, is_end_dset
        if not steps:
            return
        n = len(steps)
        if img_shape is None:
            img_shape = tuple(steps[0][0].shape)
            chunk_rows = min(32, max(1, n))
            pix_dset = f.create_dataset(
                "pixels",
                shape=(0, *img_shape),
                maxshape=(None, *img_shape),
                dtype=np.uint8,
                chunks=(chunk_rows, *img_shape),
                compression="gzip",
                compression_opts=4,
            )
            act_dset = f.create_dataset(
                "action",
                shape=(0, ACTION_DIM),
                maxshape=(None, ACTION_DIM),
                dtype=np.float32,
                chunks=(chunk_rows, ACTION_DIM),
                compression="gzip",
                compression_opts=4,
            )
            instr_dset = f.create_dataset(
                "instruction",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dtype,
                chunks=(chunk_rows,),
                compression="gzip",
                compression_opts=4,
            )
            is_end_dset = f.create_dataset(
                "is_end",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(chunk_rows,),
                compression="gzip",
                compression_opts=4,
            )
        assert pix_dset is not None and act_dset is not None
        assert instr_dset is not None and is_end_dset is not None

        for pix, _, _, _ in steps:
            if pix.shape != img_shape:
                raise ValueError(
                    f"Inconsistent image shape {pix.shape} vs {img_shape}; "
                    "use --resize W H or preprocess to a fixed size."
                )

        old = total
        new = total + n
        pix_dset.resize((new, *img_shape))
        act_dset.resize((new, ACTION_DIM))
        instr_dset.resize((new,))
        is_end_dset.resize((new,))
        pix_block = np.stack([s[0] for s in steps], axis=0)
        act_block = np.stack([s[1] for s in steps], axis=0)
        is_end_block = np.asarray([s[3] for s in steps], dtype=np.float32)
        pix_dset[old:new] = pix_block
        act_dset[old:new] = act_block
        instr_dset[old:new] = [s[2] for s in steps]
        is_end_dset[old:new] = is_end_block
        ep_lens_list.append(n)
        total = new

    with h5py.File(out_path, "w") as f:
        if nw <= 1:
            ep_bar = tqdm(
                episodes,
                desc=progress_desc,
                unit="ep",
                disable=not show_progress,
            )
            for ep in ep_bar:
                _consume_steps(_load_one(ep))
                if show_progress:
                    ep_bar.set_postfix_str(
                        f"{len(ep_lens_list)} kept, {total} steps", refresh=False
                    )
        else:
            with ThreadPoolExecutor(max_workers=nw) as executor:
                step_iter = executor.map(_load_one, episodes)
                if show_progress:
                    step_iter = tqdm(
                        step_iter,
                        total=len(episodes),
                        desc=progress_desc,
                        unit="ep",
                    )
                for steps in step_iter:
                    _consume_steps(steps)
                    if show_progress:
                        step_iter.set_postfix_str(
                            f"{len(ep_lens_list)} kept, {total} steps", refresh=False
                        )

        if total == 0:
            img_dir = images_root / "images"
            hint = ""
            if not img_dir.is_dir():
                hint = (
                    f" Expected image directory missing: {img_dir}. "
                    "Download the full HF repo (with images.zip), then unzip, e.g.: "
                    f"mkdir -p {images_root}/images && unzip {images_root}/images.zip -d {images_root}/images"
                )
            raise ValueError(
                "No valid episodes after conversion (check images_root, images on disk, and action format)."
                + hint
            )

        ep_lens = np.array(ep_lens_list, dtype=np.int64)
        ep_offset = np.cumsum(np.concatenate([[0], ep_lens[:-1]])).astype(np.int64)
        f.create_dataset("ep_len", data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offset)

    LOGGER.info(
        "Wrote %s (%d episodes, %d timesteps) to %s",
        out_path.name,
        len(ep_lens_list),
        total,
        out_path,
    )
    return out_path


def convert_eb_nav_json_to_hdf5(
    json_path: str | Path,
    images_root: str | Path,
    output_h5_path: str | Path,
    *,
    variant: str = "multi_step",
    only_success: bool = False,
    require_action_success: bool = False,
    resize: tuple[int, int] | None = None,
    max_episodes: int | None = None,
    show_progress: bool = True,
    num_workers: int | None = None,
    split_test_ratio: float | None = None,
    split_seed: int = 42,
    output_test_h5_path: str | Path | None = None,
) -> Path:
    """
    Read EB-Nav JSON and write an HDF5 file compatible with ``stable_worldmodel.HDF5Dataset``.

    Data is written episode-by-episode into extendable HDF5 datasets to limit RAM use.

    Args:
        json_path: Path to ``eb-nav_dataset_multi_step.json`` or ``eb-nav_dataset_single_step.json``.
        images_root: Root directory for resolving relative image paths (unzipped ``images/``).
        output_h5_path: Destination ``.h5`` file path (training split when ``split_test_ratio`` is set).
        variant: ``\"multi_step\"`` or ``\"single_step\"`` (same parser if structure matches).
        only_success: If True, drop episodes with ``success < 0.5``.
        require_action_success: If True, skip steps with ``action_success < 0.5``.
        resize: Optional ``(width, height)`` for PIL resize; default keeps native resolution.
        max_episodes: If set, only process the first N JSON episodes (debug / subset).
        show_progress: If True, show a tqdm bar over JSON episodes during conversion.
        num_workers: Thread count for loading episodes in parallel (1 = no thread pool).
            HDF5 is still written from the main thread only. Default: min(8, CPU count).
        split_test_ratio: If set (e.g. ``0.1``), randomly assign that fraction of episodes to a
            test HDF5 using ``split_seed`` for reproducibility; train is written to ``output_h5_path``.
        split_seed: RNG seed for the train/test episode split (default ``42``).
        output_test_h5_path: Path for the test split when ``split_test_ratio`` is set; if omitted,
            derived from ``output_h5_path`` by replacing ``_train`` with ``_test`` in the stem when
            possible, else ``<stem>_test.h5``.

    The output file includes ``instruction`` and ``is_end`` aligned with ``pixels`` / ``action``.
    """
    json_path = Path(json_path).resolve()
    images_root = Path(images_root).resolve()
    out_path = Path(output_h5_path).resolve()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    episodes = list(_iter_episode_dicts(data))
    if not episodes:
        raise ValueError(f"No episodes found in {json_path}")

    if max_episodes is not None:
        episodes = episodes[: max(0, int(max_episodes))]

    if variant not in ("multi_step", "single_step"):
        warnings.warn(f"Unknown variant {variant!r}, using multi_step parsing.", stacklevel=2)

    common_kw = dict(
        resize=resize,
        only_success=only_success,
        require_action_success=require_action_success,
        show_progress=show_progress,
        num_workers=num_workers,
    )

    if split_test_ratio is None:
        return _convert_episode_list_to_hdf5(
            episodes,
            images_root,
            out_path,
            progress_desc="EB-Nav → HDF5",
            **common_kw,
        )

    train_eps, test_eps = _split_episodes_train_test(
        episodes,
        test_ratio=split_test_ratio,
        split_seed=split_seed,
    )
    if not train_eps:
        raise ValueError(
            "Train split is empty after train/test split; reduce --split-test-ratio or check episodes."
        )
    test_out: Path
    if output_test_h5_path is not None:
        test_out = Path(output_test_h5_path).resolve()
    else:
        stem = out_path.stem
        if "_train" in stem:
            test_stem = stem.replace("_train", "_test", 1)
        else:
            test_stem = f"{stem}_test"
        test_out = (out_path.parent / f"{test_stem}.h5").resolve()

    LOGGER.info(
        "Train/test split: seed=%s ratio=%s → train %d ep, test %d ep → %s / %s",
        split_seed,
        split_test_ratio,
        len(train_eps),
        len(test_eps),
        out_path,
        test_out,
    )

    _convert_episode_list_to_hdf5(
        train_eps,
        images_root,
        out_path,
        progress_desc="EB-Nav train → HDF5",
        **common_kw,
    )
    if test_eps:
        _convert_episode_list_to_hdf5(
            test_eps,
            images_root,
            test_out,
            progress_desc="EB-Nav test → HDF5",
            **common_kw,
        )
    else:
        LOGGER.warning(
            "Train/test split produced 0 test episodes (JSON episodes=%d, ratio=%s); "
            "skip writing test HDF5.",
            len(episodes),
            split_test_ratio,
        )
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EB-Nav trajectory JSON to LeWM / stable_worldmodel HDF5."
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to eb-nav_dataset_multi_step.json or single_step.json (required unless --download).",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Directory used to resolve image paths (required unless --download).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .h5 path (required unless --download).",
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
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Only convert the first N episodes from the JSON (smoke test / subset).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (e.g. for logs or non-TTY).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel threads for loading episodes (default: min(8, CPU count); "
            "HDF5 writes stay single-threaded). Use 1 to disable parallelism."
        ),
    )
    p.add_argument(
        "--split-test-ratio",
        type=float,
        default=None,
        metavar="P",
        help=(
            "If set (e.g. 0.1), randomly assign this fraction of JSON episodes to a test "
            "HDF5 with reproducible split (--split-seed); train goes to --output."
        ),
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for train/test episode split (default: 42).",
    )
    p.add_argument(
        "--output-test",
        type=Path,
        default=None,
        help=(
            "Test split .h5 when --split-test-ratio is set; default: name derived from "
            "--output (e.g. *_train.h5 → *_test.h5)."
        ),
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

    missing = [
        n
        for n, v in (
            ("--json", args.json),
            ("--images-root", args.images_root),
            ("--output", args.output),
        )
        if v is None
    ]
    if missing:
        raise SystemExit(
            "error: conversion requires: "
            + ", ".join(missing)
            + " (or use --download DIR to only fetch the HF dataset)."
        )

    out = args.output
    if out.suffix.lower() != ".h5" and args.cache_dir is not None:
        datasets_dir = Path(args.cache_dir).resolve() / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        out = datasets_dir / f"{out.name}.h5"

    resize = tuple(args.resize) if args.resize is not None else None
    out_test = Path(args.output_test).resolve() if args.output_test is not None else None
    convert_eb_nav_json_to_hdf5(
        json_path=args.json,
        images_root=args.images_root,
        output_h5_path=out,
        variant=args.variant,
        only_success=args.only_success,
        require_action_success=args.require_action_success,
        resize=resize,
        max_episodes=args.max_episodes,
        show_progress=not args.no_progress,
        num_workers=args.workers,
        split_test_ratio=args.split_test_ratio,
        split_seed=args.split_seed,
        output_test_h5_path=out_test,
    )


if __name__ == "__main__":
    main()
