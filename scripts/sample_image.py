#!/usr/bin/env python3
"""Sample random Z-planes from microscopy volumes and save as OME-TIFF per channel.

For each scene (all scenes when ``--scene-index`` is omitted), randomly selects
``--n-samples`` Z-planes and writes one OME-TIFF per channel, e.g.::

    Animal 1-scene-0.z-0010.Dpn.tif

No preprocessing or segmentation is performed.

Intended for batch use via scripts/batch_process.sbatch::

    python scripts/sample_image.py -i "$input_file" -o "$output_dir" --n-samples 5
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow

import vistiq
from vistiq.io import ImageLoader, ImageLoaderConfig, ImageWriter, ImageWriterConfig, unstack_image
from vistiq.utils import check_device, get_scenes

logger = logging.getLogger(__name__)

DEFAULT_RENAME_CHANNEL = {"Red": "Dpn", "Green": "Scrib", "Blue": "EdU"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random Z-planes and save each channel as OME-TIFF.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input image path (.lif, .tif, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for sampled plane TIFFs",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=None,
        help="Scene index to load (default: process every scene in the file)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of Z-planes to sample per scene",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Z-plane selection (default: nondeterministic)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--channels-from-filename",
        action="store_true",
        help="Parse channel names from the input basename: split on '-', take the "
        "last N segments (N = number of channels). On failure, apply "
        "DEFAULT_RENAME_CHANNEL to loader channel names.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        metavar="NAME_OR_INDEX",
        help="Comma-separated channel names and/or 0-based indices to save "
        "(default: all channels), e.g. Dpn,1,EdU",
    )
    parser.add_argument(
        "--exclude-top",
        type=int,
        default=0,
        metavar="N",
        help="Exclude the top N Z-planes (highest indices) from sampling (default: 0)",
    )
    parser.add_argument(
        "--exclude-bottom",
        type=int,
        default=0,
        metavar="N",
        help="Exclude the bottom N Z-planes (lowest indices) from sampling (default: 0)",
    )
    return parser.parse_args(argv)


def parse_channel_list(channel_arg: str | None) -> list[str] | None:
    """Split a comma-separated ``--channel`` value into non-empty tokens."""
    if channel_arg is None:
        return None
    tokens = [token.strip() for token in channel_arg.split(",") if token.strip()]
    return tokens or None


def resolve_channel_indices(
    selections: list[str],
    channel_names: list[str],
) -> list[int]:
    """Map ``--channel`` tokens to 0-based indices (names first, then integers)."""
    if not channel_names:
        raise ValueError("Cannot resolve --channel: image has no channel_names in metadata")

    name_to_index = {name: index for index, name in enumerate(channel_names)}
    indices: list[int] = []
    for token in selections:
        if token in name_to_index:
            indices.append(name_to_index[token])
            continue
        if token.isdigit():
            index = int(token)
            if index < 0 or index >= len(channel_names):
                raise ValueError(
                    f"Channel index {index} out of range (0-{len(channel_names) - 1}); "
                    f"channels={channel_names}"
                )
            indices.append(index)
            continue
        raise ValueError(
            f"Unknown channel {token!r}; available names={channel_names}, "
            f"indices=0-{len(channel_names) - 1}"
        )

    seen: set[int] = set()
    unique_indices: list[int] = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
    return unique_indices


def channel_names_from_filename(path: Path, n_channels: int) -> list[str] | None:
    """Return the last *n_channels* dash-separated segments of the file basename."""
    if n_channels < 1:
        return None
    stem = path.name.rsplit(".", 1)[0]
    parts = [part.strip() for part in stem.split("-") if part.strip()]
    if len(parts) < n_channels:
        return None
    return parts[-n_channels:]


def resolve_channel_names(
    metadata: dict,
    input_path: Path,
    *,
    channels_from_filename: bool,
) -> None:
    """Set ``metadata['channel_names']`` from filename or default rename map."""
    channel_names = metadata.get("channel_names")
    if not channel_names or not channels_from_filename:
        return

    parsed = channel_names_from_filename(input_path, len(channel_names))
    if parsed is not None:
        metadata["channel_names"] = parsed
        logger.info("Channel names from filename: %s", parsed)
        return

    metadata["channel_names"] = [
        DEFAULT_RENAME_CHANNEL.get(name, name) for name in channel_names
    ]
    logger.warning(
        "Could not parse %d channel name(s) from %r; using DEFAULT_RENAME_CHANNEL",
        len(channel_names),
        input_path.name,
    )


def input_stem(input_path: Path) -> str:
    """Basename of the input file without extension."""
    return input_path.name.rsplit(".", 1)[0]


def scene_indices_to_process(input_path: Path, scene_index: int | None) -> list[int]:
    """Return scene indices to load from *input_path*."""
    if scene_index is not None:
        return [scene_index]
    scenes = get_scenes(input_path)
    if scenes:
        return list(range(len(scenes)))
    return [0]


def eligible_z_indices(
    n_planes: int,
    *,
    exclude_top: int = 0,
    exclude_bottom: int = 0,
) -> np.ndarray:
    """Return Z indices available for sampling after excluding stack ends."""
    if n_planes < 1:
        raise ValueError("Stack has no Z-planes to sample")
    if exclude_top < 0 or exclude_bottom < 0:
        raise ValueError("--exclude-top and --exclude-bottom must be >= 0")

    start = exclude_bottom
    end = n_planes - exclude_top
    if start >= end:
        raise ValueError(
            f"Cannot sample: {n_planes} Z-plane(s) with exclude_bottom={exclude_bottom} "
            f"and exclude_top={exclude_top} leaves no eligible planes"
        )
    return np.arange(start, end, dtype=np.int64)


PhysicalPixelSizes = namedtuple("PhysicalPixelSizes", ["Z", "Y", "X"])


def physical_pixel_sizes_from_scale(scale: Any) -> PhysicalPixelSizes | None:
    """Build OME-TIFF ``physical_pixel_sizes`` from a bioio :class:`Scale`."""
    if scale is None:
        return None
    values = scale._asdict() if hasattr(scale, "_asdict") else {}
    y_size = values.get("Y")
    x_size = values.get("X")
    if y_size is None and x_size is None:
        return None
    return PhysicalPixelSizes(
        Z=abs(values["Z"]) if values.get("Z") is not None else 1.0,
        Y=abs(y_size) if y_size is not None else 1.0,
        X=abs(x_size) if x_size is not None else 1.0,
    )


def prepare_channel_write_metadata(
    channel_metadata: dict[str, Any],
    *,
    stack_metadata: dict[str, Any],
    z_index: int,
) -> dict[str, Any]:
    """Merge stack pixel-size metadata into per-channel write metadata."""
    meta = copy.deepcopy(channel_metadata)
    for key in ("physical_pixel_sizes", "scale", "pixel_unit"):
        if meta.get(key) is None and stack_metadata.get(key) is not None:
            meta[key] = stack_metadata[key]

    if meta.get("physical_pixel_sizes") is None:
        pps = physical_pixel_sizes_from_scale(
            meta.get("scale") or stack_metadata.get("scale")
        )
        if pps is not None:
            meta["physical_pixel_sizes"] = pps

    meta["z_index"] = z_index
    return meta


def sample_z_indices(
    n_planes: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    exclude_top: int = 0,
    exclude_bottom: int = 0,
) -> np.ndarray:
    """Sample up to *n_samples* distinct eligible Z indices."""
    if n_samples < 1:
        raise ValueError("--n-samples must be at least 1")

    eligible = eligible_z_indices(
        n_planes,
        exclude_top=exclude_top,
        exclude_bottom=exclude_bottom,
    )
    n_draw = min(n_samples, len(eligible))
    if n_draw < n_samples:
        logger.warning(
            "Requested %d Z-plane(s) but only %d eligible; sampling all eligible",
            n_samples,
            len(eligible),
        )
    return np.sort(rng.choice(eligible, size=n_draw, replace=False))


def save_plane_channels(
    plane: np.ndarray,
    plane_metadata: dict,
    stack_metadata: dict,
    output_dir: Path,
    *,
    stem: str,
    scene_index: int,
    z_index: int,
    writer_config: ImageWriterConfig,
    channel_indices: list[int] | None = None,
) -> list[Path]:
    """Write one Z-plane as separate OME-TIFF files per channel."""
    if "C" in plane_metadata.get("axes", []):
        channel_images, channel_metadatas = unstack_image(plane, plane_metadata, "C")
    else:
        channel_images = (plane,)
        channel_metadatas = (plane_metadata,)

    written: list[Path] = []
    base = output_dir / f"{stem}-scene-{scene_index}.z-{z_index:04d}.tif"
    for ch_index, (ch_image, ch_metadata) in enumerate(
        zip(channel_images, channel_metadatas, strict=True)
    ):
        if channel_indices is not None and ch_index not in channel_indices:
            continue
        write_metadata = prepare_channel_write_metadata(
            ch_metadata,
            stack_metadata=stack_metadata,
            z_index=z_index,
        )
        channel_names = write_metadata.get("channel_names") or ["Channel"]
        pps = write_metadata.get("physical_pixel_sizes")
        logger.debug(
            "Writing %s channel %s z=%04d physical_pixel_sizes=%s scale=%s",
            base.name,
            channel_names[0],
            z_index,
            pps,
            write_metadata.get("scale"),
        )
        ImageWriter(writer_config).run(ch_image, base, metadata=write_metadata)
        written.append(base.with_suffix(f".{channel_names[0]}.tif"))
    return written


@flow(name="vistiq.sample_image")
def sample_image(
    input_path: Path,
    output_dir: Path,
    *,
    n_samples: int,
    scene_index: int | None = None,
    seed: int | None = None,
    channels_from_filename: bool = False,
    channel: str | None = None,
    exclude_top: int = 0,
    exclude_bottom: int = 0,
) -> list[Path]:
    """Load scene(s), sample Z-planes, and save per-channel OME-TIFFs."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if n_samples < 1:
        raise ValueError("--n-samples must be at least 1")

    rng = np.random.default_rng(seed)
    stem = input_stem(input_path)
    writer_config = ImageWriterConfig(overwrite=True)
    written_paths: list[Path] = []
    channel_selections = parse_channel_list(channel)

    for scene_idx in scene_indices_to_process(input_path, scene_index):
        logger.info("Loading %s scene_index=%s", input_path, scene_idx)
        loader_config = ImageLoaderConfig(
            squeeze=True,
            rename_channel=None if channels_from_filename else DEFAULT_RENAME_CHANNEL,
            scene_index=scene_idx,
            split_channels=False,
        )
        stack, metadata = ImageLoader(loader_config).run(input_path)
        resolve_channel_names(
            metadata,
            input_path,
            channels_from_filename=channels_from_filename,
        )

        channel_names = metadata.get("channel_names") or []
        channel_indices = (
            resolve_channel_indices(channel_selections, channel_names)
            if channel_selections is not None
            else None
        )
        if channel_indices is not None:
            selected_names = [channel_names[i] for i in channel_indices]
            logger.info("Saving channels: %s (indices %s)", selected_names, channel_indices)

        axes = metadata.get("axes", [])
        if "Z" not in axes:
            raise ValueError(
                f"Scene {scene_idx} has no Z axis (axes={axes!r}); cannot sample planes"
            )

        planes, plane_metadatas = unstack_image(stack, metadata, "Z")
        eligible = eligible_z_indices(
            len(planes),
            exclude_top=exclude_top,
            exclude_bottom=exclude_bottom,
        )
        z_indices = sample_z_indices(
            len(planes),
            n_samples,
            rng,
            exclude_top=exclude_top,
            exclude_bottom=exclude_bottom,
        )
        logger.info(
            "Scene %s: sampling Z-planes %s (%d eligible of %d; "
            "exclude_bottom=%d exclude_top=%d)",
            scene_idx,
            z_indices.tolist(),
            len(eligible),
            len(planes),
            exclude_bottom,
            exclude_top,
        )

        for z_index in z_indices:
            paths = save_plane_channels(
                planes[z_index],
                plane_metadatas[z_index],
                metadata,
                output_dir,
                stem=stem,
                scene_index=scene_idx,
                z_index=int(z_index),
                writer_config=writer_config,
                channel_indices=channel_indices,
            )
            written_paths.extend(paths)
            logger.info(
                "Scene %s Z=%04d -> %s",
                scene_idx,
                z_index,
                ", ".join(p.name for p in paths),
            )

    return written_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(vistiq.__name__).setLevel(getattr(logging, args.log_level))
    logger.info("Available Torch accelerators: %s", check_device())

    try:
        written = sample_image(
            args.input,
            args.output,
            n_samples=args.n_samples,
            scene_index=args.scene_index,
            seed=args.seed,
            channels_from_filename=args.channels_from_filename,
            channel=args.channel,
            exclude_top=args.exclude_top,
            exclude_bottom=args.exclude_bottom,
        )
    except Exception:
        logger.exception("Image sampling failed")
        return 1

    logger.info("Wrote %d file(s) to %s", len(written), args.output)
    for path in written:
        logger.info("  %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
