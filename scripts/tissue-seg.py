#!/usr/bin/env python3
"""Lobe and brain tissue segmentation (notebook: notebooks/segment-lobe.ipynb).

Loads a multi-channel microscopy volume, preprocesses it, segments lobes with
tiled MicroSAM, derives a binary brain mask, saves label TIFFs, and writes a
region-measurements CSV. Napari visualization is omitted.

Intended for batch use via scripts/batch_process.sbatch::

    python scripts/tissue-seg.py -i "$input_file" -o "$output_dir"
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np

import vistiq
from vistiq.io import ImageLoader, ImageLoaderConfig, ImageWriter, ImageWriterConfig
from vistiq.preprocess import FuncProcessorConfig, PreprocessFlow, PreprocessFlowConfig, RescaleConfig
from vistiq.segment import (
    MicroSAMSegmenterConfig,
    RangeFilterConfig,
    RegionAnalyzer,
    RegionAnalyzerConfig,
    RegionFilterConfig,
    TiledSegmentationFlow,
    TiledSegmentationFlowConfig,
)
from vistiq.utils import ArrayIteratorConfig, check_device

logger = logging.getLogger(__name__)

DEFAULT_RENAME_CHANNEL = {"Red": "Dpn", "Green": "Scrib", "Blue": "EdU"}
DEFAULT_EMBEDDING_PATH = os.environ.get("VISTIQ_EMBEDDING_PATH", "./embeddings")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment brain lobes from a microscopy volume (MicroSAM + tiled flow).",
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
        help="Output directory for label TIFFs and measurements CSV",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Scene index for multi-scene containers (default: 0)",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path(DEFAULT_EMBEDDING_PATH),
        help="Directory with MicroSAM embeddings "
        f"(default: VISTIQ_EMBEDDING_PATH or {DEFAULT_EMBEDDING_PATH!r})",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=-1,
        help="Worker count for PreprocessFlow (default: -1, all cores)",
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=2,
        help="Worker count for TiledSegmentationFlow (default: 2)",
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
    return parser.parse_args(argv)


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
    if not channel_names:
        return

    if not channels_from_filename:
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


def build_preprocess_config() -> PreprocessFlowConfig:
    """Preprocess pipeline from notebooks/segment-lobe.ipynb."""
    return PreprocessFlowConfig(
        processors=[
            RescaleConfig(
                low=2,
                high=98,
                dtype=np.uint8,
                iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
            ),
            FuncProcessorConfig(
                func="skimage.filters.gaussian",
                kwargs={"sigma": 1.0},
                iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
            ),
            FuncProcessorConfig(
                func="skimage.exposure.adjust_gamma",
                kwargs={"gamma": 0.2},
                iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
            ),
            FuncProcessorConfig(
                func="skimage.exposure.adjust_sigmoid",
                iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
            ),
            RescaleConfig(
                dtype=np.uint8,
                iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
            ),
            FuncProcessorConfig(
                func="numpy.max",
                kwargs={"axis": ("C",)},
                strict_axis=False,
                dtype=np.uint16,
            ),
        ]
    )


def build_segmentation_config(embedding_path: Path) -> TiledSegmentationFlowConfig:
    """Tiled MicroSAM segmentation config from notebooks/segment-lobe.ipynb."""
    return TiledSegmentationFlowConfig(
        segmenter=MicroSAMSegmenterConfig(
            iterator_config=ArrayIteratorConfig(slice_def=()),
            embedding_path=str(embedding_path),
        ),
        region_filter=RegionFilterConfig(
            filters=[
                RangeFilterConfig(
                    attribute="cross_sectional_area-xy",
                    range=(2000, np.inf),
                ),
                RangeFilterConfig(
                    attribute="cross_sectional_area-xz",
                    range=(2000, np.inf),
                ),
                RangeFilterConfig(
                    attribute="cross_sectional_area-yz",
                    range=(2000, np.inf),
                ),
                RangeFilterConfig(
                    attribute="aspect_ratio",
                    range=(0.5, 1.0),
                ),
            ]
        ),
        tile_factor=(3, 3),
        resize_factor=(0.25, 0.25),
        iou_threshold=0.5,
        consensus_threshold=0.75,
    )


def output_stem(input_path: Path, scene_index: int) -> str:
    """Base filename stem for outputs: ``{basename}-scene-{scene_index}``."""
    basename = input_path.name.rsplit(".", 1)[0]
    return f"{basename}-scene-{scene_index}"


def run_tissue_segmentation(
    input_path: Path,
    output_dir: Path,
    *,
    scene_index: int = 0,
    embedding_path: Path = Path(DEFAULT_EMBEDDING_PATH),
    preprocess_workers: int = -1,
    segment_workers: int = 2,
    channels_from_filename: bool = False,
) -> dict[str, Path]:
    """Run load → preprocess → segment → save workflow."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    embedding_path = embedding_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not embedding_path.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_path}")

    stem = output_stem(input_path, scene_index)
    tif_base = output_dir / f"{stem}.tif"

    logger.info("Loading %s (scene_index=%s)", input_path, scene_index)
    loader_config = ImageLoaderConfig(
        squeeze=True,
        rename_channel=None if channels_from_filename else DEFAULT_RENAME_CHANNEL,
        scene_index=scene_index,
        split_channels=False,
    )
    img, metadata = ImageLoader(loader_config).run(input_path)
    resolve_channel_names(
        metadata,
        input_path,
        channels_from_filename=channels_from_filename,
    )

    logger.info("Preprocessing")
    c_img, c_metadata = PreprocessFlow(build_preprocess_config()).run(
        img,
        metadata=metadata,
        workers=preprocess_workers,
    )

    logger.info("Segmenting lobes (embedding_path=%s)", embedding_path)
    lobe_labels = TiledSegmentationFlow(
        build_segmentation_config(embedding_path)
    ).run(
        c_img,
        metadata=c_metadata,
        workers=segment_workers,
        verbose=0,
    )
    unique_labels = np.unique(lobe_labels)
    logger.info(
        "Segmentation complete: %d labels (incl. background), shape=%s",
        len(unique_labels),
        lobe_labels.shape,
    )

    brain_label = (lobe_labels > 0).astype(np.uint16)

    lobe_metadata = copy.deepcopy(c_metadata)
    lobe_metadata["channel_names"] = ["Lobe"]
    brain_metadata = copy.deepcopy(c_metadata)
    brain_metadata["channel_names"] = ["Brain"]

    writer_config = ImageWriterConfig(overwrite=True)
    logger.info("Saving lobe labels to %s", tif_base)
    ImageWriter(writer_config).run(lobe_labels, tif_base, metadata=lobe_metadata)
    logger.info("Saving brain mask to %s", tif_base)
    ImageWriter(writer_config).run(brain_label, tif_base, metadata=brain_metadata)

    logger.info("Analyzing lobe regions")
    analyzer_config = RegionAnalyzerConfig(
        properties=[
            "slice_annotations",
            "volume",
            "centroid",
            "cross_sectional_area",
            "bbox",
            "aspect_ratio",
        ],
        iterator_config=ArrayIteratorConfig(slice_def=()),
        output_type="dataframe",
        map_axes=True,
    )
    lobe_measurements = RegionAnalyzer(analyzer_config).run(
        lobe_labels,
        metadata=lobe_metadata,
    )
    csv_path = output_dir / f"{stem}.Lobe.csv"
    lobe_measurements.sort_values(["volume"], ascending=False).to_csv(csv_path, index=False)
    logger.info("Wrote measurements to %s (%d rows)", csv_path, len(lobe_measurements))

    lobe_tif = tif_base.with_suffix(".Lobe.tif")
    brain_tif = tif_base.with_suffix(".Brain.tif")
    return {
        "lobe_labels": lobe_tif,
        "brain_mask": brain_tif,
        "measurements": csv_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(vistiq.__name__).setLevel(getattr(logging, args.log_level))
    logger.info("Available Torch accelerators: %s", check_device())

    try:
        outputs = run_tissue_segmentation(
            args.input,
            args.output,
            scene_index=args.scene_index,
            embedding_path=args.embedding_path,
            preprocess_workers=args.preprocess_workers,
            segment_workers=args.segment_workers,
            channels_from_filename=args.channels_from_filename,
        )
    except Exception:
        logger.exception("Tissue segmentation failed")
        return 1

    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
