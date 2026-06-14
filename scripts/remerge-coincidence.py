#!/usr/bin/env python3
"""Re-merge existing Coincidence_*.csv files into Features_*.csv tables.

Use this to fix Features CSVs after the coincidence merge dtype bug, without
re-running coincidence detection.

Examples:
  # Dry-run on one output folder
  python scripts/remerge-coincidence.py /path/to/jobrun/output --dry-run

  # Update Features_*.csv in place
  python scripts/remerge-coincidence.py /path/to/jobrun/output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vistiq.analysis.coincidence import remerge_coincidence_into_features

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-merge Coincidence_*.csv columns into Features_*.csv for one "
            "output folder."
        )
    )
    parser.add_argument(
        "work_dir",
        type=Path,
        help="Folder containing Features_*.csv and Coincidence_*.csv files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing Features_*.csv",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    work_dir = args.work_dir.expanduser().resolve()
    if not work_dir.is_dir():
        logger.error("Not a directory: %s", work_dir)
        return 1

    updated = remerge_coincidence_into_features(work_dir, dry_run=args.dry_run)
    if not updated and not args.dry_run:
        logger.warning("No Features_*.csv files were updated in %s", work_dir)
        return 1

    if args.dry_run:
        logger.info("Dry-run complete for %s", work_dir)
    else:
        logger.info("Updated %d feature file(s) in %s", len(updated), work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
