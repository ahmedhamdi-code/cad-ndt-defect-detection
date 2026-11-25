#!/usr/bin/env python3
"""Validate the AIRBUS source files and build the signed real-STFT caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vae_cad.config import make_config
from vae_cad.data import validate_dataset
from vae_cad.preprocessing import (
    build_stft_caches,
    check_dataset_parity,
    validate_cache,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the repeated exhaustive finite-value scan of raw HDF5 arrays.",
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Recompute source SHA-256 values when validating an existing cache.",
    )
    parser.add_argument(
        "--full-cache-scan",
        action="store_true",
        help="Scan every generated cache value again after preparation.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard only generated cache files and rebuild them from raw data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = make_config(data_root=args.data_root, cache_root=args.cache_root)
    print(f"Validating source data in {config.data.data_root}", flush=True)
    dataset = validate_dataset(config, check_finite=not args.fast)
    parity = check_dataset_parity(dataset, train_indices=(0,), test_indices=(0,))
    if not all(report.passed for report in parity):
        raise RuntimeError("STFT parity check failed")
    print("STFT parity check passed", flush=True)

    def progress(split: str, completed: int, total: int) -> None:
        if completed == total or completed == 1 or completed % 25 == 0:
            print(f"{split}: {completed}/{total} series", flush=True)

    result = build_stft_caches(
        config,
        dataset=dataset,
        resume=True,
        restart=args.restart,
        progress=progress,
    )
    report = validate_cache(
        config,
        dataset=dataset,
        verify_source_hashes=args.verify_source_hashes,
        scan_finite=args.full_cache_scan,
    )
    action = "reused" if result.reused_complete_cache else "built"
    print(
        f"Cache {action}: train={report.train_shape}, test={report.test_shape}, "
        f"dtype={report.dtype}",
        flush=True,
    )
    print(f"Manifest: {result.manifest_path}", flush=True)


if __name__ == "__main__":
    main()

