#!/usr/bin/env python3
"""Aggregate completed AIRBUS runs and generate the reproduction report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vae_cad.config import (
    DEFAULT_SEEDS,
    canonical_json_sha256,
    make_config,
    protocol_payload,
)
from vae_cad.data import cache_identity, load_cache_manifest, open_cache
from vae_cad.metrics import write_aggregate_metrics
from vae_cad.reporting import generate_reproduction_report
from vae_cad.validation import (
    DEFAULT_METHOD_COLUMNS,
    ValidationSpec,
    validate_completed_runs,
    validate_metrics_summary,
)


EXPECTED_METHODS = tuple(DEFAULT_METHOD_COLUMNS)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    return parser.parse_args()


def aggregate(
    *,
    output_root: Path | None = None,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Path:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Aggregation requires at least two distinct seeds")
    config = make_config(
        data_root=data_root,
        cache_root=cache_root,
        output_root=output_root,
        seeds=seeds,
        device=device,
    )
    cache_manifest = dict(load_cache_manifest(config))
    cache_manifest_sha256 = cache_manifest.get("manifest_payload_sha256")
    unhashed_manifest = dict(cache_manifest)
    unhashed_manifest.pop("manifest_payload_sha256", None)
    if (
        not isinstance(cache_manifest_sha256, str)
        or canonical_json_sha256(unhashed_manifest) != cache_manifest_sha256
    ):
        raise ValueError("Cache manifest payload hash is invalid")
    expected_protocol = protocol_payload(config, cache_manifest_sha256)
    expected_protocol_sha256 = canonical_json_sha256(expected_protocol)
    spec = ValidationSpec(
        seeds=seeds,
        n_test_series=config.data.test_series,
        n_test_normal=config.data.normal_test_series,
        n_test_anomaly=config.data.anomalous_test_series,
        patches_per_series=config.data.patches_per_series,
        n_train_series=config.split.train_series,
        n_validation_series=config.split.validation_series,
        n_calibration_series=config.split.calibration_series,
    )
    validation = validate_completed_runs(
        config.output_root,
        spec=spec,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    current_cache_identities = {
        "train": cache_identity(open_cache("train", config)),
        "test": cache_identity(open_cache("test", config)),
    }
    if validation["cache_identities_sha256"] != canonical_json_sha256(
        current_cache_identities
    ):
        raise ValueError("Completed runs do not match the current cache files")
    per_run_path, summary_path = write_aggregate_metrics(
        config.output_root,
        seeds=seeds,
        expected_methods=EXPECTED_METHODS,
    )
    validate_metrics_summary(
        per_run_path,
        summary_path,
        spec=spec,
        expected_methods=EXPECTED_METHODS,
    )
    report_path = generate_reproduction_report(
        config.output_root,
        per_run=per_run_path,
        summary=summary_path,
        seeds=seeds,
        configuration=config.as_dict(),
    )
    print(f"Per-run metrics: {per_run_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return report_path


def main() -> None:
    args = _arguments()
    aggregate(
        output_root=args.output_root,
        data_root=args.data_root,
        cache_root=args.cache_root,
        seeds=tuple(args.seeds),
        device=args.device,
    )


if __name__ == "__main__":
    main()
