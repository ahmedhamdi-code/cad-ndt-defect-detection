#!/usr/bin/env python3
"""Train or resume one seeded AIRBUS variational autoencoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vae_cad.config import (
    canonical_json_sha256,
    make_config,
    protocol_payload,
    source_tree_sha256,
)
from vae_cad.data import (
    load_cache_manifest,
    make_series_split,
    open_cache,
    sha256_file,
    validate_dataset,
    write_split_manifest,
)
from vae_cad.preprocessing import validate_cache
from vae_cad.training import train_seed
from vae_cad.validation import (
    DEFAULT_METHOD_COLUMNS,
    ValidationSpec,
    validate_run_directory,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_training(
    *,
    seed: int,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    device: str = "cpu",
    resume: bool = True,
    verbose: int = 2,
) -> Path:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    config = make_config(
        data_root=data_root,
        cache_root=cache_root,
        output_root=output_root,
        seeds=(seed,),
        device=device,
    )
    run_directory = config.output_root / f"seed_{seed}"
    marker = run_directory / "run.json"
    dataset = validate_dataset(config, check_finite=False)
    validate_cache(config, dataset=dataset, scan_finite=False)
    cache_manifest = load_cache_manifest(config)
    train_cache = open_cache("train", config)
    split = make_series_split(seed, dataset.train.series_ids, config)
    cache_manifest_sha256 = str(cache_manifest["manifest_payload_sha256"])
    if marker.is_file():
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid completion marker: {marker}") from exc
        expected_protocol = protocol_payload(config, cache_manifest_sha256)
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"complete", "validated"}
            or int(record.get("seed", -1)) != seed
            or record.get("protocol") != expected_protocol
            or record.get("protocol_sha256")
            != canonical_json_sha256(expected_protocol)
        ):
            raise RuntimeError(
                "The existing completed run uses another protocol; use the "
                "orchestrator's explicit --overwrite option to retrain it"
            )
        validate_run_directory(
            run_directory,
            seed=seed,
            method_columns=DEFAULT_METHOD_COLUMNS,
            spec=ValidationSpec(),
        )
        print(f"Seed {seed} is already complete and valid: {run_directory}", flush=True)
        return run_directory

    run_directory.mkdir(parents=True, exist_ok=True)
    split_path = run_directory / "split_manifest.csv"
    write_split_manifest(split_path, split, dataset.train.series_ids)

    request = {
        "schema_version": 1,
        "seed": seed,
        "protocol_version": config.protocol_version,
        "source_tree_sha256": source_tree_sha256(),
        "vae": asdict(config.vae),
        "split": asdict(config.split),
        "cache_manifest_payload_sha256": cache_manifest_sha256,
        "split_manifest_sha256": sha256_file(split_path),
    }
    request["request_sha256"] = canonical_json_sha256(request)
    request_path = run_directory / "training_request.json"
    if request_path.is_file():
        stored = json.loads(request_path.read_text(encoding="utf-8"))
        if stored != request:
            raise RuntimeError(
                "Existing training artifacts were created with another data or model contract"
            )
    else:
        preexisting = (run_directory / "model.keras").exists() or (
            run_directory / "training_metadata.json"
        ).exists()
        if preexisting:
            raise RuntimeError("Untracked training artifacts require manual inspection")
        _atomic_json(request_path, request)

    print(f"Training seed {seed} on {device}", flush=True)
    result = train_seed(
        train_cache,
        split,
        run_directory,
        config,
        resume=resume,
        verbose=verbose,
    )
    print(
        f"Training {result.status}: epochs={result.epochs_completed}, "
        f"best_epoch={result.best_epoch}",
        flush=True,
    )
    return run_directory


def main() -> None:
    args = _arguments()
    run_training(
        seed=args.seed,
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        device=args.device,
        resume=not args.no_resume,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
