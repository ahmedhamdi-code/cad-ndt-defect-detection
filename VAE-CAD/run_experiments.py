#!/usr/bin/env python3
"""Run preprocessing, three seeded experiments, and result aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vae_cad.config import DEFAULT_SEEDS, make_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--skip-preparation", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recompute only the requested generated seed directories.",
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def _append_path(arguments: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        arguments.extend((flag, str(value.expanduser().resolve())))


def _run(arguments: list[str]) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    args = _arguments()
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Use at least two distinct non-negative seeds")
    config = make_config(
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        seeds=seeds,
        device=args.device,
    )

    common_paths: list[str] = []
    _append_path(common_paths, "--data-root", args.data_root)
    _append_path(common_paths, "--cache-root", args.cache_root)
    if not args.skip_preparation:
        command = [sys.executable, str(REPOSITORY_ROOT / "prepare_data.py"), *common_paths]
        if config.data.cache_manifest_path.is_file():
            command.append("--fast")
        _run(command)

    experiment_paths = list(common_paths)
    _append_path(experiment_paths, "--output-root", args.output_root)
    config.output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_directory = (config.output_root / f"seed_{seed}").resolve()
        if run_directory.parent != config.output_root.resolve():
            raise RuntimeError("Resolved seed directory escaped the output root")
        if args.overwrite and run_directory.exists():
            print(f"Removing generated run directory: {run_directory}", flush=True)
            shutil.rmtree(run_directory)
        _run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "train.py"),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--verbose",
                str(args.verbose),
                *experiment_paths,
            ]
        )
        _run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "inference.py"),
                "--seed",
                str(seed),
                "--device",
                args.device,
                *experiment_paths,
            ]
        )

    aggregate_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "aggregate_results.py"),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--device",
        args.device,
    ]
    _append_path(aggregate_command, "--data-root", args.data_root)
    _append_path(aggregate_command, "--cache-root", args.cache_root)
    _append_path(aggregate_command, "--output-root", args.output_root)
    _run(aggregate_command)
    print("All requested runs are complete and validated", flush=True)


if __name__ == "__main__":
    main()
