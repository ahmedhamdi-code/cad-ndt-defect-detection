#!/usr/bin/env python3
"""Fit the fixed detectors and evaluate one trained AIRBUS VAE."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np
import pandas as pd

from vae_cad.cad import (
    CADState,
    aggregate_patch_scores,
    fit_mae_threshold,
    make_bin_edges,
    predict_mae_patch_or,
    predict_patch_or,
    predict_series_cad,
    residual_histograms,
    wasserstein_scores,
)
from vae_cad.config import (
    canonical_json_sha256,
    make_config,
    protocol_payload,
    source_tree_sha256,
)
from vae_cad.data import (
    cache_identity,
    iter_patch_batches,
    load_cache_manifest,
    make_series_split,
    open_cache,
    sha256_file,
    validate_dataset,
)
from vae_cad.metrics import classification_metrics
from vae_cad.model import load_vae
from vae_cad.preprocessing import validate_cache
from vae_cad.training import load_normalization, training_input_contract
from vae_cad.validation import (
    ValidationSpec,
    validate_cad_state,
    validate_metrics_against_predictions,
    validate_prediction_table,
    validate_run_directory,
    validate_split_manifest,
)


METHOD_COLUMNS = {
    "VAE+MAE": ("mae_score", "mae_prediction"),
    "VAE+CAD-W1-patch": ("cad_patch_score", "cad_patch_prediction"),
    "VAE+CAD-W1-series": ("cad_series_score", "cad_series_prediction"),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute detector artifacts while preserving the trained model.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_value(payload: object) -> object:
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _validate_exact_split(
    path: Path,
    split: object,
    series_ids: np.ndarray,
    *,
    seed: int,
    spec: ValidationSpec,
) -> None:
    frame = validate_split_manifest(path, seed=seed, spec=spec)
    for partition, positions in split.as_dict(copy=False).items():
        rows = frame.loc[frame["partition"].astype(str) == partition]
        if "series_position" not in rows:
            raise ValueError("Split manifest has no series_position column")
        actual_positions = rows["series_position"].to_numpy(dtype=np.int64)
        actual_ids = rows["series_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_positions, positions):
            raise ValueError(f"Saved {partition} positions differ from the seeded split")
        if not np.array_equal(actual_ids, series_ids[positions]):
            raise ValueError(f"Saved {partition} IDs differ from the seeded split")


def _validate_training_contract(
    run_directory: Path,
    *,
    seed: int,
    config: object,
    split: object,
    series_ids: np.ndarray,
    train_cache: np.ndarray,
    cache_manifest_sha256: str,
    spec: ValidationSpec,
) -> None:
    _validate_exact_split(
        run_directory / "split_manifest.csv",
        split,
        series_ids,
        seed=seed,
        spec=spec,
    )
    request_path = run_directory / "training_request.json"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training request: {request_path}") from exc
    if not isinstance(request, dict):
        raise ValueError("training_request.json must contain an object")
    stored_request_hash = request.get("request_sha256")
    unhashed_request = dict(request)
    unhashed_request.pop("request_sha256", None)
    if stored_request_hash != canonical_json_sha256(unhashed_request):
        raise ValueError("Training request hash is invalid")
    expected = {
        "schema_version": 1,
        "seed": seed,
        "protocol_version": config.protocol_version,
        "source_tree_sha256": source_tree_sha256(),
        "vae": _json_value(asdict(config.vae)),
        "split": _json_value(asdict(config.split)),
        "cache_manifest_payload_sha256": cache_manifest_sha256,
        "split_manifest_sha256": sha256_file(run_directory / "split_manifest.csv"),
    }
    for key, expected_value in expected.items():
        if request.get(key) != expected_value:
            raise ValueError(f"Training request field {key!r} differs from inference")

    metadata_path = run_directory / "training_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("training_metadata.json must contain an object")
    if metadata.get("schema_version") != 2:
        raise ValueError("Training metadata uses an unsupported schema")
    if metadata.get("status") != "complete" or int(metadata.get("seed", -1)) != seed:
        raise ValueError("Training metadata is not complete for the requested seed")
    if metadata.get("device") != config.device:
        raise ValueError("Training metadata contains another device policy")
    if metadata.get("vae_config") != expected["vae"]:
        raise ValueError("Training metadata contains another VAE configuration")
    if metadata.get("model_sha256") != sha256_file(run_directory / "model.keras"):
        raise ValueError("The trained model differs from training_metadata.json")
    expected_training_contract = training_input_contract(
        train_cache,
        split.train_indices,
        split.validation_indices,
    )
    if metadata.get("training_contract") != _json_value(expected_training_contract):
        raise ValueError("The trained model was produced from another cache or split")
    artifact_records = metadata.get("artifacts")
    if not isinstance(artifact_records, dict):
        raise ValueError("Training metadata has no artifact manifest")
    for name in (
        "model.keras",
        "encoder.keras",
        "decoder.keras",
        "model_summary.txt",
        "history.csv",
        "normalization.npz",
        "latent_diagnostics.json",
        "latent_diagnostics.npz",
    ):
        path = run_directory / name
        record = artifact_records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Training metadata omits artifact {name!r}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Training artifact size changed: {name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Training artifact hash changed: {name}")


def _package_versions() -> dict[str, str]:
    packages = ("tensorflow", "numpy", "scipy", "pandas", "scikit-learn", "h5py")
    result: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _residual_batches(
    model: object,
    cache: np.ndarray,
    series_indices: np.ndarray,
    *,
    mean: float,
    std: float,
    batch_size: int,
    stage: str,
):
    total_patches = int(series_indices.size) * 120
    completed = 0
    for batch_index, raw_batch in enumerate(
        iter_patch_batches(
            cache,
            series_indices,
            batch_size,
            shuffle=False,
            normalization=None,
        ),
        start=1,
    ):
        standardized = (
            (raw_batch - np.float32(mean)) / np.float32(std)
        ).astype(np.float32, copy=False)
        reconstruction_standardized = np.asarray(
            model.reconstruct_deterministic(standardized).numpy(), dtype=np.float32
        )
        reconstruction = (
            reconstruction_standardized * np.float32(std) + np.float32(mean)
        ).astype(np.float32, copy=False)
        residuals = np.abs(raw_batch - reconstruction).astype(np.float32, copy=False)
        if residuals.shape != raw_batch.shape or not bool(np.isfinite(residuals).all()):
            raise FloatingPointError(f"Invalid residuals during {stage}")
        completed += int(residuals.shape[0])
        if batch_index == 1 or completed == total_patches or batch_index % 200 == 0:
            print(f"{stage}: {completed}/{total_patches} patches", flush=True)
        yield residuals


def _training_statistics(
    model: object,
    train_cache: np.ndarray,
    train_indices: np.ndarray,
    *,
    mean: float,
    std: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_patches = int(train_indices.size) * 120
    mae = np.empty(n_patches, dtype=np.float64)
    automatic_bin_counts = np.empty(n_patches, dtype=np.int32)
    residual_min = np.inf
    residual_max = -np.inf
    cursor = 0
    for residuals in _residual_batches(
        model,
        train_cache,
        train_indices,
        mean=mean,
        std=std,
        batch_size=batch_size,
        stage="training statistics",
    ):
        count = int(residuals.shape[0])
        flattened = residuals.reshape(count, -1)
        mae[cursor : cursor + count] = flattened.mean(axis=1, dtype=np.float64)
        for offset, patch in enumerate(flattened):
            automatic_bin_counts[cursor + offset] = max(
                2, int(np.histogram_bin_edges(patch, bins="auto").size - 1)
            )
        residual_min = min(residual_min, float(flattened.min()))
        residual_max = max(residual_max, float(flattened.max()))
        cursor += count
    if cursor != n_patches:
        raise RuntimeError("Training residual scan ended early")
    return mae, automatic_bin_counts, float(residual_min), float(residual_max)


def _reference_histogram(
    model: object,
    train_cache: np.ndarray,
    train_indices: np.ndarray,
    bin_edges: np.ndarray,
    *,
    mean: float,
    std: float,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    histogram_sum = np.zeros(bin_edges.size - 1, dtype=np.float64)
    count = 0
    for residuals in _residual_batches(
        model,
        train_cache,
        train_indices,
        mean=mean,
        std=std,
        batch_size=batch_size,
        stage="reference histogram",
    ):
        histograms = residual_histograms(residuals, bin_edges)
        histogram_sum += histograms.sum(axis=0, dtype=np.float64)
        count += int(histograms.shape[0])
    if count == 0 or float(histogram_sum.sum()) <= 0.0:
        raise RuntimeError("No training histograms were accumulated")
    reference = histogram_sum / histogram_sum.sum()
    if not np.isclose(reference.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Reference histogram does not have unit mass")
    return reference, count


def _partition_scores(
    model: object,
    cache: np.ndarray,
    series_indices: np.ndarray,
    bin_edges: np.ndarray,
    reference: np.ndarray,
    *,
    mean: float,
    std: float,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_patches = int(series_indices.size) * 120
    mae = np.empty(n_patches, dtype=np.float64)
    w1 = np.empty(n_patches, dtype=np.float64)
    cursor = 0
    for residuals in _residual_batches(
        model,
        cache,
        series_indices,
        mean=mean,
        std=std,
        batch_size=batch_size,
        stage=stage,
    ):
        count = int(residuals.shape[0])
        flattened = residuals.reshape(count, -1)
        mae[cursor : cursor + count] = flattened.mean(axis=1, dtype=np.float64)
        histograms = residual_histograms(residuals, bin_edges)
        w1[cursor : cursor + count] = wasserstein_scores(
            histograms, reference, bin_edges
        )
        cursor += count
    if cursor != n_patches:
        raise RuntimeError(f"{stage} residual scan ended early")
    n_series = int(series_indices.size)
    return mae.reshape(n_series, 120), w1.reshape(n_series, 120)


def _cad_archive(
    path: Path,
    state: CADState,
    *,
    automatic_bin_counts: np.ndarray,
    automatic_bin_median: float,
) -> None:
    _atomic_savez(
        path,
        bin_edges=state.bin_edges,
        bin_centres=state.bin_centres,
        reference_histogram=state.reference_histogram,
        calibration_scores=state.calibration_scores,
        alpha=np.asarray(state.alpha, dtype=np.float64),
        calibration_level=np.asarray(state.calibration_level),
        aggregation=np.asarray(state.aggregation),
        n_training_histograms=np.asarray(
            state.n_training_histograms, dtype=np.int64
        ),
        automatic_bin_counts=np.asarray(automatic_bin_counts, dtype=np.int32),
        automatic_bin_median=np.asarray(automatic_bin_median, dtype=np.float64),
        p_value_formula=np.asarray(
            "(1 + count(calibration_score >= candidate_score)) / "
            "(n_calibration + 1)"
        ),
    )


def _metric_row(
    seed: int,
    method: str,
    rule: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    values = classification_metrics(y_true, prediction, score)
    values["confusion_matrix"] = json.dumps(values["confusion_matrix"])
    return {"seed": seed, "method": method, "rule": rule, **values}


def run_inference(
    *,
    seed: int,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    device: str = "cpu",
    overwrite: bool = False,
) -> Path:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    started_at = _utc_now()
    config = make_config(
        data_root=data_root,
        cache_root=cache_root,
        output_root=output_root,
        seeds=(seed,),
        device=device,
    )
    run_directory = config.output_root / f"seed_{seed}"
    run_marker = run_directory / "run.json"
    existing_run = None
    if run_marker.is_file():
        try:
            existing_run = json.loads(run_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid completion marker: {run_marker}") from exc
        if not isinstance(existing_run, dict):
            raise ValueError(f"Completion marker must contain an object: {run_marker}")

    required_training = (
        "model.keras",
        "encoder.keras",
        "decoder.keras",
        "model_summary.txt",
        "history.csv",
        "split_manifest.csv",
        "normalization.npz",
        "training_request.json",
        "training_metadata.json",
        "latent_diagnostics.json",
        "latent_diagnostics.npz",
    )
    missing = [name for name in required_training if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Training is incomplete; missing artifacts: {missing}")

    dataset = validate_dataset(config, check_finite=False)
    validate_cache(config, dataset=dataset, scan_finite=False)
    cache_manifest = load_cache_manifest(config)
    train_cache = open_cache("train", config)
    test_cache = open_cache("test", config)
    current_cache_identities = {
        "train": cache_identity(train_cache),
        "test": cache_identity(test_cache),
    }
    split = make_series_split(seed, dataset.train.series_ids, config)
    spec = ValidationSpec()
    cache_manifest_sha256 = str(cache_manifest["manifest_payload_sha256"])
    _validate_training_contract(
        run_directory,
        seed=seed,
        config=config,
        split=split,
        series_ids=dataset.train.series_ids,
        train_cache=train_cache,
        cache_manifest_sha256=cache_manifest_sha256,
        spec=spec,
    )
    persisted_protocol = protocol_payload(config, cache_manifest_sha256)
    protocol_sha256 = canonical_json_sha256(persisted_protocol)
    if existing_run is not None and not overwrite:
        if (
            existing_run.get("status") not in {"complete", "validated"}
            or int(existing_run.get("seed", -1)) != seed
            or existing_run.get("protocol") != persisted_protocol
            or existing_run.get("protocol_sha256") != protocol_sha256
            or existing_run.get("cache_identities") != current_cache_identities
        ):
            raise RuntimeError(
                "The existing completed run uses another protocol; use --overwrite "
                "only if recomputation is intended"
            )
        validate_run_directory(
            run_directory,
            seed=seed,
            method_columns=METHOD_COLUMNS,
            spec=spec,
        )
        print(f"Seed {seed} is already complete and valid: {run_directory}", flush=True)
        return run_directory
    normalization = load_normalization(run_directory / "normalization.npz")
    model = load_vae(run_directory / "model.keras", compile_model=False, device=device)

    train_mae, automatic_bin_counts, residual_min, residual_max = (
        _training_statistics(
            model,
            train_cache,
            split.train_indices,
            mean=normalization.mean,
            std=normalization.std,
            batch_size=config.vae.batch_size,
        )
    )
    automatic_bin_median = float(np.median(automatic_bin_counts))
    n_bins = max(2, int(np.rint(automatic_bin_median)))
    print(
        f"CAD bins: {n_bins} (median auto count {automatic_bin_median:.3f})",
        flush=True,
    )
    bin_edges = make_bin_edges(residual_min, residual_max, n_bins)
    reference, n_training_histograms = _reference_histogram(
        model,
        train_cache,
        split.train_indices,
        bin_edges,
        mean=normalization.mean,
        std=normalization.std,
        batch_size=config.vae.batch_size,
    )
    calibration_mae, calibration_w1 = _partition_scores(
        model,
        train_cache,
        split.calibration_indices,
        bin_edges,
        reference,
        mean=normalization.mean,
        std=normalization.std,
        batch_size=config.vae.batch_size,
        stage="calibration scoring",
    )
    test_indices = np.arange(config.data.test_series, dtype=np.int64)
    test_mae, test_w1 = _partition_scores(
        model,
        test_cache,
        test_indices,
        bin_edges,
        reference,
        mean=normalization.mean,
        std=normalization.std,
        batch_size=config.vae.batch_size,
        stage="test scoring",
    )

    threshold = fit_mae_threshold(train_mae, config.detector.mae_quantile)
    mae_score, mae_prediction = predict_mae_patch_or(test_mae, threshold)
    patch_state = CADState(
        bin_edges=bin_edges,
        reference_histogram=reference,
        calibration_scores=calibration_w1.reshape(-1),
        alpha=config.detector.alpha,
        calibration_level="patch",
        aggregation=config.detector.series_aggregation,
        n_training_histograms=n_training_histograms,
    )
    calibration_series_w1 = aggregate_patch_scores(
        calibration_w1, config.detector.series_aggregation, axis=1
    )
    series_state = CADState(
        bin_edges=bin_edges,
        reference_histogram=reference,
        calibration_scores=calibration_series_w1,
        alpha=config.detector.alpha,
        calibration_level="series",
        aggregation=config.detector.series_aggregation,
        n_training_histograms=n_training_histograms,
    )
    cad_patch_score, cad_patch_min_p, cad_patch_prediction = predict_patch_or(
        patch_state, test_w1
    )
    cad_series_score, cad_series_p, cad_series_prediction = predict_series_cad(
        series_state, test_w1
    )
    test_patch_p = patch_state.p_values(test_w1.reshape(-1)).reshape(test_w1.shape)

    _atomic_savez(
        run_directory / "baseline.npz",
        threshold=np.asarray(threshold, dtype=np.float64),
        quantile=np.asarray(config.detector.mae_quantile, dtype=np.float64),
        training_patch_scores=train_mae,
        decision=np.asarray("series anomaly if any patch MAE > threshold"),
    )
    _cad_archive(
        run_directory / "cad_state_patch.npz",
        patch_state,
        automatic_bin_counts=automatic_bin_counts,
        automatic_bin_median=automatic_bin_median,
    )
    _cad_archive(
        run_directory / "cad_state_series.npz",
        series_state,
        automatic_bin_counts=automatic_bin_counts,
        automatic_bin_median=automatic_bin_median,
    )
    _atomic_savez(
        run_directory / "patch_scores.npz",
        calibration_mae=calibration_mae,
        calibration_w1=calibration_w1,
        test_mae=test_mae,
        test_w1=test_w1,
        test_patch_p_values=test_patch_p,
    )

    predictions = pd.DataFrame(
        {
            "series_id": dataset.test_ids,
            "y_true": dataset.test_labels,
            "mae_score": mae_score,
            "mae_prediction": mae_prediction,
            "cad_patch_score": cad_patch_score,
            "cad_patch_min_p_value": cad_patch_min_p,
            "cad_patch_prediction": cad_patch_prediction,
            "cad_series_score": cad_series_score,
            "cad_series_p_value": cad_series_p,
            "cad_series_prediction": cad_series_prediction,
        }
    )
    metrics = pd.DataFrame.from_records(
        [
            _metric_row(
                seed,
                "VAE+MAE",
                f"Q{100 * config.detector.mae_quantile:g}; any patch",
                dataset.test_labels,
                mae_prediction,
                mae_score,
            ),
            _metric_row(
                seed,
                "VAE+CAD-W1-patch",
                f"patch alpha={config.detector.alpha:g}; any patch",
                dataset.test_labels,
                cad_patch_prediction,
                cad_patch_score,
            ),
            _metric_row(
                seed,
                "VAE+CAD-W1-series",
                f"series alpha={config.detector.alpha:g}; max W1",
                dataset.test_labels,
                cad_series_prediction,
                cad_series_score,
            ),
        ]
    )
    _atomic_csv(run_directory / "predictions.csv", predictions)
    _atomic_csv(run_directory / "metrics.csv", metrics)

    validate_cad_state(
        CADState.load(run_directory / "cad_state_patch.npz"),
        expected_alpha=config.detector.alpha,
        expected_calibration_count=config.split.calibration_series
        * config.data.patches_per_series,
    )
    validate_cad_state(
        CADState.load(run_directory / "cad_state_series.npz"),
        expected_alpha=config.detector.alpha,
        expected_calibration_count=config.split.calibration_series,
    )
    validate_prediction_table(
        run_directory / "predictions.csv", method_columns=METHOD_COLUMNS, spec=spec
    )
    validate_metrics_against_predictions(
        run_directory / "predictions.csv",
        run_directory / "metrics.csv",
        seed=seed,
        method_columns=METHOD_COLUMNS,
        spec=spec,
    )

    artifact_names = (
        "model.keras",
        "encoder.keras",
        "decoder.keras",
        "model_summary.txt",
        "history.csv",
        "split_manifest.csv",
        "normalization.npz",
        "training_request.json",
        "training_metadata.json",
        "latent_diagnostics.json",
        "latent_diagnostics.npz",
        "baseline.npz",
        "cad_state_patch.npz",
        "cad_state_series.npz",
        "patch_scores.npz",
        "predictions.csv",
        "metrics.csv",
    )
    artifacts = {
        name: {
            "bytes": (run_directory / name).stat().st_size,
            "sha256": sha256_file(run_directory / name),
        }
        for name in artifact_names
    }
    run_record: dict[str, object] = {
        "schema_version": 1,
        "status": "validated",
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "protocol_version": config.protocol_version,
        "source_tree_sha256": persisted_protocol["source_tree_sha256"],
        "configuration": config.as_dict(),
        "cache_manifest_payload_sha256": cache_manifest_sha256,
        "cache_identities": current_cache_identities,
        "protocol": persisted_protocol,
        "protocol_sha256": protocol_sha256,
        "n_bins": n_bins,
        "mae_threshold": threshold,
        "methods": list(METHOD_COLUMNS),
        "inference": (
            "decoder(z_mean); one shared calibration/test reconstruction per patch; "
            "two deterministic training passes avoid a multi-gigabyte residual cache"
        ),
        "software": _package_versions(),
        "artifacts": artifacts,
    }
    _atomic_json(run_marker, run_record)
    print(f"Seed {seed} inference validated: {run_directory}", flush=True)
    return run_directory


def main() -> None:
    args = _arguments()
    run_inference(
        seed=args.seed,
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
