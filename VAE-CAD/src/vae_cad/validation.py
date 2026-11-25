"""Consistency checks for the public AIRBUS VAE-CAD pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .cad import CADState, conformal_p_values
from .metrics import (
    DEFAULT_SEEDS,
    PERFORMANCE_METRICS,
    SUMMARY_METRICS,
    classification_metrics,
    summarize_runs,
    validate_per_run_metrics,
)


DEFAULT_METHOD_COLUMNS: dict[str, tuple[str, str]] = {
    "VAE+MAE": ("mae_score", "mae_prediction"),
    "VAE+CAD-W1-patch": ("cad_patch_score", "cad_patch_prediction"),
    "VAE+CAD-W1-series": ("cad_series_score", "cad_series_prediction"),
}


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    """Expected AIRBUS geometry and repeated-run protocol."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    n_test_series: int = 594
    n_test_normal: int = 297
    n_test_anomaly: int = 297
    patches_per_series: int = 120
    n_train_series: int = 1_206
    n_validation_series: int = 135
    n_calibration_series: int = 336

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Validation requires at least two distinct seeds")
        counts = (
            self.n_test_series,
            self.n_test_normal,
            self.n_test_anomaly,
            self.patches_per_series,
            self.n_train_series,
            self.n_validation_series,
            self.n_calibration_series,
        )
        if any(int(value) <= 0 for value in counts):
            raise ValueError("All expected counts must be positive")
        if self.n_test_normal + self.n_test_anomaly != self.n_test_series:
            raise ValueError("Normal and anomalous test counts must sum to n_test_series")


def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise ValueError(f"Validation failed: {message}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read CSV file: {path}") from exc


def _binary(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    _require(np.all(np.isfinite(result)), f"{name} contains NaN or infinity")
    _require(np.all(np.isin(result, (0.0, 1.0))), f"{name} must contain only 0 and 1")
    return result.astype(np.int8)


def _finite(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    _require(result.size > 0, f"{name} must not be empty")
    _require(np.all(np.isfinite(result)), f"{name} contains NaN or infinity")
    return result


def _close(left: object, right: object, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_cad_state(
    state: CADState,
    *,
    expected_alpha: float | None = None,
    expected_calibration_count: int | None = None,
) -> dict[str, object]:
    """Audit histogram masses, scores and conformal finite-sample bounds."""

    _require(np.all(np.isfinite(state.bin_edges)), "CAD bin edges are not finite")
    _require(np.all(np.diff(state.bin_edges) > 0.0), "CAD bin edges are not increasing")
    _require(np.all(np.isfinite(state.reference_histogram)), "CAD reference is not finite")
    _require(np.all(state.reference_histogram >= 0.0), "CAD reference has negative mass")
    _require(
        np.isclose(state.reference_histogram.sum(), 1.0, rtol=0.0, atol=1e-12),
        "CAD reference histogram does not sum to one",
    )
    _require(np.all(np.isfinite(state.calibration_scores)), "CAD calibration has non-finite scores")
    _require(np.all(state.calibration_scores >= 0.0), "CAD calibration has negative scores")
    if expected_alpha is not None:
        _require(_close(state.alpha, expected_alpha), "CAD alpha differs from the configuration")
    if expected_calibration_count is not None:
        _require(
            state.calibration_scores.size == int(expected_calibration_count),
            "CAD calibration count differs from the expected count",
        )
    calibration_p = conformal_p_values(state.calibration_scores, state.calibration_scores)
    _require(np.all((calibration_p >= 0.0) & (calibration_p <= 1.0)), "CAD p-values are invalid")
    _require(
        _close(state.minimum_p_value, 1.0 / (state.calibration_scores.size + 1.0)),
        "CAD finite-sample minimum p-value is incorrect",
    )
    return {
        "n_bins": state.n_bins,
        "calibration_level": state.calibration_level,
        "aggregation": state.aggregation,
        "n_calibration_scores": int(state.calibration_scores.size),
        "minimum_p_value": state.minimum_p_value,
        "reference_mass": float(state.reference_histogram.sum()),
    }


def validate_split_manifest(
    manifest: pd.DataFrame | str | Path,
    *,
    seed: int,
    spec: ValidationSpec = ValidationSpec(),
) -> pd.DataFrame:
    """Validate disjoint train, validation and calibration series partitions."""

    frame = _read_csv(Path(manifest)) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"partition", "series_id"}
    _require(required.issubset(frame.columns), f"split manifest lacks columns {sorted(required - set(frame.columns))}")
    if "seed" in frame:
        seeds = _finite(frame["seed"], "split seed")
        _require(np.all(seeds == int(seed)), "split manifest contains an unexpected seed")
    aliases = {
        "train": "train",
        "vae_train": "train",
        "validation": "validation",
        "val": "validation",
        "vae_validation": "validation",
        "calibration": "calibration",
        "cad_calibration": "calibration",
    }
    canonical = frame["partition"].astype(str).map(aliases)
    _require(canonical.notna().all(), "split manifest contains an unknown partition")
    frame = frame.assign(_canonical_partition=canonical)
    expected = {
        "train": spec.n_train_series,
        "validation": spec.n_validation_series,
        "calibration": spec.n_calibration_series,
    }
    _require(frame.groupby("_canonical_partition").size().to_dict() == expected, "split counts are incorrect")
    _require(not frame["series_id"].duplicated().any(), "series partitions overlap")
    _require(len(frame) == sum(expected.values()), "split manifest has an unexpected total size")
    return frame.drop(columns="_canonical_partition")


def validate_prediction_table(
    predictions: pd.DataFrame | str | Path,
    *,
    method_columns: Mapping[str, tuple[str, str]] = DEFAULT_METHOD_COLUMNS,
    spec: ValidationSpec = ValidationSpec(),
) -> pd.DataFrame:
    """Validate labels, identifiers, method scores, p-values and decisions."""

    frame = (
        _read_csv(Path(predictions))
        if not isinstance(predictions, pd.DataFrame)
        else predictions.copy(deep=True)
    )
    _require(len(frame) == spec.n_test_series, "prediction table has the wrong number of series")
    _require("series_id" in frame and "y_true" in frame, "prediction table lacks series_id or y_true")
    _require(not frame["series_id"].duplicated().any(), "prediction table has duplicate series IDs")
    truth = _binary(frame["y_true"], "y_true")
    _require(np.count_nonzero(truth == 0) == spec.n_test_normal, "normal test count is incorrect")
    _require(np.count_nonzero(truth == 1) == spec.n_test_anomaly, "anomalous test count is incorrect")
    for method, (score_column, prediction_column) in method_columns.items():
        _require(score_column in frame, f"{method} score column {score_column!r} is missing")
        _require(prediction_column in frame, f"{method} prediction column {prediction_column!r} is missing")
        scores = _finite(frame[score_column], f"{method} scores")
        _require(np.all(scores >= 0.0), f"{method} anomaly scores cannot be negative")
        _binary(frame[prediction_column], f"{method} predictions")
    for column in frame.columns:
        if "p_value" in str(column).lower() or str(column).lower().endswith("_p"):
            values = _finite(frame[column], column)
            _require(np.all((values >= 0.0) & (values <= 1.0)), f"{column} is outside [0, 1]")
    return frame


def validate_metrics_against_predictions(
    predictions: pd.DataFrame | str | Path,
    metrics: pd.DataFrame | str | Path,
    *,
    seed: int,
    method_columns: Mapping[str, tuple[str, str]] = DEFAULT_METHOD_COLUMNS,
    spec: ValidationSpec = ValidationSpec(),
) -> pd.DataFrame:
    """Recompute every saved metric directly from the prediction table."""

    prediction_frame = validate_prediction_table(
        predictions,
        method_columns=method_columns,
        spec=spec,
    )
    metric_frame = _read_csv(Path(metrics)) if not isinstance(metrics, pd.DataFrame) else metrics.copy()
    _require("method" in metric_frame, "metric table lacks method")
    _require(set(metric_frame["method"].astype(str)) == set(method_columns), "metric methods differ from predictions")
    _require(len(metric_frame) == len(method_columns), "expected one metric row per method")
    if "seed" in metric_frame:
        _require(np.all(_finite(metric_frame["seed"], "metric seed") == int(seed)), "metric seed is incorrect")
    truth = _binary(prediction_frame["y_true"], "y_true")
    for method, (score_column, prediction_column) in method_columns.items():
        saved_rows = metric_frame.loc[metric_frame["method"].astype(str) == method]
        _require(len(saved_rows) == 1, f"expected one saved row for {method}")
        saved = saved_rows.iloc[0]
        calculated = classification_metrics(
            truth,
            prediction_frame[prediction_column],
            prediction_frame[score_column],
        )
        for metric in (*PERFORMANCE_METRICS, "tn", "fp", "fn", "tp"):
            _require(metric in saved.index, f"saved metrics lack {method}/{metric}")
            _require(_close(saved[metric], calculated[metric]), f"saved {method}/{metric} is inconsistent")
    return metric_frame


def validate_metrics_summary(
    per_run: pd.DataFrame | str | Path,
    summary: pd.DataFrame | str | Path,
    *,
    spec: ValidationSpec = ValidationSpec(),
    expected_methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Recompute and compare means and sample standard deviations."""

    per_run_frame = _read_csv(Path(per_run)) if not isinstance(per_run, pd.DataFrame) else per_run.copy()
    methods = (
        tuple(str(method) for method in expected_methods)
        if expected_methods is not None
        else tuple(sorted(per_run_frame["method"].astype(str).unique()))
    )
    validated = validate_per_run_metrics(
        per_run_frame,
        expected_seeds=spec.seeds,
        expected_methods=methods,
    )
    expected = summarize_runs(
        validated,
        expected_seeds=spec.seeds,
        expected_methods=methods,
    )
    actual = _read_csv(Path(summary)) if not isinstance(summary, pd.DataFrame) else summary.copy()
    required = {"method", "metric", "n_runs", "mean", "sample_std", "ddof"}
    _require(required.issubset(actual.columns), "summary table lacks required columns")
    _require(len(actual) == len(expected), "summary table has an unexpected number of rows")
    for _, expected_row in expected.iterrows():
        rows = actual.loc[
            (actual["method"].astype(str) == str(expected_row["method"]))
            & (actual["metric"].astype(str) == str(expected_row["metric"]))
        ]
        _require(len(rows) == 1, "summary has missing or duplicate method/metric rows")
        actual_row = rows.iloc[0]
        _require(int(actual_row["n_runs"]) == len(spec.seeds), "summary n_runs is incorrect")
        _require(int(actual_row["ddof"]) == 1, "summary must use ddof=1")
        _require(_close(actual_row["mean"], expected_row["mean"]), "summary mean is incorrect")
        _require(_close(actual_row["sample_std"], expected_row["sample_std"]), "summary sample SD is incorrect")
    return actual


def validate_run_directory(
    run_directory: str | Path,
    *,
    seed: int,
    method_columns: Mapping[str, tuple[str, str]] = DEFAULT_METHOD_COLUMNS,
    spec: ValidationSpec = ValidationSpec(),
    required_artifacts: Sequence[str] = (
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
        "run.json",
    ),
) -> dict[str, object]:
    """Audit a completed seeded run and return a concise validation record."""

    root = Path(run_directory)
    missing = [name for name in required_artifacts if not (root / name).is_file()]
    _require(not missing, f"{root.name} is missing artifacts: {missing}")
    empty = [name for name in required_artifacts if (root / name).stat().st_size == 0]
    _require(not empty, f"{root.name} has empty artifacts: {empty}")
    try:
        run_record = json.loads((root / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run marker: {root / 'run.json'}") from exc
    _require(isinstance(run_record, dict), "run.json must contain an object")
    _require(int(run_record.get("seed", -1)) == int(seed), "run marker has the wrong seed")
    status = str(run_record.get("status", "")).lower()
    _require(status in {"complete", "completed", "validated"}, "run marker is not complete")

    artifact_records = run_record.get("artifacts")
    _require(isinstance(artifact_records, Mapping), "run marker has no artifact manifest")
    required_hashed = set(required_artifacts) - {"run.json"}
    _require(
        required_hashed.issubset(artifact_records),
        "run marker omits one or more required artifact hashes",
    )
    for name, raw_record in artifact_records.items():
        _require(isinstance(name, str) and Path(name).name == name, "invalid artifact name")
        _require(isinstance(raw_record, Mapping), f"invalid artifact record for {name}")
        artifact_path = root / name
        _require(artifact_path.is_file(), f"artifact listed in run marker is missing: {name}")
        _require(
            int(raw_record.get("bytes", -1)) == artifact_path.stat().st_size,
            f"artifact size differs from run marker: {name}",
        )
        _require(
            raw_record.get("sha256") == _sha256_file(artifact_path),
            f"artifact SHA-256 differs from run marker: {name}",
        )

    configuration = run_record.get("configuration")
    protocol = run_record.get("protocol")
    _require(isinstance(configuration, Mapping), "run marker has no configuration")
    _require(isinstance(protocol, Mapping), "run marker has no protocol payload")
    _require(configuration.get("seeds") == [int(seed)], "run configuration seed is incorrect")
    normalized_configuration = dict(configuration)
    normalized_configuration.pop("seeds", None)
    expected_protocol = {
        "protocol_version": run_record.get("protocol_version"),
        "source_tree_sha256": run_record.get("source_tree_sha256"),
        "configuration_without_seed": normalized_configuration,
        "cache_manifest_payload_sha256": run_record.get(
            "cache_manifest_payload_sha256"
        ),
    }
    _require(dict(protocol) == expected_protocol, "run protocol payload is inconsistent")
    protocol_sha256 = _canonical_hash(expected_protocol)
    _require(
        run_record.get("protocol_sha256") == protocol_sha256,
        "run protocol hash is invalid",
    )

    try:
        training_metadata = json.loads(
            (root / "training_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training metadata: {root / 'training_metadata.json'}") from exc
    _require(isinstance(training_metadata, Mapping), "training metadata must be an object")
    training_contract = training_metadata.get("training_contract")
    _require(isinstance(training_contract, Mapping), "training metadata has no input contract")
    cache_identity = training_contract.get("cache")
    _require(isinstance(cache_identity, Mapping), "training contract has no cache identity")
    cache_identities = run_record.get("cache_identities")
    _require(isinstance(cache_identities, Mapping), "run marker has no cache identities")
    _require(
        set(cache_identities) == {"train", "test"},
        "run marker cache identities are incomplete",
    )
    _require(
        cache_identities.get("train") == cache_identity,
        "training and inference used different training caches",
    )
    _require(
        isinstance(cache_identities.get("test"), Mapping),
        "run marker has no valid test-cache identity",
    )
    _require(
        training_contract.get("source_tree_sha256") == run_record.get("source_tree_sha256"),
        "training and inference used different source trees",
    )

    split = validate_split_manifest(root / "split_manifest.csv", seed=seed, spec=spec)
    prediction_frame = validate_prediction_table(
        root / "predictions.csv",
        method_columns=method_columns,
        spec=spec,
    )
    metric_frame = validate_metrics_against_predictions(
        prediction_frame,
        root / "metrics.csv",
        seed=seed,
        method_columns=method_columns,
        spec=spec,
    )
    return {
        "seed": int(seed),
        "run_directory": str(root.resolve()),
        "n_split_series": int(len(split)),
        "n_methods": int(len(metric_frame)),
        "protocol_sha256": protocol_sha256,
        "cache_identities_sha256": _canonical_hash(dict(cache_identities)),
        "test_identity_sha256": _canonical_hash(
            {
                "series_id": prediction_frame["series_id"].astype(str).tolist(),
                "y_true": _binary(prediction_frame["y_true"], "y_true").tolist(),
            }
        ),
        "status": "validated",
    }


def validate_completed_runs(
    outputs_root: str | Path,
    *,
    spec: ValidationSpec = ValidationSpec(),
    method_columns: Mapping[str, tuple[str, str]] = DEFAULT_METHOD_COLUMNS,
    required_artifacts: Sequence[str] | None = None,
    expected_protocol_sha256: str | None = None,
) -> dict[str, object]:
    """Validate every seeded run before any aggregate file is replaced."""

    root = Path(outputs_root)
    records: list[dict[str, object]] = []
    for seed in spec.seeds:
        keyword = {}
        if required_artifacts is not None:
            keyword["required_artifacts"] = required_artifacts
        records.append(
            validate_run_directory(
                root / f"seed_{seed}",
                seed=seed,
                method_columns=method_columns,
                spec=spec,
                **keyword,
            )
        )
    protocol_hashes = {str(record["protocol_sha256"]) for record in records}
    _require(
        len(protocol_hashes) == 1,
        "completed runs do not use one identical protocol and cache",
    )
    actual_protocol_sha256 = next(iter(protocol_hashes))
    cache_hashes = {str(record["cache_identities_sha256"]) for record in records}
    _require(
        len(cache_hashes) == 1,
        "completed runs were trained from different cache files",
    )
    test_hashes = {str(record["test_identity_sha256"]) for record in records}
    _require(
        len(test_hashes) == 1,
        "completed runs do not contain the same ordered test series and labels",
    )
    if expected_protocol_sha256 is not None:
        _require(
            actual_protocol_sha256 == expected_protocol_sha256,
            "completed runs do not match the requested aggregation protocol",
        )
    return {
        "status": "runs_validated",
        "seeds": list(spec.seeds),
        "n_runs": len(spec.seeds),
        "methods": list(method_columns),
        "protocol_sha256": actual_protocol_sha256,
        "cache_identities_sha256": next(iter(cache_hashes)),
        "test_identity_sha256": next(iter(test_hashes)),
        "runs": records,
    }


def validate_experiment(
    outputs_root: str | Path,
    *,
    spec: ValidationSpec = ValidationSpec(),
    method_columns: Mapping[str, tuple[str, str]] = DEFAULT_METHOD_COLUMNS,
    required_artifacts: Sequence[str] | None = None,
    expected_protocol_sha256: str | None = None,
) -> dict[str, object]:
    """Validate every seed and the aggregate metric files."""

    root = Path(outputs_root)
    result = validate_completed_runs(
        root,
        spec=spec,
        method_columns=method_columns,
        required_artifacts=required_artifacts,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    validate_metrics_summary(
        root / "metrics_per_run.csv",
        root / "metrics_summary.csv",
        spec=spec,
        expected_methods=tuple(method_columns),
    )
    result["status"] = "validated"
    return result
