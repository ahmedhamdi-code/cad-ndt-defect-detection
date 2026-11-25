"""Binary anomaly-detection metrics and repeated-run aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


PERFORMANCE_METRICS = (
    "sensitivity",
    "specificity",
    "f1_anomaly",
    "f1_macro",
    "accuracy",
    "auroc",
    "auprc",
)
CONFUSION_METRICS = ("tn", "fp", "fn", "tp")
SUMMARY_METRICS = (*PERFORMANCE_METRICS, *CONFUSION_METRICS)
DEFAULT_SEEDS = (42, 43, 44)


def _binary_vector(values: ArrayLike, name: str) -> NDArray[np.int8]:
    raw = np.asarray(values).reshape(-1)
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric binary values") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return numeric.astype(np.int8)


def _score_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must be non-empty and finite")
    return scores


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """Return TN, FP, FN and TP with anomaly encoded as positive class 1."""

    truth = _binary_vector(y_true, "y_true")
    prediction = _binary_vector(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return {
        "tn": int(np.count_nonzero((truth == 0) & (prediction == 0))),
        "fp": int(np.count_nonzero((truth == 0) & (prediction == 1))),
        "fn": int(np.count_nonzero((truth == 1) & (prediction == 0))),
        "tp": int(np.count_nonzero((truth == 1) & (prediction == 1))),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    anomaly_scores: ArrayLike,
) -> dict[str, object]:
    """Compute threshold and ranking metrics at the series level.

    Larger continuous scores must indicate greater abnormality.  AUROC and
    AUPRC are computed from these scores, never from binary predictions.
    """

    truth = _binary_vector(y_true, "y_true")
    prediction = _binary_vector(y_pred, "y_pred")
    scores = _score_vector(anomaly_scores, "anomaly_scores")
    if truth.shape != prediction.shape or truth.shape != scores.shape:
        raise ValueError("y_true, y_pred and anomaly_scores must have equal length")
    if np.unique(truth).size != 2:
        raise ValueError("Both normal and anomalous classes are required")

    counts = confusion_counts(truth, prediction)
    tn, fp, fn, tp = (counts[name] for name in CONFUSION_METRICS)
    f1_anomaly = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    f1_normal = _safe_ratio(2 * tn, 2 * tn + fp + fn)
    return {
        "n_series": int(truth.size),
        "n_normal": int(np.count_nonzero(truth == 0)),
        "n_anomaly": int(np.count_nonzero(truth == 1)),
        **counts,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "f1_anomaly": f1_anomaly,
        "f1_binary": f1_anomaly,
        "f1_macro": float((f1_anomaly + f1_normal) * 0.5),
        "accuracy": _safe_ratio(tp + tn, truth.size),
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
    }


compute_metrics = classification_metrics


def _as_frame(rows: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy(deep=True)
    else:
        frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        raise ValueError("The per-run metric table must not be empty")
    if "f1_anomaly" not in frame and "f1_binary" in frame:
        frame["f1_anomaly"] = frame["f1_binary"]
    if "f1_binary" not in frame and "f1_anomaly" in frame:
        frame["f1_binary"] = frame["f1_anomaly"]
    return frame


def validate_per_run_metrics(
    rows: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate one complete metric row per method and requested seed."""

    frame = _as_frame(rows)
    required = {"seed", "method", *SUMMARY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Per-run metrics are missing columns: {missing}")

    expected_seed_tuple = tuple(int(seed) for seed in expected_seeds)
    if len(expected_seed_tuple) < 2 or len(set(expected_seed_tuple)) != len(expected_seed_tuple):
        raise ValueError("At least two distinct expected seeds are required for sample SD")
    seed_values = pd.to_numeric(frame["seed"], errors="raise").to_numpy(np.float64)
    if not np.all(np.isfinite(seed_values)) or not np.all(seed_values == np.floor(seed_values)):
        raise ValueError("Seeds must be finite integers")
    frame["seed"] = seed_values.astype(np.int64)

    methods = (
        tuple(str(method) for method in expected_methods)
        if expected_methods is not None
        else tuple(sorted(frame["method"].astype(str).unique()))
    )
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Expected methods must be distinct and non-empty")
    if set(frame["seed"]) != set(expected_seed_tuple):
        raise ValueError(f"Expected exactly seeds {expected_seed_tuple}")
    if set(frame["method"].astype(str)) != set(methods):
        raise ValueError(f"Expected exactly methods {methods}")
    if frame.duplicated(["seed", "method"]).any():
        raise ValueError("Duplicate seed/method metric rows were found")
    if len(frame) != len(expected_seed_tuple) * len(methods):
        raise ValueError("Expected exactly one metric row per seed and method")

    for column in SUMMARY_METRICS:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Metric {column!r} contains NaN or infinity")
        if column in PERFORMANCE_METRICS and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Metric {column!r} must lie in [0, 1]")
        if column in CONFUSION_METRICS and (
            np.any(values < 0.0) or not np.all(values == np.floor(values))
        ):
            raise ValueError(f"Confusion count {column!r} must be a non-negative integer")
        frame[column] = values

    for _, row in frame.iterrows():
        total = int(row["tn"] + row["fp"] + row["fn"] + row["tp"])
        if "n_series" in frame and total != int(row["n_series"]):
            raise ValueError("Confusion counts do not sum to n_series")
        sensitivity = _safe_ratio(row["tp"], row["tp"] + row["fn"])
        specificity = _safe_ratio(row["tn"], row["tn"] + row["fp"])
        f1_anomaly = _safe_ratio(2 * row["tp"], 2 * row["tp"] + row["fp"] + row["fn"])
        accuracy = _safe_ratio(row["tp"] + row["tn"], total)
        checks = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_anomaly": f1_anomaly,
            "accuracy": accuracy,
        }
        for metric, expected in checks.items():
            if not np.isclose(float(row[metric]), expected, rtol=1e-9, atol=1e-12):
                raise ValueError(f"Saved {metric} is inconsistent with confusion counts")
    return frame.sort_values(["seed", "method"]).reset_index(drop=True)


def summarize_runs(
    rows: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_methods: Sequence[str] | None = None,
    dataset: str = "AIRBUS",
) -> pd.DataFrame:
    """Compute mean and sample SD across independent seeded runs.

    The sample standard deviation uses ``ddof=1`` and therefore divides the
    sum of squared deviations by ``n_runs - 1``.
    """

    frame = validate_per_run_metrics(
        rows,
        expected_seeds=expected_seeds,
        expected_methods=expected_methods,
    )
    methods = (
        tuple(str(method) for method in expected_methods)
        if expected_methods is not None
        else tuple(sorted(frame["method"].astype(str).unique()))
    )
    records: list[dict[str, object]] = []
    for method in methods:
        subset = frame.loc[frame["method"].astype(str) == method].sort_values("seed")
        for metric in SUMMARY_METRICS:
            values = subset[metric].to_numpy(np.float64)
            mean = float(np.mean(values))
            sample_std = float(np.std(values, ddof=1))
            records.append(
                {
                    "dataset": str(dataset),
                    "method": method,
                    "metric": metric,
                    "n_runs": int(values.size),
                    "mean": mean,
                    "sample_std": sample_std,
                    "ddof": 1,
                    "formatted_mean_std": f"{mean:.3f} ± {sample_std:.3f}",
                }
            )
    return pd.DataFrame.from_records(records)


aggregate_runs = summarize_runs


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace one CSV only after its complete temporary file is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_run_metrics(
    outputs_root: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    filename: str = "metrics.csv",
) -> pd.DataFrame:
    """Collect per-seed metric files from ``outputs/seed_<seed>`` folders."""

    root = Path(outputs_root)
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"seed_{int(seed)}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing metric file: {path}")
        frame = pd.read_csv(path, float_precision="round_trip")
        if "seed" not in frame:
            frame.insert(0, "seed", int(seed))
        elif not np.all(pd.to_numeric(frame["seed"], errors="raise") == int(seed)):
            raise ValueError(f"{path} contains an unexpected seed")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_aggregate_metrics(
    outputs_root: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_methods: Sequence[str] | None = None,
    dataset: str = "AIRBUS",
) -> tuple[Path, Path]:
    """Validate, aggregate and save the public per-run and summary tables."""

    root = Path(outputs_root)
    root.mkdir(parents=True, exist_ok=True)
    per_run = collect_run_metrics(root, seeds=seeds)
    per_run = validate_per_run_metrics(
        per_run,
        expected_seeds=seeds,
        expected_methods=expected_methods,
    )
    summary = summarize_runs(
        per_run,
        expected_seeds=seeds,
        expected_methods=expected_methods,
        dataset=dataset,
    )
    per_run_path = root / "metrics_per_run.csv"
    summary_path = root / "metrics_summary.csv"
    _atomic_write_csv(per_run, per_run_path)
    _atomic_write_csv(summary, summary_path)
    return per_run_path, summary_path
