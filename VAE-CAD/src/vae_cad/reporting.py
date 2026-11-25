"""Generate a compact, auditable Markdown reproduction report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .metrics import DEFAULT_SEEDS, PERFORMANCE_METRICS, summarize_runs


METRIC_LABELS = {
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1_anomaly": "Binary F1 (anomaly)",
    "f1_binary": "Binary F1 (anomaly)",
    "f1_macro": "Macro F1",
    "accuracy": "Accuracy",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


def _format_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def _cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _markdown_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    header = "| " + " | ".join(_cell(value) for value in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _configuration_mapping(configuration: object | None) -> dict[str, Any]:
    if configuration is None:
        return {}
    if isinstance(configuration, Mapping):
        return {str(key): value for key, value in configuration.items()}
    if is_dataclass(configuration):
        return asdict(configuration)
    if hasattr(configuration, "as_dict"):
        value = configuration.as_dict()
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    if hasattr(configuration, "to_dict"):
        value = configuration.to_dict()
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    if hasattr(configuration, "manifest_parameters"):
        value = configuration.manifest_parameters()
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    return {"value": str(configuration)}


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _load_tables(
    outputs_root: Path,
    per_run: pd.DataFrame | str | Path | None,
    summary: pd.DataFrame | str | Path | None,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(per_run, pd.DataFrame):
        per_run_frame = per_run.copy(deep=True)
    else:
        per_run_path = Path(per_run) if per_run is not None else outputs_root / "metrics_per_run.csv"
        per_run_frame = pd.read_csv(per_run_path, float_precision="round_trip")
    if "f1_anomaly" not in per_run_frame and "f1_binary" in per_run_frame:
        per_run_frame["f1_anomaly"] = per_run_frame["f1_binary"]

    methods = tuple(sorted(per_run_frame["method"].astype(str).unique()))
    if isinstance(summary, pd.DataFrame):
        summary_frame = summary.copy(deep=True)
    elif summary is not None:
        summary_frame = pd.read_csv(Path(summary), float_precision="round_trip")
    else:
        summary_path = outputs_root / "metrics_summary.csv"
        summary_frame = (
            pd.read_csv(summary_path, float_precision="round_trip")
            if summary_path.is_file()
            else summarize_runs(
                per_run_frame,
                expected_seeds=seeds,
                expected_methods=methods,
            )
        )
    return per_run_frame, summary_frame


def _summary_value(summary: pd.DataFrame, method: str, metric: str) -> str:
    rows = summary.loc[
        (summary["method"].astype(str) == method)
        & (summary["metric"].astype(str) == metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {method}/{metric}")
    row = rows.iloc[0]
    return f"{_format_number(row['mean'])} ± {_format_number(row['sample_std'])}"


def _rule_for_method(method: str, rows: pd.DataFrame) -> str:
    subset = rows.loc[rows["method"].astype(str) == method]
    if "rule" in subset and subset["rule"].notna().all():
        values = tuple(subset["rule"].astype(str).unique())
        if len(values) == 1:
            return values[0]
    lowered = method.lower()
    if "mae" in lowered:
        return "training-normal MAE quantile"
    if "paper" in lowered or "patch" in lowered:
        return "patch p-value, any-patch rule"
    if "cad" in lowered or "wasserstein" in lowered or "w1" in lowered:
        return "series-level conformal W1"
    return "fixed before test evaluation"


def generate_reproduction_report(
    outputs_root: str | Path,
    *,
    per_run: pd.DataFrame | str | Path | None = None,
    summary: pd.DataFrame | str | Path | None = None,
    output_path: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    configuration: object | None = None,
) -> Path:
    """Write a report containing values computed by completed local runs."""

    root = Path(outputs_root).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "reproduction_report.md"
    )
    seed_tuple = tuple(int(seed) for seed in seeds)
    n_runs = len(seed_tuple)
    if n_runs < 2:
        raise ValueError("At least two runs are required for a sample standard deviation")
    per_run_frame, summary_frame = _load_tables(root, per_run, summary, seed_tuple)
    methods = tuple(sorted(per_run_frame["method"].astype(str).unique()))

    per_run_rows: list[list[str]] = []
    for _, row in per_run_frame.sort_values(["seed", "method"]).iterrows():
        confusion = (
            f"[[{int(row['tn'])}, {int(row['fp'])}], "
            f"[{int(row['fn'])}, {int(row['tp'])}]]"
        )
        per_run_rows.append(
            [
                str(int(row["seed"])),
                str(row["method"]),
                confusion,
                *(_format_number(row[metric]) for metric in PERFORMANCE_METRICS),
            ]
        )

    aggregate_rows: list[list[str]] = []
    for method in methods:
        n_rows = summary_frame.loc[
            (summary_frame["method"].astype(str) == method)
            & (summary_frame["metric"].astype(str) == "sensitivity")
        ]
        if len(n_rows) != 1:
            raise ValueError(f"Cannot determine n_runs for {method}")
        aggregate_rows.append(
            [
                method,
                _rule_for_method(method, per_run_frame),
                str(int(n_rows.iloc[0]["n_runs"])),
                *(_summary_value(summary_frame, method, metric) for metric in PERFORMANCE_METRICS),
            ]
        )

    configuration_map = _configuration_mapping(configuration)
    configuration_block = json.dumps(configuration_map, indent=2, sort_keys=True, default=str)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    seed_text = ", ".join(str(seed) for seed in seed_tuple)
    text = f"""# AIRBUS VAE-CAD reproduction report

Generated: {now}

## Scope

This experiment applies a variational autoencoder and conformal anomaly detection to the AIRBUS dataset. The complete protocol was executed independently with seeds {seed_text}. Only normal development series are used to train the VAE, define residual distributions, learn the MAE threshold and calibrate conformal p-values. Test labels are used only for final evaluation.

All detector settings are fixed before the labelled test set is evaluated.

## Data representation

Each time series is transformed into signed real-valued STFT patches. Training, validation and conformal calibration are separated at the series level before patches are expanded. Standardization parameters are fitted exclusively on normal training patches. Reconstructions are converted back to the original STFT units before residual scores are computed.

## Model

The model is a convolutional VAE with distinct `z_mean` and `z_log_var` outputs, stochastic reparameterized sampling during training and a reconstruction-plus-KL objective. Inference is deterministic and uses `decoder(z_mean)`. The linear decoder output is compatible with signed standardized STFT values. Because the publication does not fully specify an AIRBUS VAE architecture, this implementation makes the chosen architecture and training settings explicit. For numerical safety, the effective `z_log_var` used for sampling and KL is capped above at 10; the raw encoder output is retained for diagnostics.

## Detection rules

- **VAE+MAE** uses mean absolute reconstruction error. Its threshold is fitted from normal training data only.
- **VAE+CAD-W1 (patch-level)** represents each absolute residual as a normalized histogram, computes Wasserstein-1 on shared bin centres, obtains finite-sample conformal p-values from patches belonging to independently held-out normal series and applies the paper-like any-patch rule at series level.
- **VAE+CAD-W1 (series-level)** first aggregates the patch W1 scores of each series, then calibrates one conformal p-value per series. This makes alpha a series-level decision parameter.

Residuals outside the normal training range are clipped into the edge bins. Consequently, every pixel contributes to a histogram and no probability mass is lost. Conformal p-values use `(1 + number of calibration scores >= candidate score) / (n_calibration + 1)`.

## Configuration

```json
{configuration_block}
```

## Results by run

{_markdown_table(
    [
        "Seed", "Method", "Confusion matrix", "Sensitivity", "Specificity",
        "Binary F1 (anomaly)", "Macro F1", "Accuracy", "AUROC", "AUPRC",
    ],
    per_run_rows,
)}

The confusion matrix order is `[[TN, FP], [FN, TP]]`. AUROC and AUPRC use continuous anomaly scores rather than binary decisions.

## Aggregate results

Each metric is computed on a complete test set and then aggregated across independent seeded runs.

{_markdown_table(
    [
        "Method", "Rule", "n", "Sensitivity", "Specificity",
        "Binary F1 (anomaly)", "Macro F1", "Accuracy", "AUROC", "AUPRC",
    ],
    aggregate_rows,
)}

The reported standard deviation is the between-run sample standard deviation (`ddof=1`). With {n_runs} runs, squared deviations from the mean are divided by `{n_runs} - 1 = {n_runs - 1}`. It measures variability caused by the seeded partition, model initialization and optimization. It is not an image-level standard deviation and it is not a confidence interval.

## Reproducibility and interpretation

The binary anomaly F1 treats anomaly as the positive class. Macro F1 gives equal weight to the normal-class F1 and anomaly-class F1. These quantities answer different questions and should not be interchanged.

Patches from one calibration series are temporally related. This dependence weakens the patch-level exchangeability assumption, and those patches must not be interpreted as independent series-level observations. Patch-level alpha is not a false-positive guarantee at series level because every series contains multiple patch tests. Series-level calibration avoids that multiplicity by producing one calibrated score and p-value per series.

The {n_runs} runs provide an estimate of run-to-run variation, but a larger seed sample would give a more stable estimate. The results are conditional on the supplied dataset split and preprocessing protocol. They should not be interpreted as uncertainty for new aircraft, acquisition conditions or domains without an appropriate grouped external evaluation.

## Output files

The reproducibility artifacts include per-run models, split manifests, fitted normalization and detector states, predictions and metrics. `metrics_per_run.csv` contains one row per seed and method. `metrics_summary.csv` contains the mean and sample standard deviation for every reported metric.
"""
    return _atomic_write_text(destination, text)


write_reproduction_report = generate_reproduction_report
