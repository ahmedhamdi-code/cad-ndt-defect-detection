"""Residual-based anomaly scores and conformal anomaly detection.

The CAD implementation represents every absolute reconstruction residual as a
probability histogram.  All histograms use bin boundaries learned from normal
training data.  Values outside that range are clipped into the edge bins, so
no probability mass is discarded.  Wasserstein-1 is computed on the bin
centres with the histogram values as probability masses.

This module never reads class labels.  Fitting requires only normal training
and normal calibration residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


Aggregation = Literal["max", "top5", "q95"]
CalibrationLevel = Literal["patch", "series"]


def _finite_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _absolute_residuals(values: ArrayLike, name: str) -> NDArray[np.float64]:
    residuals = _finite_array(values, name)
    if np.any(residuals < 0.0):
        raise ValueError(f"{name} must contain absolute, non-negative residuals")
    return residuals


def _validated_edges(values: ArrayLike) -> NDArray[np.float64]:
    edges = _finite_array(values, "bin_edges").reshape(-1)
    if edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be strictly increasing and define at least two bins")
    return edges


def canonical_aggregation(value: str) -> Aggregation:
    """Return a canonical name for a supported series aggregation."""

    aliases: dict[str, Aggregation] = {
        "max": "max",
        "maximum": "max",
        "top5": "top5",
        "top-5": "top5",
        "top_5": "top5",
        "mean_top5": "top5",
        "mean_top_5": "top5",
        "q95": "q95",
        "quantile95": "q95",
        "quantile_0.95": "q95",
    }
    key = str(value).strip().lower().replace(" ", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("aggregation must be one of: max, top5, q95") from exc


def make_bin_edges(
    residual_min: float,
    residual_max: float,
    n_bins: int,
) -> NDArray[np.float64]:
    """Create shared, finite and strictly increasing residual-bin edges."""

    if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) != n_bins or n_bins < 2:
        raise ValueError("n_bins must be an integer greater than or equal to two")
    lower = float(residual_min)
    upper = float(residual_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Residual bounds must be finite")
    if lower < 0.0:
        raise ValueError("Absolute residuals cannot have a negative lower bound")
    if upper < lower:
        raise ValueError("residual_max must be greater than or equal to residual_min")
    if upper == lower:
        upper = lower + max(abs(lower), 1.0) * 1e-9
    return np.linspace(lower, upper, int(n_bins) + 1, dtype=np.float64)


def median_auto_bin_count(
    residual_batches: ArrayLike | Iterable[ArrayLike],
) -> int:
    """Return the median NumPy ``auto`` bin count across training patches.

    The input can be a single array with a leading patch dimension or an
    iterable of such batches.  Only integer counts are retained in memory.
    ``np.rint`` makes the conversion of a half-integer median explicit.
    """

    batches: Iterable[ArrayLike]
    if isinstance(residual_batches, np.ndarray):
        batches = (residual_batches,)
    else:
        batches = residual_batches
    counts: list[int] = []
    for raw_batch in batches:
        batch = _absolute_residuals(raw_batch, "residual batch")
        if batch.ndim < 2 or batch.shape[0] == 0:
            raise ValueError("Each residual batch must have a non-empty patch dimension")
        for patch in batch:
            count = np.histogram_bin_edges(patch.reshape(-1), bins="auto").size - 1
            counts.append(int(count))
    if not counts:
        raise ValueError("At least one training patch is required")
    result = int(np.rint(np.median(np.asarray(counts, dtype=np.float64))))
    return max(result, 2)


def residual_histogram(
    residual: ArrayLike,
    bin_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Convert one absolute-residual patch into a unit-mass histogram."""

    values = _absolute_residuals(residual, "residual").reshape(-1)
    edges = _validated_edges(bin_edges)
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    if int(counts.sum()) != int(values.size):
        raise RuntimeError("Histogram construction lost residual probability mass")
    histogram = counts.astype(np.float64) / float(values.size)
    if not np.isclose(histogram.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Residual histogram does not have unit mass")
    return histogram


def residual_histograms(
    residuals: ArrayLike,
    bin_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Convert a batch of residual patches into probability histograms."""

    values = _absolute_residuals(residuals, "residuals")
    edges = _validated_edges(bin_edges)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("residuals must have a non-empty leading patch dimension")
    result = np.empty((values.shape[0], edges.size - 1), dtype=np.float64)
    for index, patch in enumerate(values):
        result[index] = residual_histogram(patch, edges)
    return result


def normalize_histograms(histograms: ArrayLike) -> NDArray[np.float64]:
    """Validate and row-normalize a two-dimensional histogram array."""

    masses = _finite_array(histograms, "histograms")
    if masses.ndim != 2 or masses.shape[1] < 2:
        raise ValueError("histograms must be a two-dimensional array with at least two bins")
    if np.any(masses < 0.0):
        raise ValueError("Histogram masses cannot be negative")
    totals = masses.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Every histogram must have positive mass")
    return masses / totals


def mean_reference_histogram(histograms: ArrayLike) -> NDArray[np.float64]:
    """Return the normalized L2 barycentre of normal training histograms."""

    normalized = normalize_histograms(histograms)
    reference = normalized.mean(axis=0, dtype=np.float64)
    reference /= reference.sum()
    return reference


def fit_reference_histogram(
    residual_batches: ArrayLike | Iterable[ArrayLike],
    bin_edges: ArrayLike,
) -> tuple[NDArray[np.float64], int]:
    """Fit a reference histogram from streamed normal-training residuals."""

    edges = _validated_edges(bin_edges)
    batches: Iterable[ArrayLike]
    if isinstance(residual_batches, np.ndarray):
        batches = (residual_batches,)
    else:
        batches = residual_batches
    histogram_sum = np.zeros(edges.size - 1, dtype=np.float64)
    n_patches = 0
    for batch in batches:
        histograms = residual_histograms(batch, edges)
        histogram_sum += histograms.sum(axis=0, dtype=np.float64)
        n_patches += int(histograms.shape[0])
    if n_patches == 0:
        raise ValueError("At least one normal training residual is required")
    reference = histogram_sum / float(n_patches)
    reference /= reference.sum()
    return reference, n_patches


def wasserstein_1(
    masses_a: ArrayLike,
    masses_b: ArrayLike,
    support: ArrayLike,
) -> float:
    """Compute exact one-dimensional W1 on a common ordered support."""

    first = _finite_array(masses_a, "masses_a").reshape(-1)
    second = _finite_array(masses_b, "masses_b").reshape(-1)
    points = _finite_array(support, "support").reshape(-1)
    if first.size < 2 or first.shape != second.shape or first.shape != points.shape:
        raise ValueError("Mass vectors and support must have the same length of at least two")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise ValueError("Probability masses cannot be negative")
    if np.any(np.diff(points) <= 0.0):
        raise ValueError("support must be strictly increasing")
    first_total = float(first.sum())
    second_total = float(second.sum())
    if first_total <= 0.0 or second_total <= 0.0:
        raise ValueError("Both distributions must have positive mass")
    cdf_difference = np.cumsum(first / first_total - second / second_total)[:-1]
    return float(np.sum(np.abs(cdf_difference) * np.diff(points), dtype=np.float64))


def wasserstein_scores(
    histograms: ArrayLike,
    reference_histogram: ArrayLike,
    bin_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Compute W1 from every histogram to a common reference distribution."""

    masses = normalize_histograms(histograms)
    reference = _finite_array(reference_histogram, "reference_histogram").reshape(-1)
    edges = _validated_edges(bin_edges)
    if masses.shape[1] != reference.size or edges.size != reference.size + 1:
        raise ValueError("Histogram, reference and bin-edge shapes are incompatible")
    if np.any(reference < 0.0) or float(reference.sum()) <= 0.0:
        raise ValueError("reference_histogram must contain non-negative positive mass")
    reference = reference / reference.sum()
    centres = (edges[:-1] + edges[1:]) * 0.5
    cdf_difference = np.cumsum(masses - reference[None, :], axis=1)[:, :-1]
    scores = np.sum(np.abs(cdf_difference) * np.diff(centres)[None, :], axis=1)
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise RuntimeError("Wasserstein scores are invalid")
    return scores.astype(np.float64, copy=False)


def score_residuals(
    residuals: ArrayLike,
    bin_edges: ArrayLike,
    reference_histogram: ArrayLike,
) -> NDArray[np.float64]:
    """Histogram and W1-score one residual batch in a single call."""

    histograms = residual_histograms(residuals, bin_edges)
    return wasserstein_scores(histograms, reference_histogram, bin_edges)


def conformal_p_values(
    calibration_scores: ArrayLike,
    candidate_scores: ArrayLike,
) -> NDArray[np.float64]:
    """Compute finite-sample conformal p-values including ties and ``+1``."""

    calibration = _finite_array(calibration_scores, "calibration_scores").reshape(-1)
    candidate_array = _finite_array(candidate_scores, "candidate_scores")
    candidate_shape = candidate_array.shape
    candidates = candidate_array.reshape(-1)
    ordered = np.sort(calibration)
    first_equal = np.searchsorted(ordered, candidates, side="left")
    greater_or_equal = ordered.size - first_equal
    p_values = (1.0 + greater_or_equal) / (ordered.size + 1.0)
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise RuntimeError("Conformal p-values are outside [0, 1]")
    return p_values.reshape(candidate_shape).astype(np.float64, copy=False)


def aggregate_patch_scores(
    patch_scores: ArrayLike,
    aggregation: str = "max",
    *,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Aggregate patch anomaly scores into one score per series."""

    scores = _finite_array(patch_scores, "patch_scores")
    method = canonical_aggregation(aggregation)
    axis = int(axis)
    if scores.ndim == 0:
        raise ValueError("patch_scores must contain a patch dimension")
    n_patches = scores.shape[axis]
    if n_patches == 0:
        raise ValueError("The patch dimension must not be empty")
    if method == "max":
        return np.max(scores, axis=axis)
    if method == "q95":
        return np.quantile(scores, 0.95, axis=axis, method="linear")
    count = min(5, n_patches)
    partitioned = np.partition(scores, n_patches - count, axis=axis)
    selected = np.take(partitioned, np.arange(n_patches - count, n_patches), axis=axis)
    return np.mean(selected, axis=axis, dtype=np.float64)


def mean_absolute_residual_scores(residuals: ArrayLike) -> NDArray[np.float64]:
    """Return the mean absolute residual for every leading sample."""

    values = _absolute_residuals(residuals, "residuals")
    if values.ndim < 2:
        raise ValueError("residuals must have a leading sample dimension")
    return values.reshape(values.shape[0], -1).mean(axis=1, dtype=np.float64)


def fit_mae_threshold(training_scores: ArrayLike, quantile: float) -> float:
    """Fit a MAE threshold from normal training scores only."""

    scores = _finite_array(training_scores, "training_scores").reshape(-1)
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly inside (0, 1)")
    return float(np.quantile(scores, float(quantile), method="linear"))


def predict_mae_patch_or(
    patch_scores: ArrayLike,
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Apply the paper-like rule: a series is anomalous if any patch is."""

    scores = _finite_array(patch_scores, "patch_scores")
    if scores.ndim != 2:
        raise ValueError("patch_scores must have shape (n_series, n_patches)")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    series_scores = scores.max(axis=1)
    predictions = np.any(scores > float(threshold), axis=1).astype(np.int8)
    return series_scores, predictions


def patch_level_decision(
    calibration_patch_scores: ArrayLike,
    evaluation_patch_scores: ArrayLike,
    alpha: float = 0.02,
) -> dict[str, NDArray[np.float64] | NDArray[np.int8]]:
    """Apply patch calibration followed by the paper-like any-patch rule."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    scores = _finite_array(evaluation_patch_scores, "evaluation_patch_scores")
    if scores.ndim != 2:
        raise ValueError("evaluation_patch_scores must have shape (n_series, n_patches)")
    p_values = conformal_p_values(calibration_patch_scores, scores.reshape(-1)).reshape(scores.shape)
    return {
        "patch_p_values": p_values,
        "series_scores": scores.max(axis=1),
        "series_p_values": p_values.min(axis=1),
        "predictions": np.any(p_values <= float(alpha), axis=1).astype(np.int8),
    }


def series_level_decision(
    calibration_patch_scores: ArrayLike,
    evaluation_patch_scores: ArrayLike,
    aggregation: str = "max",
    alpha: float = 0.02,
) -> dict[str, NDArray[np.float64] | NDArray[np.int8]]:
    """Aggregate patches first and calibrate one conformal test per series."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    calibration = _finite_array(calibration_patch_scores, "calibration_patch_scores")
    evaluation = _finite_array(evaluation_patch_scores, "evaluation_patch_scores")
    if calibration.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("Calibration and evaluation patch scores must be two-dimensional")
    if calibration.shape[1] != evaluation.shape[1]:
        raise ValueError("Calibration and evaluation series must use the same patch count")
    calibration_series = aggregate_patch_scores(calibration, aggregation, axis=1)
    evaluation_series = aggregate_patch_scores(evaluation, aggregation, axis=1)
    p_values = conformal_p_values(calibration_series, evaluation_series)
    return {
        "calibration_series_scores": calibration_series,
        "series_scores": evaluation_series,
        "series_p_values": p_values,
        "predictions": (p_values <= float(alpha)).astype(np.int8),
    }


@dataclass(frozen=True, slots=True)
class CADState:
    """Persistable fitted state for patch-level or series-level CAD."""

    bin_edges: NDArray[np.float64]
    reference_histogram: NDArray[np.float64]
    calibration_scores: NDArray[np.float64]
    alpha: float = 0.02
    calibration_level: CalibrationLevel = "patch"
    aggregation: Aggregation = "max"
    n_training_histograms: int = 0

    def __post_init__(self) -> None:
        edges = _validated_edges(self.bin_edges)
        reference = _finite_array(self.reference_histogram, "reference_histogram").reshape(-1)
        calibration = _finite_array(self.calibration_scores, "calibration_scores").reshape(-1)
        level = str(self.calibration_level)
        aggregation = canonical_aggregation(self.aggregation)
        if reference.shape != (edges.size - 1,):
            raise ValueError("reference_histogram and bin_edges have incompatible shapes")
        if np.any(reference < 0.0) or float(reference.sum()) <= 0.0:
            raise ValueError("reference_histogram must contain non-negative positive mass")
        if np.any(calibration < 0.0):
            raise ValueError("calibration_scores cannot be negative")
        if level not in ("patch", "series"):
            raise ValueError("calibration_level must be 'patch' or 'series'")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        if int(self.n_training_histograms) < 0:
            raise ValueError("n_training_histograms cannot be negative")
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "reference_histogram", reference / reference.sum())
        object.__setattr__(self, "calibration_scores", calibration)
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "calibration_level", level)
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(self, "n_training_histograms", int(self.n_training_histograms))

    @property
    def n_bins(self) -> int:
        return int(self.reference_histogram.size)

    @property
    def bin_centres(self) -> NDArray[np.float64]:
        return (self.bin_edges[:-1] + self.bin_edges[1:]) * 0.5

    @property
    def minimum_p_value(self) -> float:
        return 1.0 / (self.calibration_scores.size + 1.0)

    def score_residuals(self, residuals: ArrayLike) -> NDArray[np.float64]:
        """Compute patch-level W1 scores from absolute residuals."""

        histograms = residual_histograms(residuals, self.bin_edges)
        return wasserstein_scores(histograms, self.reference_histogram, self.bin_edges)

    def p_values(self, scores: ArrayLike) -> NDArray[np.float64]:
        """Compute p-values against the fitted calibration scores."""

        return conformal_p_values(self.calibration_scores, scores)

    def predict_from_patch_scores(
        self,
        patch_scores: ArrayLike,
    ) -> dict[str, NDArray[np.float64] | NDArray[np.int8]]:
        """Apply the state's fitted calibration to grouped patch scores."""

        scores = _finite_array(patch_scores, "patch_scores")
        if scores.ndim != 2:
            raise ValueError("patch_scores must have shape (n_series, n_patches)")
        if self.calibration_level == "patch":
            p_values = self.p_values(scores.reshape(-1)).reshape(scores.shape)
            return {
                "patch_p_values": p_values,
                "series_scores": scores.max(axis=1),
                "series_p_values": p_values.min(axis=1),
                "predictions": np.any(p_values <= self.alpha, axis=1).astype(np.int8),
            }
        series_scores = aggregate_patch_scores(scores, self.aggregation, axis=1)
        p_values = self.p_values(series_scores)
        return {
            "series_scores": series_scores,
            "series_p_values": p_values,
            "predictions": (p_values <= self.alpha).astype(np.int8),
        }

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        """Return a JSON-compatible description of the fitted state."""

        result: dict[str, object] = {
            "n_bins": self.n_bins,
            "alpha": self.alpha,
            "calibration_level": self.calibration_level,
            "aggregation": self.aggregation,
            "n_training_histograms": self.n_training_histograms,
            "n_calibration_scores": int(self.calibration_scores.size),
            "minimum_p_value": self.minimum_p_value,
            "residual_min": float(self.bin_edges[0]),
            "residual_max": float(self.bin_edges[-1]),
        }
        if include_arrays:
            result.update(
                {
                    "bin_edges": self.bin_edges.tolist(),
                    "bin_centres": self.bin_centres.tolist(),
                    "reference_histogram": self.reference_histogram.tolist(),
                    "calibration_scores": self.calibration_scores.tolist(),
                }
            )
        return result

    def save(self, path: str | Path) -> Path:
        """Save the state in a non-pickled NumPy archive."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            bin_edges=self.bin_edges,
            reference_histogram=self.reference_histogram,
            calibration_scores=self.calibration_scores,
            alpha=np.asarray(self.alpha, dtype=np.float64),
            calibration_level=np.asarray(self.calibration_level),
            aggregation=np.asarray(self.aggregation),
            n_training_histograms=np.asarray(self.n_training_histograms, dtype=np.int64),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CADState":
        """Load a state saved by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                bin_edges=payload["bin_edges"],
                reference_histogram=payload["reference_histogram"],
                calibration_scores=payload["calibration_scores"],
                alpha=float(payload["alpha"].item()),
                calibration_level=str(payload["calibration_level"].item()),
                aggregation=str(payload["aggregation"].item()),
                n_training_histograms=int(payload["n_training_histograms"].item()),
            )


def build_cad_state(
    bin_edges: ArrayLike,
    reference_histogram: ArrayLike,
    calibration_histograms: ArrayLike,
    *,
    alpha: float = 0.02,
    calibration_level: CalibrationLevel = "patch",
    aggregation: Aggregation = "max",
    patches_per_series: int | None = None,
    n_training_histograms: int = 0,
) -> CADState:
    """Build CAD state from a fitted reference and normal calibration data.

    For series-level calibration, either pass a three-dimensional histogram
    array ``(n_series, n_patches, n_bins)`` or pass a two-dimensional array and
    specify ``patches_per_series``.
    """

    edges = _validated_edges(bin_edges)
    reference = _finite_array(reference_histogram, "reference_histogram").reshape(-1)
    histograms = _finite_array(calibration_histograms, "calibration_histograms")
    level = str(calibration_level)
    if level == "patch":
        if histograms.ndim != 2:
            raise ValueError("Patch calibration histograms must be two-dimensional")
        calibration_scores = wasserstein_scores(histograms, reference, edges)
    elif level == "series":
        if histograms.ndim == 3:
            shape = histograms.shape[:2]
            flat = histograms.reshape(-1, histograms.shape[-1])
        elif histograms.ndim == 2 and patches_per_series is not None:
            if int(patches_per_series) <= 0 or histograms.shape[0] % int(patches_per_series):
                raise ValueError("Calibration patches cannot be grouped into complete series")
            shape = (histograms.shape[0] // int(patches_per_series), int(patches_per_series))
            flat = histograms
        else:
            raise ValueError("Series calibration requires grouped histograms or patches_per_series")
        patch_scores = wasserstein_scores(flat, reference, edges).reshape(shape)
        calibration_scores = aggregate_patch_scores(patch_scores, aggregation, axis=1)
    else:
        raise ValueError("calibration_level must be 'patch' or 'series'")
    return CADState(
        bin_edges=edges,
        reference_histogram=reference,
        calibration_scores=calibration_scores,
        alpha=alpha,
        calibration_level=level,  # type: ignore[arg-type]
        aggregation=aggregation,
        n_training_histograms=n_training_histograms,
    )


def fit_patch_cad_state(
    bin_edges: ArrayLike,
    reference_histogram: ArrayLike,
    calibration_patch_scores: ArrayLike,
    *,
    alpha: float = 0.02,
    n_training_histograms: int = 0,
) -> CADState:
    """Build a patch-calibrated state from precomputed normal W1 scores."""

    return CADState(
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        reference_histogram=np.asarray(reference_histogram, dtype=np.float64),
        calibration_scores=np.asarray(calibration_patch_scores, dtype=np.float64).reshape(-1),
        alpha=alpha,
        calibration_level="patch",
        aggregation="max",
        n_training_histograms=n_training_histograms,
    )


def fit_series_cad_state(
    bin_edges: ArrayLike,
    reference_histogram: ArrayLike,
    calibration_patch_scores: ArrayLike,
    *,
    aggregation: str = "max",
    alpha: float = 0.02,
    n_training_histograms: int = 0,
) -> CADState:
    """Build a series-calibrated state from grouped normal patch W1 scores."""

    calibration = _finite_array(calibration_patch_scores, "calibration_patch_scores")
    if calibration.ndim != 2:
        raise ValueError("calibration_patch_scores must have shape (n_series, n_patches)")
    return CADState(
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        reference_histogram=np.asarray(reference_histogram, dtype=np.float64),
        calibration_scores=aggregate_patch_scores(calibration, aggregation, axis=1),
        alpha=alpha,
        calibration_level="series",
        aggregation=canonical_aggregation(aggregation),
        n_training_histograms=n_training_histograms,
    )


def predict_patch_or(
    state: CADState,
    patch_scores: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Apply patch-calibrated CAD and the paper-like any-patch series rule."""

    if state.calibration_level != "patch":
        raise ValueError("predict_patch_or requires a patch-calibrated CAD state")
    scores = _finite_array(patch_scores, "patch_scores")
    if scores.ndim != 2:
        raise ValueError("patch_scores must have shape (n_series, n_patches)")
    patch_p_values = state.p_values(scores.reshape(-1)).reshape(scores.shape)
    series_scores = scores.max(axis=1)
    minimum_p_values = patch_p_values.min(axis=1)
    predictions = np.any(patch_p_values <= state.alpha, axis=1).astype(np.int8)
    return series_scores, minimum_p_values, predictions


def predict_series_cad(
    state: CADState,
    patch_scores: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Aggregate patches first, then apply series-level conformal calibration."""

    if state.calibration_level != "series":
        raise ValueError("predict_series_cad requires a series-calibrated CAD state")
    scores = _finite_array(patch_scores, "patch_scores")
    if scores.ndim != 2:
        raise ValueError("patch_scores must have shape (n_series, n_patches)")
    series_scores = aggregate_patch_scores(scores, state.aggregation, axis=1)
    p_values = state.p_values(series_scores)
    predictions = (p_values <= state.alpha).astype(np.int8)
    return series_scores, p_values, predictions
