"""Exact AIRBUS STFT preprocessing and restartable memory-mapped caches.

The reference code implicitly discarded the imaginary part when assigning
complex STFT coefficients to a real array. This module makes the projection
explicit: it keeps ``Re(STFT)`` as signed ``float32`` values, drops the final
time column, and creates 120 chronological ``64 x 64 x 1`` patches per series.
No dataset-level normalization is performed during preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
import uuid

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy import signal

from .config import DEFAULT_CONFIG, DataConfig, ExperimentConfig, STFTConfig
from .data import (
    DatasetIndex,
    SeriesSource,
    load_cache_manifest,
    open_cache,
    open_hdf5_values,
    ordered_ids_sha256,
    sha256_file,
    validate_dataset,
)


FloatPatches = NDArray[np.float32]
ProgressCallback = Callable[[str, int, int], None]
CACHE_SCHEMA_VERSION = 2
PREPROCESSING_VERSION = "airbus-real-stft-v1"


class CacheStateError(RuntimeError):
    """Raised when partial cache artifacts cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class CacheBuildResult:
    """Paths and status returned after preparing the caches."""

    train_cache_path: Path
    test_cache_path: Path
    manifest_path: Path
    reused_complete_cache: bool
    resumed_partial_cache: bool


@dataclass(frozen=True, slots=True)
class CacheValidationReport:
    """Result of validating cache metadata and, optionally, all values."""

    train_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    dtype: str
    source_hashes_verified: bool
    finite_values_scanned: bool
    all_finite: bool
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Comparison of explicit parameters with the reference SciPy defaults."""

    frequency_axis_equal: bool
    time_axis_equal: bool
    patches_equal: bool
    max_abs_difference: float

    @property
    def passed(self) -> bool:
        return self.frequency_axis_equal and self.time_axis_equal and self.patches_equal


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON in the destination directory and atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _partial_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.stem}.partial{final_path.suffix}")


def _state_path(data: DataConfig) -> Path:
    return data.cache_root / "stft_build_state.json"


def _as_signal(
    values: ArrayLike,
    *,
    expected_samples: int,
) -> NDArray[np.float64]:
    series = np.asarray(values, dtype=np.float64)
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive.")
    if series.shape != (expected_samples,):
        raise ValueError(
            f"Expected one series with {expected_samples} samples; "
            f"got {series.shape}."
        )
    if not bool(np.isfinite(series).all()):
        raise ValueError("STFT input contains non-finite values.")
    return series


def compute_stft(
    series: ArrayLike,
    config: STFTConfig | None = None,
    *,
    expected_samples: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Compute the fully specified SciPy STFT with shape ``64 x 7681``."""

    stft = config or DEFAULT_CONFIG.stft
    sample_count = (
        DEFAULT_CONFIG.data.samples_per_series
        if expected_samples is None
        else int(expected_samples)
    )
    frequencies, times, coefficients = signal.stft(
        _as_signal(series, expected_samples=sample_count), **stft.scipy_kwargs()
    )
    if tuple(coefficients.shape) != stft.raw_shape:
        raise ValueError(
            f"Unexpected STFT shape: expected {stft.raw_shape}, got {coefficients.shape}."
        )
    if frequencies.shape != (stft.raw_shape[0],) or times.shape != (stft.raw_shape[1],):
        raise ValueError("Unexpected STFT frequency or time axis shape.")
    return frequencies, times, coefficients


def stft_series_to_patches(
    series: ArrayLike,
    config: STFTConfig | None = None,
    *,
    expected_samples: int | None = None,
) -> FloatPatches:
    """Convert one signal into 120 chronological real-valued STFT patches."""

    stft = config or DEFAULT_CONFIG.stft
    _, _, coefficients = compute_stft(
        series,
        stft,
        expected_samples=expected_samples,
    )
    retained = np.real(coefficients[:, :-1]).astype(np.float32, copy=False)
    if tuple(retained.shape) != stft.retained_shape:
        raise RuntimeError(
            f"Retained STFT has shape {retained.shape}, expected {stft.retained_shape}."
        )
    patches = retained.reshape(64, 120, 64).transpose(1, 0, 2)[..., np.newaxis]
    patches = np.ascontiguousarray(patches, dtype=np.float32)
    if patches.shape != (120, 64, 64, 1):
        raise RuntimeError(f"Unexpected patch shape: {patches.shape}.")
    if not bool(np.isfinite(patches).all()):
        raise ValueError("STFT output contains non-finite values.")
    return patches


def check_reference_parity(
    series: ArrayLike,
    config: STFTConfig | None = None,
    *,
    raise_on_failure: bool = True,
) -> ParityReport:
    """Compare the explicit call with the defaults used by the reference code."""

    stft = config or DEFAULT_CONFIG.stft
    sample_count = DEFAULT_CONFIG.data.samples_per_series
    values = _as_signal(series, expected_samples=sample_count)
    explicit_frequencies, explicit_times, explicit = compute_stft(
        values,
        stft,
        expected_samples=sample_count,
    )
    implicit_frequencies, implicit_times, implicit = signal.stft(
        values,
        fs=stft.fs,
        window=stft.window,
        nperseg=stft.nperseg,
        noverlap=stft.noverlap,
    )
    explicit_patches = stft_series_to_patches(
        values,
        stft,
        expected_samples=sample_count,
    )
    implicit_real = np.real(implicit[:, :-1]).astype(np.float32, copy=False)
    implicit_patches = np.ascontiguousarray(
        implicit_real.reshape(64, 120, 64).transpose(1, 0, 2)[..., np.newaxis]
    )
    difference = np.abs(explicit_patches - implicit_patches)
    report = ParityReport(
        frequency_axis_equal=bool(np.array_equal(explicit_frequencies, implicit_frequencies)),
        time_axis_equal=bool(np.array_equal(explicit_times, implicit_times)),
        patches_equal=bool(np.array_equal(explicit_patches, implicit_patches)),
        max_abs_difference=float(difference.max(initial=np.float32(0.0))),
    )
    if explicit.shape != implicit.shape or (raise_on_failure and not report.passed):
        raise AssertionError(f"STFT reference parity failed: {report}.")
    return report


def check_dataset_parity(
    dataset: DatasetIndex,
    *,
    train_indices: Sequence[int] = (0,),
    test_indices: Sequence[int] = (0,),
    config: STFTConfig | None = None,
) -> tuple[ParityReport, ...]:
    """Run the reference parity check on a small set of source series."""

    reports: list[ParityReport] = []
    for source, indices in (
        (dataset.train, train_indices),
        (dataset.test, test_indices),
    ):
        with open_hdf5_values(source) as values:
            for raw_index in indices:
                index = int(raw_index)
                if not 0 <= index < source.n_series:
                    raise IndexError(f"Parity index {index} is outside {source.name}.")
                reports.append(check_reference_parity(values[index, :], config))
    return tuple(reports)


def _parameters(config: ExperimentConfig) -> dict[str, Any]:
    stft = asdict(config.stft)
    stft.update(
        {
            "drop_last_time_column": True,
            "complex_projection": "real",
            "output_dtype": "float32",
            "normalization": None,
        }
    )
    payload = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "stft": stft,
        "raw_shapes": {
            "train": [config.data.train_series, config.data.samples_per_series],
            "test": [config.data.test_series, config.data.samples_per_series],
        },
        "cache_shapes": {
            "train": list(config.data.train_cache_shape),
            "test": list(config.data.test_cache_shape),
        },
        "series_layout": "series-major; patches ordered chronologically",
    }
    # A JSON round trip normalizes tuples to lists before state comparisons.
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def _source_record(path: Path, data_root: Path, hdf5_key: str | None) -> dict[str, Any]:
    stat_before = path.stat()
    digest = sha256_file(path)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise CacheStateError(f"Source changed while it was being hashed: {path}.")
    try:
        portable_path = path.relative_to(data_root).as_posix()
    except ValueError:
        portable_path = path.name
    record: dict[str, Any] = {
        "relative_path": portable_path,
        "size_bytes": int(stat_after.st_size),
        "sha256": digest,
    }
    if hdf5_key is not None:
        record["hdf5_key"] = hdf5_key
    return record


def _source_records(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    data = config.data
    return {
        "train_hdf5": _source_record(
            data.train_hdf5_path, data.data_root, data.train_hdf5_key
        ),
        "test_hdf5": _source_record(
            data.test_hdf5_path, data.data_root, data.test_hdf5_key
        ),
        "labels_csv": _source_record(data.labels_path, data.data_root, None),
    }


def _split_contracts(
    dataset: DatasetIndex,
    config: ExperimentConfig,
) -> dict[str, tuple[SeriesSource, Path, tuple[int, ...]]]:
    return {
        "train": (dataset.train, config.data.train_cache_path, config.data.train_cache_shape),
        "test": (dataset.test, config.data.test_cache_path, config.data.test_cache_shape),
    }


def _new_build_state(
    dataset: DatasetIndex,
    config: ExperimentConfig,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parameters = _parameters(config)
    splits: dict[str, Any] = {}
    for name, (source, final_path, shape) in _split_contracts(dataset, config).items():
        splits[name] = {
            "final_filename": final_path.name,
            "partial_filename": _partial_path(final_path).name,
            "shape": list(shape),
            "dtype": "float32",
            "completed_series": 0,
            "ordered_series_ids_sha256": ordered_ids_sha256(source.series_ids),
            "finiteness": None,
        }
    now = _utc_now()
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": False,
        "created_at_utc": now,
        "updated_at_utc": now,
        "parameters": parameters,
        "parameters_sha256": _json_sha256(parameters),
        "sources": dict(sources),
        "splits": splits,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheStateError(f"Invalid cache build state: {path}.") from error
    if not isinstance(payload, dict):
        raise CacheStateError(f"Cache build state must be a JSON object: {path}.")
    return payload


def _assert_resume_compatible(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("schema_version", "parameters", "parameters_sha256", "sources"):
        if actual.get(key) != expected.get(key):
            raise CacheStateError(f"Partial cache has incompatible {key!r} metadata.")
    if not isinstance(actual.get("splits"), Mapping):
        raise CacheStateError("Partial cache state has no valid split records.")
    for name, expected_split in expected["splits"].items():
        split = actual["splits"].get(name)
        if not isinstance(split, Mapping):
            raise CacheStateError(f"Partial cache has no {name!r} state.")
        for key in (
            "final_filename",
            "partial_filename",
            "shape",
            "dtype",
            "ordered_series_ids_sha256",
        ):
            if split.get(key) != expected_split.get(key):
                raise CacheStateError(f"Partial {name} cache has incompatible {key!r}.")
        completed = split.get("completed_series")
        if not isinstance(completed, int) or not 0 <= completed <= expected_split["shape"][0]:
            raise CacheStateError(f"Invalid completed-series count for {name}: {completed}.")


def _write_state(path: Path, state: MutableMapping[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_write_json(path, state)


def _scan_cache(
    array: NDArray[np.floating[Any]],
    *,
    chunk_series: int,
) -> dict[str, Any]:
    if chunk_series <= 0:
        raise ValueError("chunk_series must be positive.")
    minimum = np.inf
    maximum = -np.inf
    count = 0
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), chunk_series):
        block = np.ascontiguousarray(array[start : start + chunk_series])
        finite = np.isfinite(block)
        if not bool(finite.all()):
            first = np.argwhere(~finite)[0]
            location = (start + int(first[0]), *(int(value) for value in first[1:]))
            raise ValueError(f"Non-finite cache value at index {location}.")
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
        count += int(block.size)
        digest.update(memoryview(block).cast("B"))
    return {
        "all_finite": True,
        "checked_elements": count,
        "minimum": minimum,
        "maximum": maximum,
        "values_sha256": digest.hexdigest(),
    }


def _open_partial(path: Path, shape: tuple[int, ...], create: bool) -> np.memmap:
    if create:
        return np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
            fortran_order=False,
        )
    array = np.load(path, mmap_mode="r+", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise CacheStateError(f"Partial cache is not memory-mappable: {path}.")
    if tuple(array.shape) != shape or array.dtype != np.dtype(np.float32):
        raise CacheStateError(f"Partial cache has an invalid array contract: {path}.")
    return array


def _build_one_split(
    name: str,
    source: SeriesSource,
    final_path: Path,
    shape: tuple[int, ...],
    state: MutableMapping[str, Any],
    state_path: Path,
    config: ExperimentConfig,
    *,
    checkpoint_every: int,
    finite_scan_chunk_series: int,
    progress: ProgressCallback | None,
) -> None:
    split_state = state["splits"][name]
    completed = int(split_state["completed_series"])
    partial = _partial_path(final_path)
    total = int(shape[0])
    if final_path.exists():
        if partial.exists():
            raise CacheStateError(
                f"Both final and partial {name} caches exist; refusing ambiguity."
            )
        if completed != total:
            raise CacheStateError(f"Final {name} cache exists but its state is incomplete.")
        array = np.load(final_path, mmap_mode="r", allow_pickle=False)
        split_state["finiteness"] = _scan_cache(
            array, chunk_series=finite_scan_chunk_series
        )
        return
    if completed and not partial.is_file():
        raise CacheStateError(f"Missing partial {name} cache after {completed} series.")
    partial.parent.mkdir(parents=True, exist_ok=True)
    cache = _open_partial(partial, shape, create=not partial.exists())
    with open_hdf5_values(source) as values:
        for series_index in range(completed, total):
            cache[series_index] = stft_series_to_patches(
                values[series_index, :],
                config.stft,
                expected_samples=config.data.samples_per_series,
            )
            finished = series_index + 1
            if finished % checkpoint_every == 0 or finished == total:
                cache.flush()
                split_state["completed_series"] = finished
                _write_state(state_path, state)
            if progress is not None:
                progress(name, finished, total)
    cache.flush()
    split_state["finiteness"] = _scan_cache(
        cache, chunk_series=finite_scan_chunk_series
    )
    del cache
    os.replace(partial, final_path)
    _write_state(state_path, state)


def _remove_owned_cache_files(data: DataConfig) -> None:
    """Remove only files created by this module after an explicit restart."""

    for path in (
        data.train_cache_path,
        data.test_cache_path,
        _partial_path(data.train_cache_path),
        _partial_path(data.test_cache_path),
        data.cache_manifest_path,
        _state_path(data),
    ):
        if path.exists():
            if not path.is_file():
                raise CacheStateError(f"Refusing to remove a non-file artifact: {path}.")
            path.unlink()


def build_stft_caches(
    config: ExperimentConfig | None = None,
    *,
    dataset: DatasetIndex | None = None,
    resume: bool = True,
    restart: bool = False,
    checkpoint_every: int = 8,
    finite_scan_chunk_series: int = 8,
    progress: ProgressCallback | None = None,
) -> CacheBuildResult:
    """Build or resume the two caches and commit a reproducibility manifest."""

    experiment = config or DEFAULT_CONFIG
    data = experiment.data
    if checkpoint_every <= 0 or finite_scan_chunk_series <= 0:
        raise ValueError("Checkpoint and scan chunk sizes must be positive.")
    data.cache_root.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(data)
    if restart:
        _remove_owned_cache_files(data)

    dataset = dataset or validate_dataset(experiment, check_finite=True)
    if data.cache_manifest_path.exists():
        validate_cache(
            experiment,
            dataset=dataset,
            verify_source_hashes=True,
            scan_finite=False,
        )
        return CacheBuildResult(
            data.train_cache_path,
            data.test_cache_path,
            data.cache_manifest_path,
            reused_complete_cache=True,
            resumed_partial_cache=False,
        )

    sources = _source_records(experiment)
    expected_state = _new_build_state(dataset, experiment, sources)
    resumed = state_path.exists()
    if resumed:
        if not resume:
            raise CacheStateError(
                f"A partial build exists at {state_path}; enable resume or restart."
            )
        state = _load_json_object(state_path)
        _assert_resume_compatible(state, expected_state)
    else:
        unexpected = [
            path
            for path in (
                data.train_cache_path,
                data.test_cache_path,
                _partial_path(data.train_cache_path),
                _partial_path(data.test_cache_path),
            )
            if path.exists()
        ]
        if unexpected:
            raise CacheStateError(
                "Cache artifacts exist without a compatible manifest or build state: "
                + ", ".join(str(path) for path in unexpected)
            )
        state = expected_state
        _write_state(state_path, state)

    for name, (source, final_path, shape) in _split_contracts(dataset, experiment).items():
        _build_one_split(
            name,
            source,
            final_path,
            shape,
            state,
            state_path,
            experiment,
            checkpoint_every=checkpoint_every,
            finite_scan_chunk_series=finite_scan_chunk_series,
            progress=progress,
        )

    caches: dict[str, Any] = {}
    for name, (source, final_path, shape) in _split_contracts(dataset, experiment).items():
        finiteness = state["splits"][name]["finiteness"]
        if not isinstance(finiteness, Mapping) or finiteness.get("all_finite") is not True:
            raise CacheStateError(f"Missing finite-value proof for the {name} cache.")
        caches[name] = {
            "filename": final_path.name,
            "shape": list(shape),
            "dtype": "float32",
            "order": "C",
            "ordered_series_ids_sha256": ordered_ids_sha256(source.series_ids),
            "file_size_bytes": int(final_path.stat().st_size),
            "finiteness": finiteness,
        }
    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "created_at_utc": state["created_at_utc"],
        "completed_at_utc": _utc_now(),
        "parameters": state["parameters"],
        "parameters_sha256": state["parameters_sha256"],
        "sources": state["sources"],
        "caches": caches,
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    _atomic_write_json(data.cache_manifest_path, manifest)
    if state_path.exists():
        state_path.unlink()
    validate_cache(experiment, dataset=dataset, scan_finite=False)
    return CacheBuildResult(
        data.train_cache_path,
        data.test_cache_path,
        data.cache_manifest_path,
        reused_complete_cache=False,
        resumed_partial_cache=resumed,
    )


def _validate_source_provenance(
    manifest: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    verify_all_hashes: bool,
) -> bool:
    """Check portable source metadata and always authenticate the label file."""

    records = manifest.get("sources")
    if not isinstance(records, Mapping):
        raise ValueError("Cache manifest has no valid source records.")
    expected = {
        "train_hdf5": (
            config.data.train_hdf5_path,
            config.data.train_hdf5_key,
            False,
        ),
        "test_hdf5": (
            config.data.test_hdf5_path,
            config.data.test_hdf5_key,
            False,
        ),
        "labels_csv": (config.data.labels_path, None, True),
    }
    for name, (path, hdf5_key, always_hash) in expected.items():
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Cache manifest has no {name!r} source record.")
        if record.get("relative_path") != path.name:
            raise ValueError(f"The {name} filename differs from the cache manifest.")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"The {name} size differs from the cache manifest.")
        if hdf5_key is not None and record.get("hdf5_key") != hdf5_key:
            raise ValueError(f"The {name} HDF5 key differs from the cache manifest.")
        if (verify_all_hashes or always_hash) and record.get("sha256") != sha256_file(path):
            raise ValueError(f"The {name} SHA-256 differs from the cache manifest.")
    return verify_all_hashes


def validate_cache(
    config: ExperimentConfig | None = None,
    *,
    dataset: DatasetIndex | None = None,
    verify_source_hashes: bool = False,
    scan_finite: bool = False,
    finite_scan_chunk_series: int = 8,
) -> CacheValidationReport:
    """Validate the cache manifest, array contracts, and optional file contents."""

    experiment = config or DEFAULT_CONFIG
    data = experiment.data
    manifest = load_cache_manifest(data)
    stored_digest = manifest.get("manifest_payload_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_payload_sha256", None)
    if stored_digest != _json_sha256(unhashed):
        raise ValueError("Cache manifest payload hash does not match its contents.")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Cache manifest uses an unsupported schema version.")
    parameters = _parameters(experiment)
    if manifest.get("parameters") != parameters:
        raise ValueError("Cache preprocessing parameters differ from the current configuration.")
    if manifest.get("parameters_sha256") != _json_sha256(parameters):
        raise ValueError("Cache parameter hash is invalid.")

    dataset = dataset or validate_dataset(experiment, check_finite=False)
    source_hashes_verified = _validate_source_provenance(
        manifest,
        experiment,
        verify_all_hashes=verify_source_hashes,
    )

    train = open_cache("train", data, require_manifest=False)
    test = open_cache("test", data, require_manifest=False)
    arrays = {"train": train, "test": test}
    sources = {"train": dataset.train, "test": dataset.test}
    all_finite = True
    cache_records = manifest.get("caches")
    if not isinstance(cache_records, Mapping):
        raise ValueError("Cache manifest has no valid cache records.")
    for name, array in arrays.items():
        record = cache_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Cache manifest has no {name!r} record.")
        if record.get("shape") != list(array.shape) or record.get("dtype") != "float32":
            raise ValueError(f"Manifest contract does not match the {name} cache.")
        if record.get("file_size_bytes") != int(
            (data.train_cache_path if name == "train" else data.test_cache_path).stat().st_size
        ):
            raise ValueError(f"File size does not match the {name} cache manifest.")
        if record.get("ordered_series_ids_sha256") != ordered_ids_sha256(
            sources[name].series_ids
        ):
            raise ValueError(f"Series order does not match the {name} cache manifest.")
        finite_record = record.get("finiteness")
        if not isinstance(finite_record, Mapping) or finite_record.get("all_finite") is not True:
            raise ValueError(f"No finite-value proof is recorded for the {name} cache.")
        recorded_minimum = float(finite_record.get("minimum", np.nan))
        recorded_maximum = float(finite_record.get("maximum", np.nan))
        recorded_values_sha256 = finite_record.get("values_sha256")
        if not (
            np.isfinite(recorded_minimum)
            and np.isfinite(recorded_maximum)
            and recorded_minimum < 0.0 < recorded_maximum
            and isinstance(recorded_values_sha256, str)
            and len(recorded_values_sha256) == 64
        ):
            raise ValueError(
                f"The {name} cache is not a finite signed real-STFT representation."
            )
        if scan_finite:
            current_scan = _scan_cache(
                array, chunk_series=finite_scan_chunk_series
            )
            if current_scan["values_sha256"] != recorded_values_sha256:
                raise ValueError(f"The {name} cache content hash has changed.")
            all_finite = all_finite and bool(current_scan["all_finite"])
    return CacheValidationReport(
        train_shape=tuple(train.shape),
        test_shape=tuple(test.shape),
        dtype="float32",
        source_hashes_verified=source_hashes_verified,
        finite_values_scanned=scan_finite,
        all_finite=all_finite,
        manifest=manifest,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "PREPROCESSING_VERSION",
    "CacheBuildResult",
    "CacheStateError",
    "CacheValidationReport",
    "ParityReport",
    "build_stft_caches",
    "check_dataset_parity",
    "check_reference_parity",
    "compute_stft",
    "stft_series_to_patches",
    "validate_cache",
]
