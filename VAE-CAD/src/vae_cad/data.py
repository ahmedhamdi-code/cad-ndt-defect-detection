"""Validated, bounded-memory access to AIRBUS signals and STFT caches.

The HDF5 files were produced by pandas, but their fixed-format layout is read
directly with :mod:`h5py`. This avoids a PyTables dependency. Large arrays are
scanned in chunks, and caches are opened as read-only NumPy memory maps.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import uuid

import h5py
import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, DataConfig, ExperimentConfig, SplitConfig


IntArray = NDArray[np.int64]
LabelArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SeriesSource:
    """Validated description of one signal matrix in an HDF5 file."""

    name: str
    path: Path
    hdf5_key: str
    series_ids: IntArray
    n_series: int
    samples_per_series: int
    finite_checked: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_series, self.samples_per_series)

    @property
    def values_path(self) -> str:
        return f"{self.hdf5_key}/block0_values"


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """Validated train/test sources and labels in official row order."""

    train: SeriesSource
    test: SeriesSource
    test_ids: IntArray
    test_labels: LabelArray
    config: DataConfig

    @property
    def label_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.test_labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}


@dataclass(frozen=True, slots=True)
class SeriesSplit:
    """A reproducible partition of complete normal training series."""

    seed: int
    train_indices: IntArray
    validation_indices: IntArray
    calibration_indices: IntArray

    @property
    def ae_train_indices(self) -> IntArray:
        """Alias retained for explicit autoencoder/VAE terminology."""

        return self.train_indices

    @property
    def ae_validation_indices(self) -> IntArray:
        """Alias retained for explicit autoencoder/VAE terminology."""

        return self.validation_indices

    @property
    def cad_calibration_indices(self) -> IntArray:
        """Alias retained for explicit CAD terminology."""

        return self.calibration_indices

    @property
    def all_indices(self) -> IntArray:
        return np.concatenate(
            (self.train_indices, self.validation_indices, self.calibration_indices)
        )

    def as_dict(self, *, copy: bool = True) -> dict[str, IntArray]:
        values = {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "calibration": self.calibration_indices,
        }
        return {key: value.copy() for key, value in values.items()} if copy else values


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _data_config(
    config: DataConfig | ExperimentConfig | str | Path | None,
) -> DataConfig:
    if config is None:
        return DEFAULT_CONFIG.data
    if isinstance(config, ExperimentConfig):
        return config.data
    if isinstance(config, DataConfig):
        return config
    return DataConfig(data_root=Path(config), cache_root=DEFAULT_CONFIG.data.cache_root)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it completely into memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(cache: np.ndarray) -> dict[str, Any]:
    """Identify a cache without repeatedly hashing a multi-gigabyte memmap.

    Production caches are bound by resolved path, byte size and nanosecond
    modification time. Generic in-memory arrays are hashed completely.
    """

    identity: dict[str, Any] = {
        "shape": [int(value) for value in cache.shape],
        "dtype": str(cache.dtype),
    }
    filename = getattr(cache, "filename", None)
    if filename is not None:
        cache_path = Path(filename).expanduser().resolve()
        stat = cache_path.stat()
        identity.update(
            {
                "storage": "file",
                "path": str(cache_path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
        return identity

    digest = hashlib.sha256()
    for start in range(int(cache.shape[0])):
        block = np.ascontiguousarray(cache[start : start + 1])
        digest.update(memoryview(block).cast("B"))
    identity.update({"storage": "memory", "data_sha256": digest.hexdigest()})
    return identity


def ordered_ids_sha256(series_ids: Sequence[int] | IntArray) -> str:
    """Hash ordered IDs using a platform-independent integer encoding."""

    values = np.asarray(series_ids, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("series_ids must be one-dimensional.")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _scan_hdf5_finiteness(dataset: h5py.Dataset, chunk_series: int) -> None:
    if chunk_series <= 0:
        raise ValueError("chunk_series must be positive.")
    for start in range(0, int(dataset.shape[0]), chunk_series):
        stop = min(start + chunk_series, int(dataset.shape[0]))
        finite = np.isfinite(np.asarray(dataset[start:stop]))
        if not bool(finite.all()):
            first = np.argwhere(~finite)[0]
            location = (start + int(first[0]), int(first[1]))
            raise ValueError(f"Non-finite source value at row/column {location}.")


def _validate_hdf5_source(
    name: str,
    path: Path,
    key: str,
    expected_shape: tuple[int, int],
    *,
    check_finite: bool,
    chunk_series: int,
) -> SeriesSource:
    _require_file(path, f"{name} HDF5 file")
    with h5py.File(path, mode="r") as handle:
        if key not in handle or not isinstance(handle[key], h5py.Group):
            raise KeyError(f"Missing HDF5 group /{key} in {path}.")
        group = handle[key]
        required = ("axis0", "axis1", "block0_items", "block0_values")
        missing = [entry for entry in required if entry not in group]
        if missing:
            raise KeyError(f"Missing entries in /{key}: {', '.join(missing)}.")

        values = group["block0_values"]
        if not isinstance(values, h5py.Dataset):
            raise TypeError(f"/{key}/block0_values is not an HDF5 dataset.")
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape: expected {expected_shape}, "
                f"got {tuple(values.shape)}."
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"{name} signals must be floating point, got {values.dtype}.")

        sample_axis = np.asarray(group["axis0"][:], dtype=np.int64)
        block_items = np.asarray(group["block0_items"][:], dtype=np.int64)
        series_ids = np.asarray(group["axis1"][:], dtype=np.int64)
        expected_samples = np.arange(expected_shape[1], dtype=np.int64)
        expected_ids = np.arange(expected_shape[0], dtype=np.int64)
        if not np.array_equal(sample_axis, expected_samples):
            raise ValueError(
                f"/{key}/axis0 is not ordered from 0 to {expected_shape[1] - 1}."
            )
        if not np.array_equal(block_items, sample_axis):
            raise ValueError(f"/{key}/block0_items does not match axis0.")
        if not np.array_equal(series_ids, expected_ids):
            raise ValueError(
                f"/{key}/axis1 is not ordered from 0 to {expected_shape[0] - 1}."
            )
        if check_finite:
            _scan_hdf5_finiteness(values, chunk_series)

    return SeriesSource(
        name=name,
        path=path,
        hdf5_key=key,
        series_ids=_readonly(series_ids),
        n_series=expected_shape[0],
        samples_per_series=expected_shape[1],
        finite_checked=check_finite,
    )


def load_ground_truth(
    config: DataConfig | ExperimentConfig | str | Path | None = None,
) -> tuple[IntArray, LabelArray]:
    """Load and validate the ordered ``seqID,anomaly`` CSV without pandas."""

    data = _data_config(config)
    _require_file(data.labels_path, "ground-truth CSV")
    identifiers: list[int] = []
    labels: list[int] = []
    with data.labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["seqID", "anomaly"]:
            raise ValueError(
                "Ground-truth header must be exactly 'seqID,anomaly'; "
                f"got {header}."
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Ground-truth row {line_number} must have two fields.")
            try:
                identifier = int(row[0])
                raw_label = float(row[1])
            except ValueError as error:
                raise ValueError(f"Invalid ground-truth row {line_number}: {row}.") from error
            if not np.isfinite(raw_label) or raw_label not in (0.0, 1.0):
                raise ValueError(f"Label on row {line_number} is not binary: {row[1]!r}.")
            identifiers.append(identifier)
            labels.append(int(raw_label))

    ids = np.asarray(identifiers, dtype=np.int64)
    values = np.asarray(labels, dtype=np.uint8)
    expected_ids = np.arange(data.test_series, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(
            f"Ground-truth IDs must contain 0..{data.test_series - 1} in row order."
        )
    counts = np.bincount(values, minlength=2)
    expected_counts = (data.normal_test_series, data.anomalous_test_series)
    if tuple(int(value) for value in counts) != expected_counts:
        raise ValueError(
            f"Expected normal/anomaly counts {expected_counts}, got {tuple(counts)}."
        )
    return _readonly(ids), _readonly(values)


def validate_dataset(
    config: DataConfig | ExperimentConfig | str | Path | None = None,
    *,
    check_finite: bool = True,
    chunk_series: int = 8,
) -> DatasetIndex:
    """Validate raw shapes, ordering, labels, and optionally every value."""

    data = _data_config(config)
    train = _validate_hdf5_source(
        "training",
        data.train_hdf5_path,
        data.train_hdf5_key,
        (data.train_series, data.samples_per_series),
        check_finite=check_finite,
        chunk_series=chunk_series,
    )
    test = _validate_hdf5_source(
        "test",
        data.test_hdf5_path,
        data.test_hdf5_key,
        (data.test_series, data.samples_per_series),
        check_finite=check_finite,
        chunk_series=chunk_series,
    )
    test_ids, labels = load_ground_truth(data)
    if not np.array_equal(test.series_ids, test_ids):
        raise ValueError("HDF5 test rows and ground-truth IDs are not aligned.")
    return DatasetIndex(train, test, test_ids, labels, data)


@contextmanager
def open_hdf5_values(source: SeriesSource) -> Iterator[h5py.Dataset]:
    """Yield a signal matrix from an HDF5 file opened read-only."""

    with h5py.File(source.path, mode="r") as handle:
        if source.values_path not in handle:
            raise KeyError(f"Missing HDF5 dataset /{source.values_path}.")
        values = handle[source.values_path]
        if tuple(values.shape) != source.shape:
            raise ValueError(
                f"Source shape changed: expected {source.shape}, got {tuple(values.shape)}."
            )
        yield values


def iter_raw_series(
    source: SeriesSource,
    indices: Sequence[int] | NDArray[np.integer] | None = None,
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    """Yield ``(row_position, series_id, signal)`` from one read-only handle."""

    positions = (
        np.arange(source.n_series, dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    if positions.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if positions.size and (
        int(positions.min()) < 0 or int(positions.max()) >= source.n_series
    ):
        raise IndexError(f"Series position outside [0, {source.n_series}).")
    with open_hdf5_values(source) as values:
        for raw_position in positions:
            position = int(raw_position)
            signal = np.asarray(values[position, :], dtype=np.float64)
            if signal.shape != (source.samples_per_series,):
                raise ValueError(f"Malformed series at row {position}: {signal.shape}.")
            if not bool(np.isfinite(signal).all()):
                raise ValueError(f"Non-finite values in {source.name} series {position}.")
            yield position, int(source.series_ids[position]), signal


def make_series_split(
    seed: int,
    series_ids: Sequence[int] | IntArray | None = None,
    config: SplitConfig | ExperimentConfig | None = None,
) -> SeriesSplit:
    """Partition at series level before any series is expanded into patches."""

    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer.")
    split_config = (
        DEFAULT_CONFIG.split
        if config is None
        else config.split
        if isinstance(config, ExperimentConfig)
        else config
    )
    total = split_config.total
    ids = (
        np.arange(total, dtype=np.int64)
        if series_ids is None
        else np.asarray(series_ids, dtype=np.int64)
    )
    if ids.shape != (total,) or np.unique(ids).size != ids.size:
        raise ValueError(f"Expected exactly {total} unique ordered series IDs.")
    permutation = np.random.default_rng(int(seed)).permutation(total).astype(np.int64)
    first = split_config.train_series
    second = first + split_config.validation_series
    split = SeriesSplit(
        seed=int(seed),
        train_indices=_readonly(permutation[:first].copy()),
        validation_indices=_readonly(permutation[first:second].copy()),
        calibration_indices=_readonly(permutation[second:].copy()),
    )
    if np.unique(split.all_indices).size != total:
        raise RuntimeError("Generated series split is not a complete partition.")
    return split


def write_split_manifest(
    path: str | Path,
    split: SeriesSplit,
    series_ids: Sequence[int] | IntArray,
) -> None:
    """Atomically write auditable series assignments."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(series_ids, dtype=np.int64)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
        raise ValueError("series_ids must be a one-dimensional unique array.")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("seed", "partition", "series_position", "series_id"))
            for partition, positions in split.as_dict(copy=False).items():
                for raw_position in positions:
                    position = int(raw_position)
                    writer.writerow((split.seed, partition, position, int(ids[position])))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_cache_manifest(
    config: DataConfig | ExperimentConfig | None = None,
) -> Mapping[str, Any]:
    """Read the cache commit marker and reject incomplete builds."""

    data = _data_config(config)
    if not data.cache_manifest_path.is_file():
        raise FileNotFoundError(
            f"Cache manifest not found: {data.cache_manifest_path}. "
            "Run prepare_data.py first."
        )
    try:
        payload = json.loads(data.cache_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid cache manifest: {data.cache_manifest_path}.") from error
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(
            f"Cache manifest is not a completed commit: {data.cache_manifest_path}."
        )
    return payload


def open_cache(
    split: str,
    config: DataConfig | ExperimentConfig | None = None,
    *,
    require_manifest: bool = True,
) -> np.memmap:
    """Open and validate a final STFT cache in read-only mode."""

    data = _data_config(config)
    normalized = split.lower()
    if normalized in {"train", "training"}:
        path, expected_shape = data.train_cache_path, data.train_cache_shape
    elif normalized in {"test", "validation", "valid"}:
        path, expected_shape = data.test_cache_path, data.test_cache_shape
    else:
        raise ValueError("split must be 'train' or 'test'.")
    if require_manifest:
        load_cache_manifest(data)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {normalized} cache: {path}.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.flags.writeable:
        raise TypeError(f"Cache must be a read-only NumPy memory map: {path}.")
    if tuple(array.shape) != expected_shape or array.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Unexpected cache contract for {path}: {array.shape}, {array.dtype}; "
            f"expected {expected_shape}, float32."
        )
    return array


def iter_patch_batches(
    cache: NDArray[np.floating[Any]],
    series_indices: Sequence[int] | NDArray[np.integer],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
    epoch: int = 0,
    normalization: tuple[float, float] | None = None,
    autoencoder_targets: bool = False,
) -> Iterator[FloatArray | tuple[FloatArray, FloatArray]]:
    """Yield bounded-memory patch batches from selected complete series."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    expected_tail = (
        DEFAULT_CONFIG.data.patches_per_series,
        *DEFAULT_CONFIG.data.patch_shape,
    )
    if cache.ndim != 5 or tuple(cache.shape[1:]) != expected_tail:
        raise ValueError(f"Cache must have shape (series, {expected_tail}); got {cache.shape}.")
    if cache.dtype != np.dtype(np.float32):
        raise TypeError(f"Cache dtype must be float32, got {cache.dtype}.")
    selected = np.asarray(series_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("series_indices must be a non-empty one-dimensional array.")
    if int(selected.min()) < 0 or int(selected.max()) >= int(cache.shape[0]):
        raise IndexError("series_indices contain a position outside the cache.")
    if np.unique(selected).size != selected.size:
        raise ValueError("series_indices must not contain duplicates.")

    mean, standard_deviation = (0.0, 1.0) if normalization is None else normalization
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Normalization statistics must be finite with a positive std.")
    total = selected.size * expected_tail[0]
    order = np.arange(total, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
        rng.shuffle(order)
    for start in range(0, total, batch_size):
        flat = order[start : start + batch_size]
        series_positions = selected[flat // expected_tail[0]]
        patch_positions = flat % expected_tail[0]
        batch = np.asarray(cache[series_positions, patch_positions], dtype=np.float32)
        if normalization is not None:
            batch = ((batch - np.float32(mean)) / np.float32(standard_deviation)).astype(
                np.float32, copy=False
            )
        yield (batch, batch) if autoencoder_targets else batch


def count_patch_batches(n_series: int, batch_size: int) -> int:
    """Return the number of non-dropping batches for complete series."""

    if n_series < 0 or batch_size <= 0:
        raise ValueError("n_series must be non-negative and batch_size positive.")
    total = n_series * DEFAULT_CONFIG.data.patches_per_series
    return (total + batch_size - 1) // batch_size


__all__ = [
    "DatasetIndex",
    "SeriesSource",
    "SeriesSplit",
    "cache_identity",
    "count_patch_batches",
    "iter_patch_batches",
    "iter_raw_series",
    "load_cache_manifest",
    "load_ground_truth",
    "make_series_split",
    "open_cache",
    "open_hdf5_values",
    "ordered_ids_sha256",
    "sha256_file",
    "validate_dataset",
    "write_split_manifest",
]
