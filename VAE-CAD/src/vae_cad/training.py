"""Bounded-memory, resumable training utilities for the AIRBUS VAE.

Splits are expressed as complete series indices.  They are expanded into
patches only by the batch iterator, preventing patches from one series from
crossing the train, validation, and conformal-calibration partitions.
Normalization is estimated from normal training series only and is then frozen.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import uuid

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import ExperimentConfig, VAEConfig, source_tree_sha256
from .data import SeriesSplit, cache_identity, count_patch_batches, iter_patch_batches
from .model import (
    BetaWarmup,
    ConvolutionalVAE,
    build_vae,
    configure_device,
    effective_log_variance,
    load_vae,
    model_summary_text,
    save_vae,
    set_global_seed,
    validate_vae,
)


HISTORY_COLUMNS = (
    "epoch",
    "learning_rate",
    "beta",
    "train_reconstruction_loss",
    "train_kl_loss",
    "train_total_loss",
    "validation_reconstruction_loss",
    "validation_kl_loss",
    "validation_total_loss",
)


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Scalar normalization estimated from normal training patches only."""

    mean: float
    std: float
    count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("Normalization mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("Normalization standard deviation must be positive")
        if self.count <= 1:
            raise ValueError("Normalization count must exceed one")

    @property
    def pair(self) -> tuple[float, float]:
        return self.mean, self.std


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and diagnostics committed by a completed training run."""

    output_dir: Path
    model_path: Path
    history_path: Path
    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    normalization: NormalizationStats
    latent_diagnostics: Mapping[str, Any]
    status: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _json_round_trip(payload: Any) -> Any:
    """Normalize tuples and NumPy-free values exactly as JSON storage does."""

    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _indices_sha256(indices: np.ndarray) -> str:
    """Hash an ordered one-dimensional series-index array portably."""

    canonical = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def training_input_contract(
    cache: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    """Bind reusable checkpoints to their cache and ordered partitions."""

    return {
        "source_tree_sha256": source_tree_sha256(),
        "cache": cache_identity(cache),
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "training_series": int(train_indices.size),
        "validation_series": int(validation_indices.size),
    }


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite {name}: {result!r}")
    return result


def compute_normalization(
    cache: np.ndarray,
    series_indices: Sequence[int] | np.ndarray,
    *,
    chunk_series: int = 8,
) -> NormalizationStats:
    """Compute scalar population moments in streaming float64 arithmetic."""

    if cache.ndim != 5 or cache.dtype != np.dtype(np.float32):
        raise ValueError("cache must be a five-dimensional float32 array")
    indices = np.asarray(series_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("series_indices must be a non-empty one-dimensional array")
    if np.unique(indices).size != indices.size:
        raise ValueError("series_indices must not contain duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= int(cache.shape[0]):
        raise IndexError("series_indices contain an out-of-range value")
    if chunk_series <= 0:
        raise ValueError("chunk_series must be positive")

    count = 0
    running_mean = 0.0
    running_m2 = 0.0
    for start in range(0, indices.size, chunk_series):
        selected = indices[start : start + chunk_series]
        values = np.asarray(cache[selected], dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise FloatingPointError("Non-finite value in normal training data")
        chunk_count = int(values.size)
        chunk_mean = float(values.mean(dtype=np.float64))
        centered = values - chunk_mean
        chunk_m2 = float(np.square(centered).sum(dtype=np.float64))
        if count == 0:
            count = chunk_count
            running_mean = chunk_mean
            running_m2 = chunk_m2
            continue
        total = count + chunk_count
        delta = chunk_mean - running_mean
        running_m2 += chunk_m2 + delta * delta * count * chunk_count / total
        running_mean += delta * chunk_count / total
        count = total

    variance = running_m2 / count
    if not math.isfinite(variance) or variance <= 0.0:
        raise FloatingPointError("Training variance is not finite and positive")
    return NormalizationStats(running_mean, math.sqrt(variance), count)


def save_normalization(path: str | os.PathLike[str], stats: NormalizationStats) -> None:
    """Atomically save normalization parameters as a NumPy archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(
            temporary,
            mean=np.asarray(stats.mean, dtype=np.float64),
            std=np.asarray(stats.std, dtype=np.float64),
            count=np.asarray(stats.count, dtype=np.int64),
            scope=np.asarray("normal_training_series_only"),
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_normalization(path: str | os.PathLike[str]) -> NormalizationStats:
    """Load and validate frozen normalization parameters."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"mean", "std", "count"}
        if not required.issubset(archive.files):
            raise ValueError(f"Normalization archive is missing {required - set(archive.files)}")
        return NormalizationStats(
            mean=float(archive["mean"]),
            std=float(archive["std"]),
            count=int(archive["count"]),
        )


def _dataset(
    cache: np.ndarray,
    series_indices: np.ndarray,
    *,
    normalization: NormalizationStats,
    batch_size: int,
    seed: int,
    shuffle: bool,
    start_epoch: int,
    input_shape: tuple[int, int, int],
) -> tf.data.Dataset:
    """Create a bounded-memory deterministic stream of complete epochs."""

    signature = tf.TensorSpec((None, *input_shape), dtype=tf.float32)

    def generate() -> Iterator[np.ndarray]:
        epoch = int(start_epoch)
        while True:
            yield from iter_patch_batches(
                cache,
                series_indices,
                batch_size,
                shuffle=shuffle,
                seed=seed,
                epoch=epoch if shuffle else 0,
                normalization=normalization.pair,
                autoencoder_targets=False,
            )
            epoch += 1

    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return dataset.with_options(options).prefetch(1)


def _optimizer_learning_rate(model: keras.Model) -> float:
    if not hasattr(model, "optimizer"):
        raise RuntimeError("Model is not compiled")
    value = model.optimizer.learning_rate
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _finite_float(value, "learning rate")


class _HyperparameterLogger(keras.callbacks.Callback):
    """Record the learning rate and beta actually used during each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = math.nan

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del epoch, logs
        self.learning_rate = _optimizer_learning_rate(self.model)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del epoch
        if logs is None or not isinstance(self.model, ConvolutionalVAE):
            raise RuntimeError("Epoch logs are unavailable")
        logs["learning_rate_used"] = self.learning_rate
        logs["beta"] = float(self.model.beta.numpy())


class _FiniteMetrics(keras.callbacks.Callback):
    """Stop immediately if any training or validation component is non-finite."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        names = (
            "total_loss",
            "reconstruction_loss",
            "kl_loss",
            "val_total_loss",
            "val_reconstruction_loss",
            "val_kl_loss",
        )
        if logs is None:
            raise FloatingPointError(f"Missing metrics after epoch {epoch + 1}")
        for name in names:
            if name not in logs:
                raise RuntimeError(f"Missing metric {name} after epoch {epoch + 1}")
            _finite_float(logs[name], f"epoch {epoch + 1} {name}")


class _ResumeReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    def __init__(self, resume_state: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        self.resume_state = dict(resume_state or {})
        super().__init__(**kwargs)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        super().on_train_begin(logs)
        if self.resume_state:
            self.best = float(self.resume_state["best"])
            self.wait = int(self.resume_state["wait"])
            self.cooldown_counter = int(self.resume_state["cooldown_counter"])

    def export_state(self) -> dict[str, Any]:
        return {
            "best": float(self.best),
            "wait": int(self.wait),
            "cooldown_counter": int(self.cooldown_counter),
        }


class _ResumeEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, resume_state: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        self.resume_state = dict(resume_state or {})
        super().__init__(**kwargs)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        super().on_train_begin(logs)
        if self.resume_state:
            raw_best = self.resume_state.get("best")
            self.best = None if raw_best is None else float(raw_best)
            self.wait = int(self.resume_state["wait"])
            self.stopped_epoch = int(self.resume_state["stopped_epoch"])
            self.best_epoch = int(self.resume_state["best_epoch"])

    def export_state(self) -> dict[str, Any]:
        return {
            "best": None if self.best is None else float(self.best),
            "wait": int(self.wait),
            "stopped_epoch": int(self.stopped_epoch),
            "best_epoch": int(self.best_epoch),
        }


class _BestWeights(keras.callbacks.Callback):
    """Retain each improved validation checkpoint under a versioned name."""

    def __init__(
        self,
        output_dir: Path,
        *,
        start_epoch: int,
        best: float = math.inf,
        best_epoch: int = 0,
        path: Path | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.path = path
        self.start_epoch = int(start_epoch)
        self.best = float(best)
        self.best_epoch = int(best_epoch)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if epoch < self.start_epoch:
            return
        if logs is None or "val_total_loss" not in logs:
            raise RuntimeError("Best checkpoint requires val_total_loss")
        current = _finite_float(logs["val_total_loss"], "val_total_loss")
        if current >= self.best:
            return
        destination = self.output_dir / f".best_epoch_{epoch + 1:04d}.weights.h5"
        temporary = self.output_dir / (
            f".best_epoch_{epoch + 1:04d}.{uuid.uuid4().hex}.weights.h5"
        )
        try:
            self.model.save_weights(temporary, overwrite=True)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.path = destination
        self.best = current
        self.best_epoch = int(epoch) + 1

    def export_state(self) -> dict[str, Any]:
        return {
            "best": None if not math.isfinite(self.best) else self.best,
            "best_epoch": self.best_epoch,
            "path": None if self.path is None else self.path.name,
            "sha256": (
                _sha256_file(self.path)
                if self.path is not None and self.path.is_file()
                else None
            ),
        }


def _resume_model_name(epoch: int) -> str:
    return f".resume_epoch_{int(epoch):04d}.keras"


def _resume_history_name(epoch: int) -> str:
    return f".history_epoch_{int(epoch):04d}.csv"


def _best_weights_name(epoch: int) -> str:
    return f".best_epoch_{int(epoch):04d}.weights.h5"


def _resume_artifacts(output_dir: Path) -> set[Path]:
    """Return only private, versioned artifacts owned by this trainer."""

    paths: set[Path] = set()
    for pattern in (
        ".resume_epoch_*.keras",
        ".history_epoch_*.csv",
        ".best_epoch_*.weights.h5",
        "..history_epoch_*.csv.*.tmp",
        "..training_state.json.*.tmp",
    ):
        paths.update(path for path in output_dir.glob(pattern) if path.is_file())
    return paths


def _cleanup_resume_artifacts(output_dir: Path, *, keep: set[str]) -> None:
    """Remove orphan trainer-owned versions while retaining committed names."""

    for path in _resume_artifacts(output_dir):
        if path.name not in keep:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _history_record(epoch: int, logs: Mapping[str, Any]) -> dict[str, Any]:
    source_names = {
        "learning_rate": "learning_rate_used",
        "beta": "beta",
        "train_reconstruction_loss": "reconstruction_loss",
        "train_kl_loss": "kl_loss",
        "train_total_loss": "total_loss",
        "validation_reconstruction_loss": "val_reconstruction_loss",
        "validation_kl_loss": "val_kl_loss",
        "validation_total_loss": "val_total_loss",
    }
    missing = [source for source in source_names.values() if source not in logs]
    if missing:
        raise RuntimeError(f"Incomplete epoch logs: {missing}")
    record: dict[str, Any] = {"epoch": int(epoch) + 1}
    for destination, source in source_names.items():
        record[destination] = _finite_float(logs[source], source)
    if record["train_kl_loss"] < -1e-4 or record["validation_kl_loss"] < -1e-4:
        raise FloatingPointError("KL loss became negative")
    return record


class _EpochState(keras.callbacks.Callback):
    """Commit a versioned model/history transaction at each epoch boundary."""

    def __init__(
        self,
        output_dir: Path,
        *,
        history: list[dict[str, Any]],
        seed: int,
        config: VAEConfig,
        train_steps: int,
        training_contract: Mapping[str, Any],
        device: str,
        reduce_lr: _ResumeReduceLROnPlateau,
        early_stopping: _ResumeEarlyStopping,
        best_weights: _BestWeights,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.history = history
        self.seed = int(seed)
        self.config = config
        self.train_steps = int(train_steps)
        self.training_contract = _json_round_trip(training_contract)
        self.device = str(device)
        self.reduce_lr = reduce_lr
        self.early_stopping = early_stopping
        self.best_weights = best_weights

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            raise RuntimeError("Cannot checkpoint without epoch logs")
        if len(self.history) != epoch:
            raise RuntimeError("Resume history is not contiguous")
        completed_epoch = int(epoch) + 1
        candidate_history = [*self.history, _history_record(epoch, logs)]
        history_name = _resume_history_name(completed_epoch)
        history_path = self.output_dir / history_name
        _atomic_write_csv(
            history_path,
            candidate_history,
            HISTORY_COLUMNS,
        )

        checkpoint_name = _resume_model_name(completed_epoch)
        destination = self.output_dir / checkpoint_name
        temporary = self.output_dir / (
            f".resume_epoch_{completed_epoch:04d}.{uuid.uuid4().hex}.tmp.keras"
        )
        try:
            self.model.save(temporary, overwrite=True)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        state = {
            "schema_version": 3,
            "seed": self.seed,
            "vae_config": asdict(self.config),
            "training_contract": self.training_contract,
            "device": self.device,
            "completed_epochs": completed_epoch,
            "train_steps": self.train_steps,
            "optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
            "learning_rate": _optimizer_learning_rate(self.model),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": _sha256_file(destination),
            "history": history_name,
            "history_sha256": _sha256_file(history_path),
            "training_sampling": "stateless_seeded_by_optimizer_iteration",
            "reduce_lr": self.reduce_lr.export_state(),
            "early_stopping": self.early_stopping.export_state(),
            "best_weights": self.best_weights.export_state(),
            "updated_at_utc": _utc_now(),
        }
        # This replacement is the transaction commit point.  Files from an
        # interrupted, uncommitted epoch are ignored on the next invocation.
        _atomic_write_json(self.output_dir / ".training_state.json", state)
        self.history[:] = candidate_history
        keep = {checkpoint_name, history_name}
        best_name = state["best_weights"].get("path")
        if isinstance(best_name, str):
            keep.add(best_name)
        _cleanup_resume_artifacts(self.output_dir, keep=keep)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
            raise ValueError("Partial history has unexpected columns")
        for expected_epoch, raw in enumerate(reader, start=1):
            if int(raw["epoch"]) != expected_epoch:
                raise ValueError("Partial history epochs are not contiguous")
            row: dict[str, Any] = {"epoch": expected_epoch}
            for name in HISTORY_COLUMNS[1:]:
                row[name] = _finite_float(raw[name], name)
            rows.append(row)
    return rows


def _load_resume(
    output_dir: Path,
    *,
    seed: int,
    config: VAEConfig,
    train_steps: int,
    training_contract: Mapping[str, Any],
    resume: bool,
    device: str,
) -> tuple[ConvolutionalVAE | None, list[dict[str, Any]], dict[str, Any]]:
    state_path = output_dir / ".training_state.json"
    if not state_path.is_file():
        orphaned = _resume_artifacts(output_dir)
        if orphaned and not resume:
            raise FileExistsError(
                "Uncommitted resume artifacts exist; enable resume to discard them"
            )
        if resume:
            _cleanup_resume_artifacts(output_dir, keep=set())
        return None, [], {}
    if not resume:
        raise FileExistsError(
            "A committed resume state exists; enable resume or use another output directory"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_config = _json_round_trip(asdict(config))
    expected_contract = _json_round_trip(training_contract)
    completed = state.get("completed_epochs")
    if (
        state.get("schema_version") != 3
        or state.get("seed") != int(seed)
        or state.get("vae_config") != expected_config
        or state.get("training_contract") != expected_contract
        or state.get("device") != device
        or state.get("train_steps") != int(train_steps)
        or not isinstance(completed, int)
        or completed <= 0
        or completed > config.max_epochs
    ):
        raise RuntimeError("Resume state differs from the requested training protocol")

    checkpoint_name = state.get("checkpoint")
    history_name = state.get("history")
    if checkpoint_name != _resume_model_name(completed):
        raise RuntimeError("Resume checkpoint name is not canonical")
    if history_name != _resume_history_name(completed):
        raise RuntimeError("Resume history name is not canonical")
    model_path = output_dir / checkpoint_name
    history_path = output_dir / history_name
    if not model_path.is_file() or state.get("checkpoint_sha256") != _sha256_file(
        model_path
    ):
        raise RuntimeError("Committed resume checkpoint is missing or changed")
    if not history_path.is_file() or state.get("history_sha256") != _sha256_file(
        history_path
    ):
        raise RuntimeError("Committed resume history is missing or changed")
    history = _read_history(history_path)
    if len(history) != completed:
        raise RuntimeError("Resume state and partial history disagree")

    best_state = state.get("best_weights")
    if not isinstance(best_state, dict):
        raise RuntimeError("Resume state has no valid best-weights record")
    best_name = best_state.get("path")
    best_sha256 = best_state.get("sha256")
    best_epoch = int(best_state.get("best_epoch", 0))
    keep = {checkpoint_name, history_name}
    if best_name is None:
        if best_sha256 is not None or best_epoch != 0:
            raise RuntimeError("Empty best-weights state is inconsistent")
    else:
        if best_name != _best_weights_name(best_epoch):
            raise RuntimeError("Best-weights checkpoint name is not canonical")
        best_path = output_dir / best_name
        if not best_path.is_file() or best_sha256 != _sha256_file(best_path):
            raise RuntimeError("Committed best-weights checkpoint is missing or changed")
        keep.add(best_name)

    # Orphan versions can only be products of an epoch that did not reach the
    # atomic state commit.  They are never selected over the committed state.
    _cleanup_resume_artifacts(output_dir, keep=keep)
    vae = load_vae(model_path, compile_model=True, device=device)
    expected_iterations = len(history) * train_steps
    if int(vae.optimizer.iterations.numpy()) != expected_iterations:
        raise RuntimeError("Optimizer state does not match the completed epochs")
    if vae.sampling_seed != int(seed):
        raise RuntimeError("Training sampling seed differs from the requested seed")
    if state.get("training_sampling") != "stateless_seeded_by_optimizer_iteration":
        raise RuntimeError("Resume state uses an unsupported sampling strategy")
    return vae, history, state


def _latent_diagnostics(
    vae: ConvolutionalVAE,
    cache: np.ndarray,
    series_indices: np.ndarray,
    normalization: NormalizationStats,
    batch_size: int,
    *,
    active_variance_threshold: float = 1e-2,
    active_kl_threshold: float = 1e-2,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute validation latent statistics without storing patch embeddings."""

    latent_dim = int(vae.encoder.output_shape[0][-1])
    count = 0
    sum_mean = np.zeros(latent_dim, dtype=np.float64)
    sum_squared_mean = np.zeros(latent_dim, dtype=np.float64)
    sum_kl = np.zeros(latent_dim, dtype=np.float64)
    for batch in iter_patch_batches(
        cache,
        series_indices,
        batch_size,
        shuffle=False,
        normalization=normalization.pair,
    ):
        z_mean_tensor, log_var_tensor, _ = vae.encoder(batch, training=False)
        z_mean = np.asarray(z_mean_tensor.numpy(), dtype=np.float64)
        safe_log_var = np.asarray(
            effective_log_variance(log_var_tensor, vae.log_var_cap).numpy(),
            dtype=np.float64,
        )
        kl = -0.5 * (1.0 + safe_log_var - np.square(z_mean) - np.exp(safe_log_var))
        if not np.isfinite(z_mean).all() or not np.isfinite(kl).all():
            raise FloatingPointError("Non-finite latent diagnostic")
        count += z_mean.shape[0]
        sum_mean += z_mean.sum(axis=0, dtype=np.float64)
        sum_squared_mean += np.square(z_mean).sum(axis=0, dtype=np.float64)
        sum_kl += kl.sum(axis=0, dtype=np.float64)
    means = sum_mean / count
    variances = np.maximum(sum_squared_mean / count - np.square(means), 0.0)
    mean_kl = sum_kl / count
    active_variance = int(np.count_nonzero(variances > active_variance_threshold))
    active_kl = int(np.count_nonzero(mean_kl > active_kl_threshold))
    summary = {
        "scope": "normal_validation_patches",
        "patches": count,
        "latent_dim": latent_dim,
        "mean_total_kl": float(mean_kl.sum()),
        "mean_kl_per_unit": float(mean_kl.mean()),
        "active_variance_threshold": active_variance_threshold,
        "active_kl_threshold": active_kl_threshold,
        "active_units_variance": active_variance,
        "active_units_kl": active_kl,
        "posterior_collapse": active_kl == 0,
    }
    return summary, mean_kl, variances


def _save_latent_arrays(
    path: Path,
    mean_kl: np.ndarray,
    variances: np.ndarray,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(
            temporary,
            mean_kl_per_unit=np.asarray(mean_kl, dtype=np.float64),
            z_mean_variance_per_unit=np.asarray(variances, dtype=np.float64),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_result(
    output_dir: Path,
    *,
    seed: int,
    config: VAEConfig,
    training_contract: Mapping[str, Any],
    device: str,
    requested_normalization: NormalizationStats | None,
) -> TrainingResult | None:
    metadata_path = output_dir / "training_metadata.json"
    required = (
        output_dir / "model.keras",
        output_dir / "encoder.keras",
        output_dir / "decoder.keras",
        output_dir / "model_summary.txt",
        output_dir / "history.csv",
        output_dir / "normalization.npz",
        output_dir / "latent_diagnostics.json",
        output_dir / "latent_diagnostics.npz",
        metadata_path,
    )
    existing = tuple(path for path in required if path.exists())
    if not existing:
        return None
    if metadata_path.exists():
        if not metadata_path.is_file():
            raise RuntimeError("Completed training metadata is not a regular file")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Completed training metadata exists but required artifacts are "
                f"missing or invalid: {missing}"
            )
    if not all(path.is_file() for path in required):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("Completed training metadata must contain an object")
    expected_fields = {
        "schema_version": 2,
        "status": "complete",
        "seed": int(seed),
        "device": device,
        "vae_config": _json_round_trip(asdict(config)),
        "training_contract": _json_round_trip(training_contract),
    }
    for name, expected in expected_fields.items():
        if metadata.get(name) != expected:
            raise RuntimeError(
                f"Completed training metadata field {name!r} differs from the request"
            )

    artifact_records = metadata.get("artifacts")
    if not isinstance(artifact_records, dict):
        raise RuntimeError("Completed training metadata has no artifact manifest")
    for path in required:
        if path == metadata_path:
            continue
        record = artifact_records.get(path.name)
        if not isinstance(record, dict):
            raise RuntimeError(f"Training metadata omits artifact {path.name!r}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"Completed artifact size changed: {path.name}")
        if record.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"Completed artifact hash changed: {path.name}")
    model_path = output_dir / "model.keras"
    if metadata.get("model_sha256") != _sha256_file(model_path):
        raise RuntimeError("Completed model hash differs from training metadata")
    normalization = load_normalization(output_dir / "normalization.npz")
    if metadata.get("normalization") != _json_round_trip(asdict(normalization)):
        raise RuntimeError("Completed normalization differs from training metadata")
    if requested_normalization is not None and normalization != requested_normalization:
        raise RuntimeError("Completed normalization differs from the request")
    latent = json.loads((output_dir / "latent_diagnostics.json").read_text(encoding="utf-8"))
    return TrainingResult(
        output_dir=output_dir,
        model_path=model_path,
        history_path=output_dir / "history.csv",
        epochs_completed=int(metadata["epochs_completed"]),
        best_epoch=int(metadata["best_epoch"]),
        stopped_early=bool(metadata["stopped_early"]),
        normalization=normalization,
        latent_diagnostics=latent,
        status="already_complete",
    )


def train_vae(
    train_cache: np.ndarray,
    train_series_indices: Sequence[int] | np.ndarray,
    validation_series_indices: Sequence[int] | np.ndarray,
    output_dir: str | os.PathLike[str],
    *,
    seed: int = 42,
    config: VAEConfig = VAEConfig(),
    normalization: NormalizationStats | None = None,
    device: str = "cpu",
    resume: bool = True,
    verbose: int = 2,
) -> TrainingResult:
    """Train or resume one VAE and atomically commit its reusable artifacts."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if verbose not in {0, 1, 2}:
        raise ValueError("verbose must be 0, 1, or 2")
    if device not in {"cpu", "auto"}:
        raise ValueError("device must be 'cpu' or 'auto'")
    if train_cache.ndim != 5 or tuple(train_cache.shape[2:]) != config.input_shape:
        raise ValueError("Training cache geometry does not match the VAE input")
    if train_cache.dtype != np.dtype(np.float32):
        raise TypeError("Training cache must use float32")
    train_indices = np.asarray(train_series_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_series_indices, dtype=np.int64)
    if train_indices.ndim != 1 or validation_indices.ndim != 1:
        raise ValueError("Training and validation indices must be one-dimensional")
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("Training and validation partitions must be non-empty")
    for name, indices in (
        ("training", train_indices),
        ("validation", validation_indices),
    ):
        if np.unique(indices).size != indices.size:
            raise ValueError(f"The {name} partition contains duplicate series")
        if int(indices.min()) < 0 or int(indices.max()) >= int(train_cache.shape[0]):
            raise IndexError(f"The {name} partition contains an out-of-range series")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Training and validation series overlap")
    training_contract = training_input_contract(
        train_cache,
        train_indices,
        validation_indices,
    )
    completed = _completed_result(
        output,
        seed=seed,
        config=config,
        training_contract=training_contract,
        device=device,
        requested_normalization=normalization,
    )
    if completed is not None:
        return completed

    configure_device(device)
    set_global_seed(seed)
    stats = normalization or compute_normalization(train_cache, train_indices)
    normalization_path = output / "normalization.npz"
    if normalization_path.is_file():
        stored = load_normalization(normalization_path)
        if stored != stats:
            raise RuntimeError("Stored normalization differs from the requested training data")
    else:
        save_normalization(normalization_path, stats)

    train_steps = count_patch_batches(train_indices.size, config.batch_size)
    validation_steps = count_patch_batches(validation_indices.size, config.batch_size)
    keras.backend.clear_session()
    configure_device(device)
    set_global_seed(seed)
    vae, history, state = _load_resume(
        output,
        seed=seed,
        config=config,
        train_steps=train_steps,
        training_contract=training_contract,
        resume=resume,
        device=device,
    )
    start_epoch = len(history)
    if vae is None:
        vae = build_vae(
            config,
            seed=seed,
            initial_beta=0.0,
            learning_rate=config.learning_rate,
            beta_1=config.adam_beta_1,
            device=device,
            compile_model=True,
        )

    probe = next(
        iter_patch_batches(
            train_cache,
            train_indices,
            max(2, min(config.batch_size, 8)),
            shuffle=False,
            normalization=stats.pair,
        )
    )[:2]
    preflight = validate_vae(vae, probe, check_gradients=True)

    reduce_state = state.get("reduce_lr") if state else None
    early_state = state.get("early_stopping") if state else None
    best_state = state.get("best_weights", {}) if state else {}
    reduce_lr = _ResumeReduceLROnPlateau(
        reduce_state,
        monitor="val_total_loss",
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=config.minimum_learning_rate,
        mode="min",
        min_delta=1e-4,
        verbose=1 if verbose else 0,
    )
    early_stopping = _ResumeEarlyStopping(
        early_state,
        monitor="val_total_loss",
        patience=config.early_stopping_patience,
        mode="min",
        min_delta=0.0,
        restore_best_weights=False,
        start_from_epoch=config.beta_warmup_epochs,
        verbose=1 if verbose else 0,
    )
    resume_best_name = best_state.get("path") if best_state else None
    resume_best_path = (
        output / str(resume_best_name)
        if isinstance(resume_best_name, str)
        else None
    )
    best_weights = _BestWeights(
        output,
        start_epoch=config.beta_warmup_epochs - 1,
        best=(
            math.inf
            if not best_state or best_state.get("best") is None
            else float(best_state["best"])
        ),
        best_epoch=int(best_state.get("best_epoch", 0)),
        path=resume_best_path,
    )
    if best_state.get("sha256") is not None:
        if (
            resume_best_path is None
            or not resume_best_path.is_file()
            or _sha256_file(resume_best_path) != best_state["sha256"]
        ):
            raise RuntimeError("Best validation weights differ from resume state")

    state_callback = _EpochState(
        output,
        history=history,
        seed=seed,
        config=config,
        train_steps=train_steps,
        training_contract=training_contract,
        device=device,
        reduce_lr=reduce_lr,
        early_stopping=early_stopping,
        best_weights=best_weights,
    )
    already_stopped = bool(early_state and int(early_state.get("stopped_epoch", 0)) > 0)
    if start_epoch < config.max_epochs and not already_stopped:
        training_dataset = _dataset(
            train_cache,
            train_indices,
            normalization=stats,
            batch_size=config.batch_size,
            seed=seed,
            shuffle=True,
            start_epoch=start_epoch,
            input_shape=config.input_shape,
        )
        validation_dataset = _dataset(
            train_cache,
            validation_indices,
            normalization=stats,
            batch_size=config.batch_size,
            seed=seed,
            shuffle=False,
            start_epoch=0,
            input_shape=config.input_shape,
        )
        vae.fit(
            training_dataset,
            initial_epoch=start_epoch,
            epochs=config.max_epochs,
            steps_per_epoch=train_steps,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[
                BetaWarmup(config.beta_final, config.beta_warmup_epochs),
                _HyperparameterLogger(),
                _FiniteMetrics(),
                reduce_lr,
                early_stopping,
                best_weights,
                state_callback,
            ],
            shuffle=False,
            verbose=verbose,
        )

    if not history:
        raise RuntimeError("Training completed no epoch")
    best_path = best_weights.path
    if best_path is None or not best_path.is_file() or best_weights.best_epoch <= 0:
        raise RuntimeError("No post-warm-up validation checkpoint was saved")
    vae.load_weights(best_path)
    vae.set_beta(config.beta_final)
    postflight = validate_vae(vae, probe, check_gradients=True)

    token = uuid.uuid4().hex
    temporary_model = output / f".model.{token}.keras"
    temporary_encoder = output / f".encoder.{token}.keras"
    temporary_decoder = output / f".decoder.{token}.keras"
    try:
        save_vae(
            vae,
            temporary_model,
            encoder_path=temporary_encoder,
            decoder_path=temporary_decoder,
        )
        reloaded = load_vae(temporary_model, compile_model=True, device=device)
        before = np.asarray(vae.reconstruct_deterministic(probe).numpy())
        after = np.asarray(reloaded.reconstruct_deterministic(probe).numpy())
        if not np.array_equal(before, after):
            raise RuntimeError("Serialization changed deterministic reconstructions")
        os.replace(temporary_model, output / "model.keras")
        os.replace(temporary_encoder, output / "encoder.keras")
        os.replace(temporary_decoder, output / "decoder.keras")
    finally:
        for path in (temporary_model, temporary_encoder, temporary_decoder):
            if path.exists():
                path.unlink()

    (output / "model_summary.txt").write_text(
        model_summary_text(vae) + "\n", encoding="utf-8"
    )
    _atomic_write_csv(output / "history.csv", history, HISTORY_COLUMNS)
    latent_summary, mean_kl, variances = _latent_diagnostics(
        vae,
        train_cache,
        validation_indices,
        stats,
        config.batch_size,
    )
    _atomic_write_json(output / "latent_diagnostics.json", latent_summary)
    _save_latent_arrays(output / "latent_diagnostics.npz", mean_kl, variances)

    stopped_early = len(history) < config.max_epochs
    training_artifact_names = (
        "model.keras",
        "encoder.keras",
        "decoder.keras",
        "model_summary.txt",
        "history.csv",
        "normalization.npz",
        "latent_diagnostics.json",
        "latent_diagnostics.npz",
    )
    training_artifacts = {
        name: {
            "bytes": (output / name).stat().st_size,
            "sha256": _sha256_file(output / name),
        }
        for name in training_artifact_names
    }
    metadata = {
        "schema_version": 2,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "seed": int(seed),
        "device": device,
        "vae_config": asdict(config),
        "training_contract": training_contract,
        "normalization": asdict(stats),
        "normalization_scope": "normal_training_series_only",
        "training_series": int(train_indices.size),
        "validation_series": int(validation_indices.size),
        "epochs_completed": len(history),
        "best_epoch": best_weights.best_epoch,
        "best_validation_total_loss": best_weights.best,
        "stopped_early": stopped_early,
        "inference": "decoder(z_mean)",
        "training_sampling": {
            "strategy": "stateless_seeded_by_optimizer_iteration",
            "base_seed": int(seed),
            "resume_behavior": "exact epsilon sequence from restored Adam iteration",
        },
        "log_variance_safeguard": {
            "definition": "effective_log_var = min(raw_log_var, upper_cap)",
            "upper_cap": config.effective_log_var_upper_cap,
            "applied_to": ["sampling", "KL"],
        },
        "preflight": preflight,
        "postflight": postflight,
        "latent_diagnostics": latent_summary,
        "model_sha256": _sha256_file(output / "model.keras"),
        "artifacts": training_artifacts,
    }
    _atomic_write_json(output / "training_metadata.json", metadata)

    for path in (output / ".training_state.json", best_path):
        if path.is_file():
            path.unlink()
    _cleanup_resume_artifacts(output, keep=set())

    keras.backend.clear_session()
    return TrainingResult(
        output_dir=output,
        model_path=output / "model.keras",
        history_path=output / "history.csv",
        epochs_completed=len(history),
        best_epoch=best_weights.best_epoch,
        stopped_early=stopped_early,
        normalization=stats,
        latent_diagnostics=latent_summary,
        status="trained",
    )


def train_seed(
    train_cache: np.ndarray,
    split: SeriesSplit,
    output_dir: str | os.PathLike[str],
    experiment: ExperimentConfig,
    *,
    resume: bool = True,
    verbose: int = 2,
) -> TrainingResult:
    """Convenience wrapper using one seed and settings from ExperimentConfig."""

    if int(split.seed) not in experiment.seeds:
        raise ValueError("Split seed is not listed in the experiment configuration")
    return train_vae(
        train_cache,
        split.train_indices,
        split.validation_indices,
        output_dir,
        seed=int(split.seed),
        config=experiment.vae,
        device=experiment.device,
        resume=resume,
        verbose=verbose,
    )


__all__ = [
    "HISTORY_COLUMNS",
    "NormalizationStats",
    "TrainingResult",
    "compute_normalization",
    "load_normalization",
    "save_normalization",
    "training_input_contract",
    "train_seed",
    "train_vae",
]
