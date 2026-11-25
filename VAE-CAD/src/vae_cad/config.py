"""Central configuration for the public AIRBUS VAE-CAD reproduction.

All paths are relative to the repository by default.  The experimental
parameters are collected here so the command-line entry points stay small and
the three runs use the same protocol apart from their random seeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SEEDS = (42, 43, 44)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset files, cache paths, and the validated AIRBUS data contract."""

    data_root: Path = DEFAULT_DATA_ROOT
    cache_root: Path = DEFAULT_CACHE_ROOT
    train_filename: str = "dftrain.h5"
    test_filename: str = "dfvalid.h5"
    labels_filename: str = "dfvalid_groundtruth.csv"
    train_hdf5_key: str = "dftrain"
    test_hdf5_key: str = "dfvalid"
    train_series: int = 1_677
    test_series: int = 594
    samples_per_series: int = 61_440
    normal_test_series: int = 297
    anomalous_test_series: int = 297
    patches_per_series: int = 120
    patch_shape: tuple[int, int, int] = (64, 64, 1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser().resolve())
        object.__setattr__(self, "cache_root", Path(self.cache_root).expanduser().resolve())
        if self.train_series <= 0 or self.test_series <= 0:
            raise ValueError("Dataset sizes must be positive")
        if self.samples_per_series != 61_440:
            raise ValueError("AIRBUS series must contain exactly 61,440 samples")
        if self.normal_test_series + self.anomalous_test_series != self.test_series:
            raise ValueError("Test class counts must sum to the test-set size")
        if self.patches_per_series != 120 or self.patch_shape != (64, 64, 1):
            raise ValueError("The AIRBUS cache geometry must be 120 x 64 x 64 x 1")

    @property
    def train_hdf5_path(self) -> Path:
        return self.data_root / self.train_filename

    @property
    def test_hdf5_path(self) -> Path:
        return self.data_root / self.test_filename

    @property
    def labels_path(self) -> Path:
        return self.data_root / self.labels_filename

    @property
    def train_cache_path(self) -> Path:
        return self.cache_root / "train_stft.npy"

    @property
    def test_cache_path(self) -> Path:
        return self.cache_root / "test_stft.npy"

    @property
    def cache_manifest_path(self) -> Path:
        return self.cache_root / "stft_manifest.json"

    @property
    def train_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (self.train_series, self.patches_per_series, *self.patch_shape)

    @property
    def test_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (self.test_series, self.patches_per_series, *self.patch_shape)


@dataclass(frozen=True, slots=True)
class STFTConfig:
    """Explicit STFT parameters recovered from the AIRBUS reference code."""

    fs: int = 1_024
    window: str = "hann"
    nperseg: int = 126
    noverlap: int = 118
    nfft: int = 126
    detrend: bool = False
    return_onesided: bool = True
    boundary: str = "zeros"
    padded: bool = True
    scaling: str = "spectrum"
    raw_shape: tuple[int, int] = (64, 7_681)
    retained_shape: tuple[int, int] = (64, 7_680)

    def __post_init__(self) -> None:
        if self.nperseg <= 0 or not 0 <= self.noverlap < self.nperseg:
            raise ValueError("Invalid STFT window or overlap")
        if self.nfft < self.nperseg:
            raise ValueError("nfft must be at least nperseg")
        if self.retained_shape != (64, 120 * 64):
            raise ValueError("The retained STFT must tile into 120 patches")

    def scipy_kwargs(self) -> dict[str, Any]:
        """Return every behavior-affecting argument passed to SciPy."""

        return {
            "fs": self.fs,
            "window": self.window,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "nfft": self.nfft,
            "detrend": self.detrend,
            "return_onesided": self.return_onesided,
            "boundary": self.boundary,
            "padded": self.padded,
            "axis": -1,
            "scaling": self.scaling,
        }


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Leakage-safe split sizes applied before series become patches."""

    train_series: int = 1_206
    validation_series: int = 135
    calibration_series: int = 336

    @property
    def total(self) -> int:
        return self.train_series + self.validation_series + self.calibration_series


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """True convolutional VAE and optimization parameters."""

    input_shape: tuple[int, int, int] = (64, 64, 1)
    latent_dim: int = 168
    learning_rate: float = 1e-3
    adam_beta_1: float = 0.9
    batch_size: int = 128
    max_epochs: int = 40
    beta_final: float = 1.0
    beta_warmup_epochs: int = 5
    reduce_lr_factor: float = 0.1
    reduce_lr_patience: int = 4
    minimum_learning_rate: float = 1e-6
    early_stopping_patience: int = 7
    effective_log_var_upper_cap: float = 10.0
    augmentation: bool = False

    def __post_init__(self) -> None:
        if self.input_shape != (64, 64, 1) or self.latent_dim != 168:
            raise ValueError("The public AIRBUS VAE geometry is fixed")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("Batch size and epoch count must be positive")
        if self.learning_rate <= 0 or not 0 < self.adam_beta_1 < 1:
            raise ValueError("Invalid Adam configuration")
        if self.beta_final < 0 or self.beta_warmup_epochs <= 0:
            raise ValueError("Invalid KL warm-up configuration")
        if self.effective_log_var_upper_cap <= 0:
            raise ValueError("The effective log-variance cap must be positive")
        if self.augmentation:
            raise ValueError("The public AIRBUS protocol does not use augmentation")


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Pre-specified MAE and conformal Wasserstein detector parameters."""

    mae_quantile: float = 0.99
    alpha: float = 0.02
    bins: str = "auto_median"
    series_aggregation: str = "max"

    def __post_init__(self) -> None:
        if not 0 < self.mae_quantile < 1 or not 0 < self.alpha < 1:
            raise ValueError("Quantile and alpha must lie strictly between zero and one")
        if self.bins != "auto_median":
            raise ValueError("The public protocol derives bins from normal training residuals")
        if self.series_aggregation not in {"max", "top5", "q95"}:
            raise ValueError("Unsupported series aggregation")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration shared by preprocessing, training, and scoring."""

    data: DataConfig = field(default_factory=DataConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    output_root: Path = DEFAULT_OUTPUT_ROOT
    device: str = "cpu"
    protocol_version: str = "airbus-vae-cad-w1-public-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser().resolve())
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if len(self.seeds) == 0 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be a non-empty sequence of unique integers")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be non-negative")
        if self.split.total != self.data.train_series:
            raise ValueError("The train/validation/calibration split must cover all series")
        if self.device not in {"cpu", "auto"}:
            raise ValueError("Device must be 'cpu' or 'auto'")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with resolved paths."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_config(
    *,
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    device: str = "cpu",
) -> ExperimentConfig:
    """Create a configuration with portable command-line path overrides."""

    data = DataConfig(
        data_root=DEFAULT_DATA_ROOT if data_root is None else Path(data_root),
        cache_root=DEFAULT_CACHE_ROOT if cache_root is None else Path(cache_root),
    )
    return ExperimentConfig(
        data=data,
        seeds=DEFAULT_SEEDS if seeds is None else tuple(int(seed) for seed in seeds),
        output_root=DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root),
        device=device,
    )


DEFAULT_CONFIG = ExperimentConfig()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible object using one canonical representation."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_sha256(root: str | Path = PROJECT_ROOT) -> str:
    """Hash every executable Python source file in canonical path order."""

    repository = Path(root).expanduser().resolve()
    paths = sorted(
        {
            *repository.glob("*.py"),
            *(repository / "src" / "vae_cad").glob("*.py"),
        },
        key=lambda path: path.relative_to(repository).as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No Python source files found under {repository}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def protocol_payload(
    config: ExperimentConfig,
    cache_manifest_payload_sha256: str,
) -> dict[str, Any]:
    """Return the seed-independent protocol persisted by every complete run."""

    configuration = dict(config.as_dict())
    configuration.pop("seeds", None)
    return {
        "protocol_version": config.protocol_version,
        "source_tree_sha256": source_tree_sha256(),
        "configuration_without_seed": configuration,
        "cache_manifest_payload_sha256": str(cache_manifest_payload_sha256),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "DataConfig",
    "DetectorConfig",
    "ExperimentConfig",
    "SplitConfig",
    "STFTConfig",
    "VAEConfig",
    "canonical_json_sha256",
    "make_config",
    "protocol_payload",
    "source_tree_sha256",
]
