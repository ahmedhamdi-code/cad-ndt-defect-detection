"""Convolutional variational autoencoder for signed AIRBUS STFT patches.

The input representation contains negative and positive values.  The decoder
therefore uses a linear output and the reconstruction term is the negative
log-likelihood of a unit-variance Gaussian, up to its additive constant.
Training samples from the approximate posterior; inference decodes ``z_mean``
to provide deterministic anomaly scores.
"""

from __future__ import annotations

import math
import numbers
import os
from pathlib import Path
from typing import Any, Callable, Mapping


os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import VAEConfig

CONVOLUTION_FILTERS = 32
KERNEL_SIZE = 4
LEAKY_RELU_SLOPE = 0.3


def _encoded_shape(config: VAEConfig) -> tuple[int, int, int]:
    height, width, _ = config.input_shape
    if height % 4 or width % 4:
        raise ValueError("input height and width must be divisible by four")
    return height // 4, width // 4, CONVOLUTION_FILTERS


def _flattened_size(config: VAEConfig) -> int:
    return int(np.prod(_encoded_shape(config), dtype=np.int64))


def configure_device(device: str = "cpu") -> tuple[str, ...]:
    """Configure TensorFlow device visibility and return visible device names.

    CPU execution is used by default because it was faster for this compact
    model on the reference Apple Silicon system.  ``device='auto'`` leaves the
    visibility selected before this module was imported unchanged.
    """

    normalized = str(device).strip().lower()
    if normalized not in {"cpu", "auto"}:
        raise ValueError("device must be 'cpu' or 'auto'")
    if normalized == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            if tf.config.get_visible_devices("GPU"):
                raise RuntimeError(
                    "TensorFlow initialized a GPU before CPU-only execution was requested"
                ) from exc
    return tuple(item.name for item in tf.config.get_visible_devices())


def set_global_seed(seed: int, *, deterministic_ops: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow through the Keras helper."""

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    keras.utils.set_random_seed(int(seed))
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def effective_log_variance(z_log_var: tf.Tensor, cap: float) -> tf.Tensor:
    """Apply the upper log-variance safeguard used by sampling and KL.

    The cap prevents ``exp(z_log_var)`` from overflowing on heavy-tailed,
    standardized STFT patches.  The raw encoder output remains available for
    diagnostics.  A lower cap is unnecessary because the exponential safely
    approaches zero while the linear KL term still penalizes extreme values.
    """

    values = tf.convert_to_tensor(z_log_var)
    return tf.minimum(values, tf.cast(cap, values.dtype))


@keras.utils.register_keras_serializable(package="vae_cad")
class Sampling(keras.layers.Layer):
    """Reparameterized sampling used by the encoder's explicit sample output.

    The custom training step uses a stateless optimizer-step seed instead.  Its
    sequence can therefore be reconstructed exactly from a saved optimizer
    iteration, independently of this layer's diagnostic random generator.
    """

    def __init__(
        self,
        log_var_cap: float = 10.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not math.isfinite(float(log_var_cap)):
            raise ValueError("log_var_cap must be finite")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.log_var_cap = float(log_var_cap)
        self.seed = int(seed)
        self.seed_generator = keras.random.SeedGenerator(self.seed)

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool | tf.Tensor | None = None,
    ) -> tf.Tensor:
        z_mean, z_log_var = inputs
        safe_log_var = effective_log_variance(z_log_var, self.log_var_cap)

        def sample() -> tf.Tensor:
            epsilon = keras.random.normal(
                tf.shape(z_mean), dtype=z_mean.dtype, seed=self.seed_generator
            )
            return z_mean + tf.exp(
                tf.cast(0.5, z_mean.dtype) * safe_log_var
            ) * epsilon

        if training is True:
            return sample()
        if training is False or training is None:
            return z_mean
        return tf.cond(tf.cast(training, tf.bool), sample, lambda: z_mean)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"log_var_cap": self.log_var_cap, "seed": self.seed})
        return config


def build_encoder(config: VAEConfig = VAEConfig(), *, seed: int = 42) -> keras.Model:
    """Build an encoder returning ``z_mean``, ``z_log_var`` and sampled ``z``."""

    inputs = keras.Input(shape=config.input_shape, name="encoder_input")
    x = keras.layers.Conv2D(
        CONVOLUTION_FILTERS,
        KERNEL_SIZE,
        strides=2,
        padding="same",
        name="encoder_conv",
    )(inputs)
    x = keras.layers.LeakyReLU(
        negative_slope=LEAKY_RELU_SLOPE,
        name="encoder_activation",
    )(x)
    x = keras.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding="same",
        name="encoder_pool",
    )(x)
    x = keras.layers.Flatten(name="encoder_flatten")(x)
    z_mean = keras.layers.Dense(config.latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(config.latent_dim, name="z_log_var")(x)
    z = Sampling(config.effective_log_var_upper_cap, seed=seed, name="sampling")(
        (z_mean, z_log_var)
    )
    return keras.Model(inputs, (z_mean, z_log_var, z), name="vae_encoder")


def build_decoder(config: VAEConfig = VAEConfig()) -> keras.Model:
    """Build the linear decoder appropriate for signed STFT values."""

    inputs = keras.Input((config.latent_dim,), name="decoder_input")
    x = keras.layers.Dense(_flattened_size(config), name="decoder_dense")(inputs)
    x = keras.layers.LeakyReLU(
        negative_slope=LEAKY_RELU_SLOPE,
        name="decoder_activation",
    )(x)
    x = keras.layers.Reshape(_encoded_shape(config), name="decoder_reshape")(x)
    x = keras.layers.UpSampling2D(
        size=2,
        interpolation="nearest",
        name="decoder_upsample",
    )(x)
    outputs = keras.layers.Conv2DTranspose(
        config.input_shape[-1],
        KERNEL_SIZE,
        strides=2,
        padding="same",
        activation=None,
        name="reconstruction",
    )(x)
    return keras.Model(inputs, outputs, name="vae_decoder")


def _weighted_mean(values: tf.Tensor, sample_weight: tf.Tensor | None) -> tf.Tensor:
    values = tf.reshape(tf.convert_to_tensor(values), (-1,))
    if sample_weight is None:
        return tf.reduce_mean(values)
    weights = tf.cast(tf.reshape(sample_weight, (-1,)), values.dtype)
    tf.debugging.assert_equal(tf.shape(values), tf.shape(weights))
    return tf.math.divide_no_nan(tf.reduce_sum(values * weights), tf.reduce_sum(weights))


@keras.utils.register_keras_serializable(package="vae_cad")
class ConvolutionalVAE(keras.Model):
    """VAE with an explicit Gaussian reconstruction loss and standard KL."""

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        beta: float = 0.0,
        log_var_cap: float = 10.0,
        sampling_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not math.isfinite(float(beta)) or float(beta) < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(float(log_var_cap)):
            raise ValueError("log_var_cap must be finite")
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, numbers.Integral)
            or int(sampling_seed) < 0
        ):
            raise ValueError("sampling_seed must be a non-negative integer")
        self.encoder = encoder
        self.decoder = decoder
        self.log_var_cap = float(log_var_cap)
        self.sampling_seed = int(sampling_seed)
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=keras.initializers.Constant(float(beta)),
            trainable=False,
            dtype=tf.float32,
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def set_beta(self, value: float) -> None:
        """Set the KL coefficient used by subsequent train and test steps."""

        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("beta must be finite and non-negative")
        self.beta.assign(float(value))

    def encode_mean(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the deterministic posterior mean."""

        z_mean, _, _ = self.encoder(inputs, training=False)
        return z_mean

    def reconstruct_deterministic(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reconstruct by decoding ``z_mean`` exactly once."""

        return self.decoder(self.encode_mean(inputs), training=False)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | tf.Tensor | None = None,
        deterministic: bool | None = None,
    ) -> tf.Tensor:
        z_mean, _, sampled_z = self.encoder(inputs, training=training)
        if deterministic is True or (deterministic is None and training is None):
            latent = z_mean
        elif deterministic is False:
            latent = sampled_z
        elif isinstance(training, bool):
            latent = sampled_z if training else z_mean
        else:
            latent = tf.cond(tf.cast(training, tf.bool), lambda: sampled_z, lambda: z_mean)
        return self.decoder(latent, training=training)

    def loss_components(
        self,
        inputs: tf.Tensor,
        *,
        training: bool,
        sample: bool,
        sample_weight: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return total, half-sum squared reconstruction, and KL losses."""

        x = tf.cast(inputs, self.compute_dtype)
        z_mean, raw_log_var, _ = self.encoder(x, training=training)
        safe_log_var = effective_log_variance(raw_log_var, self.log_var_cap)
        if sample:
            if not hasattr(self, "optimizer"):
                raise RuntimeError("Compile the VAE before requesting a training sample")
            # Adam's iteration is persisted in the .keras archive.  Using it as
            # the second stateless seed component makes batch k use the same
            # epsilon whether the fit is uninterrupted or resumed.
            iteration = tf.cast(self.optimizer.iterations, tf.int32)
            seed = tf.stack(
                (
                    tf.cast(self.sampling_seed % (2**31 - 1), tf.int32),
                    iteration,
                )
            )
            epsilon = tf.random.stateless_normal(
                tf.shape(z_mean), seed=seed, dtype=z_mean.dtype
            )
            latent = z_mean + tf.exp(
                tf.cast(0.5, z_mean.dtype) * safe_log_var
            ) * epsilon
        else:
            latent = z_mean
        reconstruction = self.decoder(latent, training=training)
        axes = tf.range(1, tf.rank(x))
        reconstruction_per_sample = 0.5 * tf.reduce_sum(
            tf.square(x - reconstruction), axis=axes
        )
        kl_per_sample = -0.5 * tf.reduce_sum(
            1.0 + safe_log_var - tf.square(z_mean) - tf.exp(safe_log_var),
            axis=1,
        )
        reconstruction_loss = _weighted_mean(reconstruction_per_sample, sample_weight)
        kl_loss = _weighted_mean(kl_per_sample, sample_weight)
        total_loss = reconstruction_loss + tf.cast(self.beta, reconstruction_loss.dtype) * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    @staticmethod
    def _unpack(data: Any) -> tuple[tf.Tensor, tf.Tensor | None]:
        x, _, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        if isinstance(x, Mapping):
            if len(x) != 1:
                raise ValueError("The VAE expects exactly one input tensor")
            x = next(iter(x.values()))
        return x, sample_weight

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        x, sample_weight = self._unpack(data)
        with tf.GradientTape() as tape:
            total, reconstruction, kl = self.loss_components(
                x, training=True, sample=True, sample_weight=sample_weight
            )
        gradients = tape.gradient(total, self.trainable_weights)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_weights, strict=True)
            if gradient is not None
        ]
        if not pairs:
            raise RuntimeError("No gradients were produced")
        for gradient, variable in pairs:
            tf.debugging.assert_all_finite(gradient, f"Non-finite gradient for {variable.name}")
        self.optimizer.apply_gradients(pairs)
        metric_weight = (
            tf.cast(tf.shape(x)[0], total.dtype)
            if sample_weight is None
            else tf.reduce_sum(tf.cast(sample_weight, total.dtype))
        )
        self.total_loss_tracker.update_state(total, sample_weight=metric_weight)
        self.reconstruction_loss_tracker.update_state(
            reconstruction, sample_weight=metric_weight
        )
        self.kl_loss_tracker.update_state(kl, sample_weight=metric_weight)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        x, sample_weight = self._unpack(data)
        total, reconstruction, kl = self.loss_components(
            x, training=False, sample=False, sample_weight=sample_weight
        )
        metric_weight = (
            tf.cast(tf.shape(x)[0], total.dtype)
            if sample_weight is None
            else tf.reduce_sum(tf.cast(sample_weight, total.dtype))
        )
        self.total_loss_tracker.update_state(total, sample_weight=metric_weight)
        self.reconstruction_loss_tracker.update_state(
            reconstruction, sample_weight=metric_weight
        )
        self.kl_loss_tracker.update_state(kl, sample_weight=metric_weight)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "encoder": keras.utils.serialize_keras_object(self.encoder),
                "decoder": keras.utils.serialize_keras_object(self.decoder),
                "beta": float(self.beta.numpy()),
                "log_var_cap": self.log_var_cap,
                "sampling_seed": self.sampling_seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConvolutionalVAE":
        values = dict(config)
        encoder = keras.utils.deserialize_keras_object(values.pop("encoder"))
        decoder = keras.utils.deserialize_keras_object(values.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **values)


@keras.utils.register_keras_serializable(package="vae_cad")
class BetaWarmup(keras.callbacks.Callback):
    """Linearly increase beta to its final value during the first epochs."""

    def __init__(self, final_beta: float = 1.0, warmup_epochs: int = 5) -> None:
        super().__init__()
        if not math.isfinite(float(final_beta)) or float(final_beta) < 0.0:
            raise ValueError("final_beta must be finite and non-negative")
        if isinstance(warmup_epochs, bool) or int(warmup_epochs) <= 0:
            raise ValueError("warmup_epochs must be a positive integer")
        self.final_beta = float(final_beta)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del logs
        if not isinstance(self.model, ConvolutionalVAE):
            raise TypeError("BetaWarmup must be attached to ConvolutionalVAE")
        fraction = min((int(epoch) + 1) / self.warmup_epochs, 1.0)
        self.model.set_beta(self.final_beta * fraction)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del epoch
        if logs is not None and isinstance(self.model, ConvolutionalVAE):
            logs["beta"] = float(self.model.beta.numpy())

    def get_config(self) -> dict[str, Any]:
        return {"final_beta": self.final_beta, "warmup_epochs": self.warmup_epochs}


def compile_vae(
    vae: ConvolutionalVAE,
    *,
    learning_rate: float = 1e-3,
    beta_1: float = 0.9,
    run_eagerly: bool = False,
) -> ConvolutionalVAE:
    """Compile the VAE with Adam and its model-owned loss."""

    if not isinstance(vae, ConvolutionalVAE):
        raise TypeError("vae must be a ConvolutionalVAE")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 < float(beta_1) < 1.0:
        raise ValueError("beta_1 must be between zero and one")
    vae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(learning_rate), beta_1=float(beta_1)
        ),
        run_eagerly=bool(run_eagerly),
    )
    return vae


def build_vae(
    config: VAEConfig = VAEConfig(),
    *,
    seed: int = 42,
    initial_beta: float = 0.0,
    learning_rate: float = 1e-3,
    beta_1: float = 0.9,
    device: str = "cpu",
    compile_model: bool = True,
) -> ConvolutionalVAE:
    """Build and optionally compile a reproducibly initialized VAE."""

    configure_device(device)
    set_global_seed(seed)
    encoder = build_encoder(config, seed=seed)
    decoder = build_decoder(config)
    vae = ConvolutionalVAE(
        encoder,
        decoder,
        beta=initial_beta,
        log_var_cap=config.effective_log_var_upper_cap,
        sampling_seed=seed,
        name="vae",
    )
    vae(tf.zeros((1, *config.input_shape), dtype=tf.float32), training=False)
    if compile_model:
        compile_vae(vae, learning_rate=learning_rate, beta_1=beta_1)
    return vae


def validate_vae(
    vae: ConvolutionalVAE,
    probe_batch: np.ndarray | tf.Tensor | None = None,
    *,
    check_gradients: bool = True,
) -> dict[str, Any]:
    """Validate shapes, stochastic sampling, deterministic inference and losses."""

    if not isinstance(vae, ConvolutionalVAE):
        raise TypeError("vae must be a ConvolutionalVAE")
    shape = tuple(int(value) for value in vae.encoder.input_shape[1:])
    if probe_batch is None:
        probe_batch = np.zeros((2, *shape), dtype=np.float32)
    probe = tf.cast(tf.convert_to_tensor(probe_batch), vae.compute_dtype)
    if probe.shape.rank != 4 or tuple(probe.shape[1:]) != shape:
        raise ValueError(f"probe_batch must have shape (batch, {shape})")
    if probe.shape[0] is not None and int(probe.shape[0]) < 2:
        raise ValueError("probe_batch must contain at least two examples")
    tf.debugging.assert_all_finite(probe, "Non-finite probe input")

    mean_a, log_var_a, _ = vae.encoder(probe, training=False)
    mean_b, log_var_b, _ = vae.encoder(probe, training=False)
    if not np.array_equal(mean_a.numpy(), mean_b.numpy()):
        raise RuntimeError("z_mean is not deterministic")
    if not np.array_equal(log_var_a.numpy(), log_var_b.numpy()):
        raise RuntimeError("z_log_var is not deterministic")
    _, _, sample_a = vae.encoder(probe, training=True)
    _, _, sample_b = vae.encoder(probe, training=True)
    if np.array_equal(sample_a.numpy(), sample_b.numpy()):
        raise RuntimeError("The sampling layer did not produce different draws")

    reconstruction_a = vae.reconstruct_deterministic(probe)
    reconstruction_b = vae.reconstruct_deterministic(probe)
    if tuple(reconstruction_a.shape) != tuple(probe.shape):
        raise RuntimeError("The reconstruction shape differs from the input shape")
    if not np.array_equal(reconstruction_a.numpy(), reconstruction_b.numpy()):
        raise RuntimeError("decoder(z_mean) is not deterministic")
    tf.debugging.assert_all_finite(reconstruction_a, "Non-finite reconstruction")

    total, reconstruction, kl = vae.loss_components(
        probe, training=False, sample=False
    )
    for name, value in (("total", total), ("reconstruction", reconstruction), ("KL", kl)):
        tf.debugging.assert_all_finite(value, f"Non-finite {name} loss")
    if float(kl.numpy()) < -1e-5:
        raise RuntimeError("KL loss is negative")

    gradients_finite: bool | None = None
    if check_gradients:
        if not hasattr(vae, "optimizer"):
            raise RuntimeError("Compile the VAE before validating gradients")
        with tf.GradientTape() as tape:
            gradient_loss, _, _ = vae.loss_components(
                probe, training=True, sample=True
            )
        gradients = tape.gradient(gradient_loss, vae.trainable_weights)
        if not gradients or all(item is None for item in gradients):
            raise RuntimeError("No gradients were produced")
        for gradient, variable in zip(gradients, vae.trainable_weights, strict=True):
            if gradient is not None:
                tf.debugging.assert_all_finite(gradient, f"Non-finite gradient for {variable.name}")
        gradients_finite = True

    return {
        "input_shape": list(probe.shape),
        "output_shape": list(reconstruction_a.shape),
        "latent_dim": int(vae.encoder.output_shape[0][-1]),
        "parameters": int(vae.count_params()),
        "sampling_stochastic": True,
        "training_sampling": "stateless_seeded_by_optimizer_iteration",
        "training_sampling_seed": vae.sampling_seed,
        "deterministic_inference": True,
        "log_var_cap": vae.log_var_cap,
        "total_loss": float(total.numpy()),
        "reconstruction_loss": float(reconstruction.numpy()),
        "kl_loss": float(kl.numpy()),
        "gradients_finite": gradients_finite,
    }


def save_vae(
    vae: ConvolutionalVAE,
    path: str | os.PathLike[str],
    *,
    encoder_path: str | os.PathLike[str] | None = None,
    decoder_path: str | os.PathLike[str] | None = None,
) -> None:
    """Save the complete model and optional standalone encoder and decoder."""

    targets = [Path(path)]
    if encoder_path is not None:
        targets.append(Path(encoder_path))
    if decoder_path is not None:
        targets.append(Path(decoder_path))
    for target in targets:
        if target.suffix != ".keras":
            raise ValueError(f"Keras model path must end with .keras: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    vae.save(targets[0], overwrite=True)
    if encoder_path is not None:
        vae.encoder.save(Path(encoder_path), overwrite=True)
    if decoder_path is not None:
        vae.decoder.save(Path(decoder_path), overwrite=True)


def load_vae(
    path: str | os.PathLike[str],
    *,
    compile_model: bool = True,
    device: str = "cpu",
) -> ConvolutionalVAE:
    """Load a registered VAE archive without a custom-object mapping."""

    configure_device(device)
    model = keras.models.load_model(Path(path), compile=compile_model, safe_mode=True)
    if not isinstance(model, ConvolutionalVAE):
        raise TypeError(f"Expected ConvolutionalVAE, found {type(model).__name__}")
    return model


def model_summary_text(vae: ConvolutionalVAE) -> str:
    """Return complete model, encoder and decoder summaries as text."""

    sections: list[str] = []
    for title, model in (("VAE", vae), ("Encoder", vae.encoder), ("Decoder", vae.decoder)):
        lines: list[str] = []

        def capture(line: str, **_: Any) -> None:
            lines.append(line)

        printer: Callable[..., None] = capture
        model.summary(print_fn=printer)
        sections.append(f"{title}\n{'=' * len(title)}\n" + "\n".join(lines))
    return "\n\n".join(sections)


__all__ = [
    "BetaWarmup",
    "ConvolutionalVAE",
    "Sampling",
    "VAEConfig",
    "build_decoder",
    "build_encoder",
    "build_vae",
    "compile_vae",
    "configure_device",
    "effective_log_variance",
    "load_vae",
    "model_summary_text",
    "save_vae",
    "set_global_seed",
    "validate_vae",
]
