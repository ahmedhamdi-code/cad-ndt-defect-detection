# VAE-CAD

This repository provides a clean and reproducible implementation of the AIRBUS
experiment from *Unsupervised Fault Detection in Industrial Images with
Conformal Anomaly Detection and Variational Autoencoders*.

The implementation turns vibration time series into time-frequency patches,
trains a variational autoencoder (VAE) using normal data only, and compares a
reconstruction-error baseline with conformal anomaly detection based on the
Wasserstein-1 distance between residual histograms.

No dataset is included in the repository. Download instructions are available
in [`data/README.md`](data/README.md).

## Authors

Ahmed Hamdi<sup>a,b,†</sup>, Flora Estermann<sup>a,b,†</sup>, Valerie Kaftandjian<sup>b,*</sup>,
Philippe Delachartre<sup>a</sup>, Philippe Guy<sup>b</sup>

<sup>a</sup> Univ. Lyon, INSA-Lyon, CNRS, Inserm, CREATIS UMR 5220, U1294, Villeurbanne, F-69621, France
<sup>b</sup> Univ. Lyon, INSA-Lyon, LVA, Villeurbanne, F-69621, France

<sup>†</sup> These authors contributed equally to this work.
<sup>*</sup> Corresponding author.

## Reproducibility protocol

The default command runs three independent experiments with seeds 42, 43, and
44. For each seed, the 1,677 normal training series are partitioned at series
level before their cached patches are assigned to experimental roles:

- 1,206 series for VAE fitting and detector reference construction;
- 135 series for VAE validation;
- 336 series for independent conformal calibration.

The labelled set of 594 series is used only for final evaluation. Hyperparameters
and decision rules are fixed before evaluation. The three methods reported are:

- **VAE+MAE**: patch MAE with a Q99 training threshold and a maximum/any-patch
  series rule;
- **VAE+CAD-W1 (patch calibration)**: conformal p-values calibrated at patch
  level and the historical any-rejected-patch series rule;
- **VAE+CAD-W1 (series calibration)**: patch Wasserstein scores aggregated first,
  followed by conformal calibration at series level.

The last variant gives `alpha` a direct interpretation at series level. In the
patch-calibrated variant, applying the rule to 120 patches means that
`alpha = 0.02` is not a 2% false-positive guarantee for a complete series.

Each metric is first computed on the complete 594-series test set. Results are
then summarized as the mean and sample standard deviation across the three
seeds (`ddof=1`). The standard deviation therefore measures variation caused by
training and data partitioning; it is not computed across individual test
examples.

## Installation

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow runs on CPU by default. No GPU-specific package is required.

## Data preparation

After placing the three source files in `data/raw/`, build the STFT cache:

```bash
python prepare_data.py
```

The exact transformation is a Hann-window STFT with `fs=1024`,
`nperseg=126`, `noverlap=118`, and `nfft=126`. The final time column is
removed, the real signed component is retained, and each series is divided into
120 chronological `64 x 64 x 1` patches. The cache and its provenance manifest
are written to `cache/`. Allow approximately 4.2 GiB of free space for these
generated arrays. These detailed STFT settings follow the reference notebooks,
because the publication does not specify every preprocessing argument.

For an exhaustive integrity check of an existing cache and its source files,
run:

```bash
python prepare_data.py --verify-source-hashes --full-cache-scan
```

## Run the complete experiment

```bash
python run_experiments.py
```

This command validates or prepares the cache, then executes seeds 42, 43, and 44
sequentially before aggregating their results. Existing complete runs are not
overwritten unless `--overwrite` is supplied explicitly.

Custom paths can be used without editing the source:

```bash
python run_experiments.py \
  --data-root /path/to/AIRBUS \
  --cache-root /path/to/cache \
  --output-root /path/to/outputs
```

The individual stages are also available:

```bash
python train.py --seed 42
python inference.py --seed 42
python aggregate_results.py
```

Use `--help` on any command for all options.

## Outputs

Each completed run is stored in `outputs/seed_<seed>/` and includes the model,
training history, split manifest, fitted detector state, per-series predictions,
metrics, and a completion manifest. Aggregated artifacts are written directly to
`outputs/`:

```text
outputs/
├── seed_42/
├── seed_43/
├── seed_44/
├── metrics_per_run.csv
├── metrics_summary.csv
└── reproduction_report.md
```

`metrics_summary.csv` reports the number of runs, mean, sample standard
deviation, and a formatted `mean ± SD` value for every metric and method.

## Methodological notes

- The VAE has distinct `z_mean` and `z_log_var` outputs and uses stochastic
  reparameterization during training.
- The publication does not fully specify an AIRBUS VAE architecture. This
  repository therefore records an explicit architecture and every associated
  training choice rather than leaving them implicit.
- Because the real STFT representation is signed, the decoder output is linear
  and the reconstruction term is Gaussian squared error rather than
  binary cross-entropy.
- No data augmentation is applied in this fixed AIRBUS protocol.
- Normalization statistics are learned exclusively from the normal VAE training
  partition. Residual scores are computed after returning reconstructions to
  the original STFT units.
- Scoring uses `decoder(z_mean)`, making every reconstruction deterministic for
  a fitted model. Calibration and test reconstructions are shared by all
  detectors; normal training data are read in two deterministic passes to avoid
  storing a multi-gigabyte residual cache.
- Wasserstein-1 is computed on shared bin centres using normalized histogram
  values as probability masses.
- Finite-sample conformal p-values use
  `(1 + count(calibration_score >= test_score)) / (n_calibration + 1)`.
- For numerical safety, the effective `z_log_var` used in sampling and KL is
  capped above at 10; the raw encoder output remains available to diagnostics.
- All splits are made at series level to prevent patches from the same series
  appearing in multiple normal-data partitions.
- Completion records include hashes of the source code and generated artifacts,
  preventing silent aggregation of runs produced by different implementations.

The generated report records the complete configuration and the metrics
observed during the current execution.
