# DINOv2-CAD

This folder provides the feature-based (DINOv2) pipeline of the paper. It
uses frozen DINOv2 features, a patch-level memory bank and a small
reconstruction head, and wraps the fused score in Conformal Anomaly
Detection (CAD). It covers four industrial NDT datasets.

No dataset is included. Download instructions are available in
[`data/README.md`](data/README.md).

## Pipeline

For each dataset:

1. A frozen DINOv2-small (ViT-S/14) extracts patch-level features from
   normal (defect-free) images.
2. These features fill a memory bank through greedy coreset selection.
   A test image is scored by the distance between its patches and their
   nearest neighbours in the bank.
3. A small decoder is trained on the same frozen features to
   reconstruct the image, which gives a second residual score.
4. The two scores are combined into one non-conformity score with a
   weight `w`.
5. CAD turns this score into a p-value using a held-out calibration set
   of normal samples. A sample is flagged when its p-value is below the
   significance level `alpha`.

`w` and `alpha` are selected on a held-out validation split, never on
the test set.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A CUDA GPU is used automatically if available, otherwise the code runs
on CPU (much slower). The DINOv2 backbone is downloaded through
`torch.hub` on the first run, so internet access is needed once.

## Running

Each dataset has its own standalone script. Run them from this folder:

| Dataset | Modality | Script |
|---|---|---|
| CT-Casting | X-ray CT of aluminium castings | `src/casting_dinov2_cad.py` |
| NRSD-MN | Rail surface visual inspection | `src/nrsd_dinov2_cad.py` |
| ML-NDT | Phased-array ultrasonic B-scans | `src/mlndt_dinov2_cad.py` |
| Airbus | Helicopter accelerometer signals (STFT) | `src/airbus_dinov2_cad.py` |

```bash
python src/mlndt_dinov2_cad.py
```

Each script reports sensitivity, specificity, F1 and AUROC for:

- a fixed-threshold reconstruction baseline,
- a fixed-threshold memory-bank baseline,
- the fused score under CAD.

Outputs (score arrays, ROC and p-value plots, explainability figures)
are written to `results/<dataset>/`.

## Notes

- The scripts share the same structure (data loading, DINOv2 features,
  reconstruction head, memory bank, fusion, CAD, evaluation), but each
  dataset needs its own preprocessing (patch extraction for NRSD-MN,
  STFT for Airbus, the `.bins` format for ML-NDT). They are kept as
  separate files on purpose, so that each dataset can be read on its
  own.
- Airbus works at patch level (120 STFT patches per one-minute series)
  and the patch scores are aggregated to give one decision per series.
- The Airbus test file is named `dfvalid .h5` (with a space) in the
  original download, and the script expects this name.
