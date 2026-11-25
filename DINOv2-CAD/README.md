# DINOv2-CAD

This folder provides the feature-based (DINOv2) pipeline of the paper,
applied to the ML-NDT ultrasonic weld dataset. It uses frozen DINOv2
features, a patch-level memory bank and a small reconstruction head,
and wraps the fused score in Conformal Anomaly Detection (CAD).

No dataset is included. Download instructions are available in
[`data/README.md`](data/README.md).

## Pipeline

1. A frozen DINOv2-small (ViT-S/14) extracts features from normal
   (defect-free) B-scans.
2. These features fill a memory bank. A test sample is scored by the
   distance to its nearest neighbours in the bank.
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

From this folder:

```bash
python src/mlndt_dinov2_cad.py
```

The script reports sensitivity, specificity, F1 and AUROC for:

- a fixed-threshold reconstruction baseline,
- a fixed-threshold memory-bank baseline,
- the fused score under CAD.

Results and diagnostic plots are written to `src/results/`.
