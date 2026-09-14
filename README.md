# DINOv2 + Conformal Anomaly Detection (ML-NDT)

Code for the feature-based (DINOv2) branch of a Conformal Anomaly
Detection (CAD) framework for unsupervised industrial defect detection,
applied here to the ML-NDT ultrasonic weld dataset.

This repository currently releases the pipeline for **one** dataset
(ML-NDT) as a first, minimal, reproducible example. The remaining
datasets used in the paper will be added as the code is cleaned up for
release.

## Method

Frozen DINOv2 (ViT-S/14) patch features are compared against a
memory bank of defect-free local features (nearest-neighbor distance),
combined with a lightweight reconstruction score from the same frozen
features, into a single non-conformity score. Conformal Anomaly
Detection then turns that score into a $p$-value against a held-out
calibration set of normal samples, and a sample is flagged as
anomalous when its $p$-value falls below a chosen significance level
$\alpha$.

The fusion weight between the two scores and the significance level
$\alpha$ are both selected on a held-out validation split, never on
the test set.

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available, otherwise the pipeline
falls back to CPU (much slower). The DINOv2 backbone is downloaded
automatically via `torch.hub` on first run (needs internet access
once).

## Data

This repository does not include the dataset. Expected layout:

```
data/ML-NDT/data/
  training/    # *.bins (raw B-scans) + *.labels
  validation/  # *.bins (raw B-scans) + *.labels
```

**Data availability:** the dataset is described in Virkkunen, Koskinen,
Jessen-Juhler and Rinta-Aho, "Augmented Ultrasonic Data for Machine
Learning," *Journal of Nondestructive Evaluation*, vol. 40, no. 1,
2021 (DOI: 10.1007/s10921-020-00739-5). [ADD DOWNLOAD LINK HERE once
confirmed]. The local folder name `ML-NDT-master` used during
development suggests the data ships as a GitHub repository archive;
verify the exact source before pointing readers to it.

## Running

```bash
python3 src/mlndt_dinov2_cad.py
```

This trains the reconstruction head, builds the memory bank, tunes the
fusion weight and significance level on the validation split, and
reports sensitivity, specificity, F1 and AUROC for:
- a fixed-threshold reconstruction baseline,
- a fixed-threshold memory-bank baseline,
- the fused score under Conformal Anomaly Detection.

Results and diagnostic plots are written to `results/`.

## License

See `LICENSE`.
