# CAD-NDT: Conformal Anomaly Detection for Industrial Defect Detection

This repository contains the code for the paper *Unsupervised Fault
Detection in Industrial Images with Conformal Anomaly Detection and
Variational Autoencoders*.

Both pipelines follow the same idea: a model is trained on normal
(defect-free) data only, it gives each test sample a non-conformity
score, and Conformal Anomaly Detection (CAD) turns this score into a
p-value using a held-out calibration set of normal samples. A sample is
flagged as anomalous when its p-value is below a significance level
`alpha`.

The repository has two independent folders, one for each pipeline:

| Folder | Pipeline | Datasets | Framework |
|---|---|---|---|
| [`VAE-CAD/`](VAE-CAD/) | VAE trained on the images, CAD with Wasserstein-1 distance between residual histograms | Airbus | TensorFlow |
| [`DINOv2-CAD/`](DINOv2-CAD/) | Frozen DINOv2 features + patch memory bank + small reconstruction head, CAD on the fused score | ML-NDT | PyTorch |

Each folder has its own `README.md`, `requirements.txt` and data
instructions, and can be installed and run on its own. No dataset is
included in this repository: see the `data/README.md` file in each
folder for download links.

## Quick start

```bash
git clone https://github.com/ahmedhamdi-code/cad-ndt-defect-detection.git
cd cad-ndt-defect-detection

# VAE + CAD (Airbus)
cd VAE-CAD
pip install -r requirements.txt
python prepare_data.py
python run_experiments.py

# DINOv2 + CAD (ML-NDT)
cd ../DINOv2-CAD
pip install -r requirements.txt
python src/mlndt_dinov2_cad.py
```

We recommend a separate virtual environment for each folder, because
the two pipelines use different deep learning frameworks.

## License

See [`LICENSE`](LICENSE).
