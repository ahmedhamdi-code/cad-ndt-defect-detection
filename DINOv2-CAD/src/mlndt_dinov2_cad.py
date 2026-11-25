"""
ML-NDT Ultrasonic B-scan Welds - DINOv2 + Memory Bank + CAD
------------------------------------------------------------------------
Pipeline:
  256x256 uint16 B-scan images (phased-array ultrasonic data)
  -> normalize to [0,1] -> replicate to RGB
  -> frozen DINOv2-small (ViT-S/14, 768D features, pre-cached)
  -> VAE decoder (256x256 reconstruction)
  -> p95 pixel residual per image
  -> memory bank KNN score per image
  -> ensemble non-conformity score -> CAD p-value -> anomaly label

Expected data layout (see data/README.md):
  data/ML-NDT/data/training/*.bins, *.labels
  data/ML-NDT/data/validation/*.bins, *.labels

Run: python3 src/mlndt_dinov2_cad.py
"""

import random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

# ── Config ─────────────────────────────────────────────────────────────────────
SEED       = 42
# repo_root/src/<this file> -> one level up is the repo root
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / 'data' / 'ML-NDT' / 'data'
TRAIN_DIR  = DATA_DIR / 'training'
VAL_DIR    = DATA_DIR / 'validation'
OUT_DIR    = REPO_ROOT / 'results' / 'mlndt'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SZ     = 256          # B-scan size

# VAE
VAE_EPOCHS = 20
VAE_LR     = 5e-4
VAE_BATCH  = 32
LATENT_DIM = 384
SSIM_W     = 0.7
KL_BETA    = 0.5

# CAD
CAL_SPLIT  = 0.1
CAD_ALPHA  = 0.02

# Memory bank
MAX_BANK_PRETOKENS = 40000
MAX_BANK_SIZE      = 8000
K_NEIGHBORS        = 3

ENSEMBLE_W = 0.9

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_bins(directory):
    """
    Load all .bins files in directory.
    Each file: (100, 256, 256) uint16 B-scans.
    Returns:
        images : (N, 256, 256) float32 normalised to [0, 1]
        labels : (N,) int32  0=normal  1=anomal
    """
    directory = Path(directory)
    bin_files = sorted([f for f in listdir(directory)
                        if isfile(join(directory, f)) and f.endswith('.bins')])
    all_imgs, all_labels = [], []
    for fname in tqdm(bin_files, desc=f'  loading {directory.name}', leave=False):
        raw = np.fromfile(directory / fname, dtype=np.uint16).astype(np.float32)
        imgs = raw.reshape(-1, IMG_SZ, IMG_SZ)       # (100, 256, 256)

        # Per-image normalisation to [0, 1]
        mn = imgs.min(axis=(1, 2), keepdims=True)
        mx = imgs.max(axis=(1, 2), keepdims=True)
        imgs = (imgs - mn) / (mx - mn + 1e-9)

        label_file = directory / fname.replace('.bins', '.labels')
        lab = np.loadtxt(label_file)
        if lab.ndim == 1:           # single row edge case
            lab = lab.reshape(1, -1)
        labels = lab[:, 0].astype(np.int32)

        all_imgs.append(imgs)
        all_labels.append(labels)

    return np.concatenate(all_imgs), np.concatenate(all_labels)


print('\n── Step 1: Loading ML-NDT data ──')
X_train_all, y_train_all = load_bins(TRAIN_DIR)
X_test,      y_test      = load_bins(VAL_DIR)

X_normal = X_train_all[y_train_all == 0]
# NEW: defective-labeled training B-scans -- previously discarded entirely
# (only normal training B-scans were ever used). Disjoint from the test set.
# Used ONLY to build a validation set for w/alpha selection -- test set
# stays untouched.
X_val_defect_pool = X_train_all[y_train_all == 1]
print(f'Train total: {len(X_train_all)}  normal: {len(X_normal)}  anomal: {(y_train_all==1).sum()}')
print(f'Test  total: {len(X_test)}       normal: {(y_test==0).sum()}  anomal: {(y_test==1).sum()}')

# 3-way split of normal train data: fit / calib / val-normal
n_fit   = int(0.8 * len(X_normal))
n_calib = int(0.1 * len(X_normal))
X_fit      = X_normal[:n_fit]
X_calib    = X_normal[n_fit:n_fit+n_calib]
X_val_norm = X_normal[n_fit+n_calib:]

rng = np.random.default_rng(SEED)
n_val_defect   = min(len(X_val_defect_pool), max(len(X_val_norm), 50))
val_defect_idx = rng.choice(len(X_val_defect_pool), n_val_defect, replace=False)
X_val_defect   = X_val_defect_pool[val_defect_idx]

X_val = np.concatenate([X_val_norm, X_val_defect])
y_val = np.array([0]*len(X_val_norm) + [1]*len(X_val_defect), dtype=np.int32)

print(f'Fit: {len(X_fit)} | Calib: {len(X_calib)} | '
      f'Val: {len(X_val)} (normal={len(X_val_norm)} defect={len(X_val_defect)})')

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def bscan_to_pil(arr):
    """(256, 256) float32 [0,1] → PIL RGB (grayscale replicated to 3 channels)."""
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert('RGB')

tf_dino = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_dec = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

aug = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.RandomResizedCrop(IMG_SZ, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])

# ══════════════════════════════════════════════════════════════════════════════
# 3. SSIM LOSS
# ══════════════════════════════════════════════════════════════════════════════

def _gauss_kernel(size=11, sigma=1.5, channels=3):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x**2 / (2 * sigma**2)); g /= g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return k.expand(channels, 1, size, size)

_KERNEL = None

def ssim_loss(x, y, window=11, sigma=1.5):
    global _KERNEL
    C = x.shape[1]
    if _KERNEL is None or _KERNEL.shape[0] != C:
        _KERNEL = _gauss_kernel(window, sigma, C).to(x.device)
    pad = window // 2
    mu_x  = F.conv2d(x,   _KERNEL, padding=pad, groups=C)
    mu_y  = F.conv2d(y,   _KERNEL, padding=pad, groups=C)
    mu_xx = F.conv2d(x*x, _KERNEL, padding=pad, groups=C) - mu_x**2
    mu_yy = F.conv2d(y*y, _KERNEL, padding=pad, groups=C) - mu_y**2
    mu_xy = F.conv2d(x*y, _KERNEL, padding=pad, groups=C) - mu_x*mu_y
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y + C1)*(2*mu_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1)*(mu_xx + mu_yy + C2))
    return 1 - ssim_map.mean()

def combined_loss(recon, target, mu, lv):
    rec = SSIM_W * ssim_loss(recon, target) + (1 - SSIM_W) * F.mse_loss(recon, target)
    kl  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return rec + KL_BETA * kl

# ══════════════════════════════════════════════════════════════════════════════
# 4. DINOv2
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 2: Loading DINOv2-small ──')
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
dino.eval().to(DEVICE)
for p in dino.parameters():
    p.requires_grad = False

_probe = torch.zeros(1, 3, 224, 224, device=DEVICE)
try:
    _out      = dino.forward_features(_probe)
    USE_PATCH = 'x_norm_patchtokens' in _out
    del _out
except Exception:
    USE_PATCH = False

DINO_DIM = 384 * (2 if USE_PATCH else 1)
print(f'DINOv2-small  patch-token API: {USE_PATCH}  feature dim: {DINO_DIM}')

@torch.no_grad()
def dino_features(x):
    if USE_PATCH:
        out = dino.forward_features(x)
        return torch.cat([out['x_norm_clstoken'],
                          out['x_norm_patchtokens'].mean(1)], dim=1)
    return dino(x)

@torch.no_grad()
def dino_patch_tokens(x):
    if USE_PATCH:
        return dino.forward_features(x)['x_norm_patchtokens']
    return dino(x).unsqueeze(1)

# ══════════════════════════════════════════════════════════════════════════════
# 5. DINOv2-VAE  (decoder 4→8→16→32→64→128→256 for 256×256 output)
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu      = nn.Linear(DINO_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(DINO_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),          nn.LeakyReLU(0.2),
            nn.Linear(512, 512 * 4 * 4),         nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 4, 4)),
            # 4 → 8 → 16 → 32 → 64 → 128 → 256
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,   32, 4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,   16, 4, stride=2, padding=1), nn.BatchNorm2d(16),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16,    3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def encode_feat(self, feat):
        return self.fc_mu(feat), self.fc_log_var(feat).clamp(-10, 10)

    def encode(self, x_dino):
        return self.encode_feat(dino_features(x_dino).detach())

    def decode(self, z):
        return self.decoder(z)

    def forward_feat(self, feat):
        mu, lv = self.encode_feat(feat)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


model = DINOv2VAE().to(DEVICE)
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# ══════════════════════════════════════════════════════════════════════════════
# 5b. PRE-CACHE DINOv2 FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def cache_features(imgs, desc='cache'):
    """Returns (N, DINO_DIM) float32 — DINOv2 run once per image."""
    all_feats = []
    for i in tqdm(range(0, len(imgs), 128), desc=f'  {desc}', leave=False):
        batch = imgs[i:i+128]
        x = torch.stack([tf_dino(bscan_to_pil(p)) for p in batch]).to(DEVICE)
        all_feats.append(dino_features(x).cpu())
    return torch.cat(all_feats)

print('\n── Step 3: Pre-caching DINOv2 features ──')
feat_fit   = cache_features(X_fit,   'fit feats')
feat_calib = cache_features(X_calib, 'calib feats')
feat_test  = cache_features(X_test,  'test feats')
feat_val   = cache_features(X_val,   'val feats')
print(f'  Cached: fit={feat_fit.shape}  calib={feat_calib.shape}  test={feat_test.shape}  val={feat_val.shape}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class CachedFeatDS(Dataset):
    def __init__(self, feats, imgs):
        self.feats = feats
        self.imgs  = imgs

    def __len__(self): return len(self.feats)

    def __getitem__(self, i):
        img = aug(bscan_to_pil(self.imgs[i]))
        return self.feats[i], tf_dec(img)


fit_loader = DataLoader(CachedFeatDS(feat_fit, X_fit),
                        batch_size=VAE_BATCH, shuffle=True,
                        num_workers=4, pin_memory=True)

opt = torch.optim.AdamW(model.parameters(), lr=VAE_LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=1, eta_min=1e-6)

print(f'\n── Step 4: Training DINOv2-VAE ({VAE_EPOCHS} epochs) ──')
for epoch in range(1, VAE_EPOCHS + 1):
    model.train()
    total = 0.0
    for feat, x_t in fit_loader:
        feat, x_t = feat.to(DEVICE), x_t.to(DEVICE)
        recon, mu, lv = model.forward_feat(feat)
        loss = combined_loss(recon, x_t, mu, lv)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    total /= len(fit_loader)
    sch.step()
    if epoch % 10 == 0 or epoch == 1:
        print(f'  Epoch {epoch:3d}/{VAE_EPOCHS} | loss={total:.5f} | lr={opt.param_groups[0]["lr"]:.1e}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. VAE RESIDUALS
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(cached_feats, imgs):
    """Returns (N,) p95 pixel residual per image using cached features."""
    model.eval()
    res = []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), 128), desc='  residuals', leave=False):
            feat  = cached_feats[i:i+128].to(DEVICE)
            x_t   = torch.stack([tf_dec(bscan_to_pil(p))
                                  for p in imgs[i:i+128]]).to(DEVICE)
            mu, _ = model.encode_feat(feat)
            recon = model.decode(mu)
            mae   = (x_t - recon).abs().mean(dim=1)   # (B, H, W)
            res.append(np.percentile(mae.cpu().numpy(), 95, axis=(1, 2)))
    r = np.concatenate(res)
    if np.isnan(r).any():
        r = np.where(np.isnan(r), np.nanmax(r), r)
    return r

print('\n── Step 5: Computing VAE residuals ──')
res_fit   = compute_residuals(feat_fit,   X_fit)
res_calib = compute_residuals(feat_calib, X_calib)
res_test  = compute_residuals(feat_test,  X_test)
res_val   = compute_residuals(feat_val,   X_val)
print(f'  Fit mean residual: {res_fit.mean():.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# 8. DINOv2 PATCH-TOKEN MEMORY BANK
# ══════════════════════════════════════════════════════════════════════════════

def _all_patch_tokens(imgs, desc='tokens'):
    all_toks = []
    for i in tqdm(range(0, len(imgs), 64), desc=f'  {desc}', leave=False):
        batch = imgs[i:i+64]
        x     = torch.stack([tf_dino(bscan_to_pil(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(x)
        toks  = F.normalize(toks.flatten(0, 1).float(), dim=1)
        all_toks.append(toks.cpu())
    return torch.cat(all_toks)

print('\n── Step 6: Building DINOv2 memory bank ──')
fit_tokens = _all_patch_tokens(X_fit, 'fit tokens')

if len(fit_tokens) > MAX_BANK_PRETOKENS:
    idx_pre = torch.randperm(len(fit_tokens))[:MAX_BANK_PRETOKENS]
    pool    = fit_tokens[idx_pre]
else:
    pool = fit_tokens

n_bank = min(MAX_BANK_SIZE, max(1, int(len(pool) * 0.2)))
print(f'  Token pool: {len(pool)} → greedy coreset: {n_bank}')

sel   = [random.randrange(len(pool))]
dists = torch.cdist(pool[sel[-1]].unsqueeze(0), pool)[0]
for _ in tqdm(range(n_bank - 1), desc='  coreset', leave=False):
    sel.append(int(dists.argmax()))
    dists = torch.minimum(dists, torch.cdist(pool[sel[-1]].unsqueeze(0), pool)[0])

bank = pool[sel].to(DEVICE)
print(f'  Memory bank: {bank.shape}')

def knn_scores(imgs, desc='knn'):
    scores = []
    for i in tqdm(range(0, len(imgs), 32), desc=f'  {desc}', leave=False):
        batch = imgs[i:i+32]
        x     = torch.stack([tf_dino(bscan_to_pil(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(x)
        toks  = F.normalize(toks.float(), dim=-1)
        B, N, D = toks.shape
        flat  = toks.reshape(B * N, D)
        d     = torch.cdist(flat, bank)
        knn_d = d.topk(K_NEIGHBORS, dim=1, largest=False).values.mean(1)
        knn_d = knn_d.reshape(B, N).max(1).values
        scores.append(knn_d.cpu().numpy())
    return np.concatenate(scores)

print('\n── Step 7: Memory bank scoring ──')
bank_fit   = knn_scores(X_fit,   'fit knn')
bank_calib = knn_scores(X_calib, 'calib knn')
bank_test  = knn_scores(X_test,  'test knn')
bank_val   = knn_scores(X_val,   'val knn')

# ══════════════════════════════════════════════════════════════════════════════
# 9. ENSEMBLE NCS
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 8: NCS & Ensemble ──')

ncs_vae_fit, ncs_vae_calib, ncs_vae_test, ncs_vae_val      = res_fit, res_calib, res_test, res_val
ncs_bank_fit, ncs_bank_calib, ncs_bank_test, ncs_bank_val  = bank_fit, bank_calib, bank_test, bank_val

def znorm(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-9)

vae_fit_n   = znorm(ncs_vae_fit,   ncs_vae_fit)
vae_calib_n = znorm(ncs_vae_calib, ncs_vae_fit)
vae_test_n  = znorm(ncs_vae_test,  ncs_vae_fit)
vae_val_n   = znorm(ncs_vae_val,   ncs_vae_fit)

bank_fit_n   = znorm(ncs_bank_fit,   ncs_bank_fit)
bank_calib_n = znorm(ncs_bank_calib, ncs_bank_fit)
bank_test_n  = znorm(ncs_bank_test,  ncs_bank_fit)
bank_val_n   = znorm(ncs_bank_val,   ncs_bank_fit)

# w is swept on the VALIDATION set (held-out normal + previously-unused
# defective training pool), NEVER on the test set -- test set is only
# touched once, at final reporting, below.
print('  Sweeping ensemble weight (on validation set, not test) ...')
best_auroc_ens, best_ew = 0.0, ENSEMBLE_W
for ew in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ncs = (1 - ew) * vae_val_n + ew * bank_val_n
    a   = roc_auc_score(y_val, ncs)
    print(f'    ew={ew:.1f} → val AUROC={a:.4f}')
    if a > best_auroc_ens:
        best_auroc_ens, best_ew = a, ew

print(f'  Best ensemble weight: {best_ew}  (val AUROC={best_auroc_ens:.4f})')

ncs_calib = (1 - best_ew) * vae_calib_n + best_ew * bank_calib_n
ncs_test  = (1 - best_ew) * vae_test_n  + best_ew * bank_test_n
ncs_val   = (1 - best_ew) * vae_val_n   + best_ew * bank_val_n

# ══════════════════════════════════════════════════════════════════════════════
# 10. CONFORMAL ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 9: Conformal Anomaly Detection ──')
n_cal    = len(ncs_calib)
p_values = np.array([(np.sum(ncs_calib >= s) + 1) / (n_cal + 1) for s in ncs_test])

# alpha is swept on the VALIDATION set's p-values (computed against the SAME
# calibration reference used for the real test set), NEVER on the test set.
p_values_val = np.array([(np.sum(ncs_calib >= s) + 1) / (n_cal + 1) for s in ncs_val])

best_f1_cad, best_alpha = 0.0, CAD_ALPHA
for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    y_pred_val = (p_values_val < alpha).astype(int)
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    print(f'    alpha={alpha:.2f} → val F1={f1:.4f}')
    if f1 > best_f1_cad:
        best_f1_cad, best_alpha = f1, alpha

print(f'  Best alpha: {best_alpha}  (val F1={best_f1_cad:.4f})')

# Final decision applied to the untouched test set only
y_pred_cad = (p_values < best_alpha).astype(int)
y_pred_002 = (p_values < 0.02).astype(int)

# ── Save scores for post-hoc conformal experiments ───────────────────────────
_save = OUT_DIR / 'mlndt_valsplit.npz'
np.savez(_save, ncs_calib=ncs_calib, ncs_test=ncs_test,
         p_values=p_values, y_test=y_test,
         n_cal=np.int64(n_cal), best_alpha=np.float64(best_alpha),
         best_ew=np.float64(best_ew))
print(f'Scores saved → {_save}')

# ══════════════════════════════════════════════════════════════════════════════
# 11. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, scores, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens  = tp / (tp + fn + 1e-9)
    spec  = tn / (tn + fp + 1e-9)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, scores)
    print(f'  {name}')
    print(f'    F1={f1:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  AUROC={auroc:.4f}')
    print(f'    TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    return dict(f1=f1, sens=sens, spec=spec, auroc=auroc)

print('\n── Step 10: Results ──')

thr_q60    = np.percentile(res_fit,   60)
bank_thr95 = np.percentile(bank_fit,  95)

res_vae    = metrics(y_test, (res_test  > thr_q60).astype(int),
                     res_test,   'DINOv2-VAE @ Q60 (thesis thr)')
res_bank   = metrics(y_test, (bank_test > bank_thr95).astype(int),
                     bank_test,  'Memory Bank only (p95 thr)')
res_cad002 = metrics(y_test, y_pred_002, ncs_test,
                     'Ensemble + CAD (α=0.02 thesis)')
res_best   = metrics(y_test, y_pred_cad, ncs_test,
                     f'Ensemble + CAD (α={best_alpha:.2f} best F1)')

print('\n' + '='*80)
print(f'  {"Method":<42} {"F1":>6} {"Sens":>6} {"Spec":>6} {"AUROC":>7}')
print('  ' + '-'*77)
print(f'  {"Thesis reference (ML-NDT Q60)":<42} {"0.9159":>6} {"0.8596":>6} {"0.9767":>6} {"  N/A":>7}')
print(f'  {"DINOv2-VAE @ Q60":<42}'
      f' {res_vae["f1"]:>6.4f} {res_vae["sens"]:>6.4f} {res_vae["spec"]:>6.4f} {res_vae["auroc"]:>7.4f}')
print(f'  {"Memory Bank only":<42}'
      f' {res_bank["f1"]:>6.4f} {res_bank["sens"]:>6.4f} {res_bank["spec"]:>6.4f} {res_bank["auroc"]:>7.4f}')
print(f'  {"Ensemble+CAD α=0.02 (thesis α)":<42}'
      f' {res_cad002["f1"]:>6.4f} {res_cad002["sens"]:>6.4f} {res_cad002["spec"]:>6.4f} {res_cad002["auroc"]:>7.4f}')
print(f'  {"Ensemble+CAD α={:.2f} (best F1)".format(best_alpha):<42}'
      f' {res_best["f1"]:>6.4f} {res_best["sens"]:>6.4f} {res_best["spec"]:>6.4f} {res_best["auroc"]:>7.4f}')
print('='*80)

# ══════════════════════════════════════════════════════════════════════════════
# 12. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for scores, label, color, ls in [
    (ncs_vae_test,  f'VAE p95 (AUROC={roc_auc_score(y_test, ncs_vae_test):.3f})',   'steelblue', '-'),
    (ncs_bank_test, f'Mem. Bank (AUROC={roc_auc_score(y_test, ncs_bank_test):.3f})', 'orange',   '-'),
    (ncs_test,      f'Ensemble (AUROC={best_auroc_ens:.3f})',                         'tomato',   '--'),
]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls, label=label)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC — ML-NDT Ultrasonic Welds')
ax.legend(fontsize=8)

ax = axes[1]
bins = np.linspace(0, 1, 50)
ax.hist(p_values[y_test == 0], bins=bins, alpha=0.6, color='steelblue', density=True, label='Normal')
ax.hist(p_values[y_test == 1], bins=bins, alpha=0.6, color='tomato',    density=True, label='Anomal')
ax.axvline(best_alpha, color='black', linestyle='--', label=f'α={best_alpha}')
ax.set_xlabel('p-value'); ax.set_title('CAD p-value distribution')
ax.legend()

ax = axes[2]
methods = ['Thesis\n(ML-NDT)', 'DINOv2\nVAE', 'Memory\nBank', 'Ensemble\n+CAD']
f1s    = [0.9159, res_vae['f1'],    res_bank['f1'],    res_best['f1']]
aurocs = [np.nan, res_vae['auroc'], res_bank['auroc'], res_best['auroc']]
x = np.arange(4); w = 0.3
b1 = ax.bar(x - w/2, f1s, w, color='steelblue', label='F1')
b2 = ax.bar([1+w/2, 2+w/2, 3+w/2], aurocs[1:], w, color='tomato', label='AUROC')
for bar, v in list(zip(b1, f1s)) + list(zip(b2, aurocs[1:])):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 1.15); ax.set_title('Performance vs Thesis Baseline'); ax.legend()

plt.tight_layout()
out = OUT_DIR / 'results_mlndt_dinov2_valsplit.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nPlot saved → {out}')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 13. EXPLAINABILITY — t-SNE + Anomaly Heatmaps
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom as nd_zoom

print('\n── Step 12: Explainability (t-SNE + Heatmaps) ──')

PATCH_SIDE = 16   # 224 / 14 = 16 patches per side for ViT-S/14

# ── t-SNE on DINOv2 features ────────────────────────────────────────────────
print('  Running t-SNE...')
n_tsne   = min(600, len(feat_test))
rng      = np.random.default_rng(SEED)
idx_vis  = rng.choice(len(feat_test), n_tsne, replace=False)
f2d      = TSNE(n_components=2, perplexity=30, random_state=SEED,
                max_iter=1000, init='pca').fit_transform(feat_test[idx_vis].numpy())
lbl_vis  = y_test[idx_vis]

# ── Per-patch KNN heatmap ───────────────────────────────────────────────────
@torch.no_grad()
def patch_heatmap(img_arr):
    x     = tf_dino(bscan_to_pil(img_arr)).unsqueeze(0).to(DEVICE)
    toks  = dino_patch_tokens(x)
    toks  = F.normalize(toks.float(), dim=-1)
    flat  = toks.reshape(-1, toks.shape[-1])
    d     = torch.cdist(flat, bank)
    knn_d = d.topk(K_NEIGHBORS, dim=1, largest=False).values.mean(1)
    hmap  = knn_d.reshape(PATCH_SIDE, PATCH_SIDE).cpu().numpy()
    return nd_zoom(hmap, IMG_SZ / PATCH_SIDE, order=1)

norm_idx   = np.where(y_test == 0)[0][:4]
anom_idx   = np.where(y_test == 1)[0][:4]
print('  Computing heatmaps...')
norm_hmaps = [patch_heatmap(X_test[i]) for i in norm_idx]
anom_hmaps = [patch_heatmap(X_test[i]) for i in anom_idx]

# ── Figure ──────────────────────────────────────────────────────────────────
fig_e = plt.figure(figsize=(22, 14))
gs    = gridspec.GridSpec(3, 4, figure=fig_e, hspace=0.45, wspace=0.3)

ax_t = fig_e.add_subplot(gs[0, :])
ax_t.scatter(f2d[lbl_vis == 0, 0], f2d[lbl_vis == 0, 1],
             c='steelblue', alpha=0.5, s=12, label='Normal')
ax_t.scatter(f2d[lbl_vis == 1, 0], f2d[lbl_vis == 1, 1],
             c='tomato', alpha=0.7, s=15, label='Anomal', zorder=3)
ax_t.set_title('t-SNE — DINOv2 Latent Space (ML-NDT Ultrasonic Welds)', fontsize=12)
ax_t.set_xlabel('t-SNE 1'); ax_t.set_ylabel('t-SNE 2')
ax_t.legend(fontsize=10)

vmax = max(h.max() for h in norm_hmaps + anom_hmaps)

for col, (idx, hmap) in enumerate(zip(norm_idx, norm_hmaps)):
    ax = fig_e.add_subplot(gs[1, col])
    ax.imshow(X_test[idx], cmap='gray')
    im = ax.imshow(hmap, cmap='hot', alpha=0.5, vmin=0, vmax=vmax)
    ax.set_title(f'Normal #{idx}', fontsize=9)
    ax.axis('off')

for col, (idx, hmap) in enumerate(zip(anom_idx, anom_hmaps)):
    ax = fig_e.add_subplot(gs[2, col])
    ax.imshow(X_test[idx], cmap='gray')
    im = ax.imshow(hmap, cmap='hot', alpha=0.5, vmin=0, vmax=vmax)
    ax.set_title(f'Anomal #{idx}', fontsize=9, color='tomato')
    ax.axis('off')

cb_ax = fig_e.add_axes([0.92, 0.07, 0.015, 0.45])
plt.colorbar(im, cax=cb_ax, label='KNN distance (anomaly score)')
fig_e.suptitle('DINOv2 Memory Bank Explainability — ML-NDT', fontsize=13, fontweight='bold')
out_exp = OUT_DIR / 'results_mlndt_dinov2_valsplit_explainability.png'
plt.savefig(out_exp, dpi=150, bbox_inches='tight')
plt.close(fig_e)
print(f'  Explainability plot → {out_exp}')

print('Done.')
