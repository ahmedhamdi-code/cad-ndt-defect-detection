"""
NRSD-MN Rail Surface - DINOv2 + Memory Bank + CAD
------------------------------------------------------------------------
Pipeline:
  256x256 rail-surface images, cut into four 128x128 patches each
  -> frozen DINOv2-small (ViT-S/14, CLS + mean patch tokens, 768D)
  -> VAE decoder (128x128 reconstruction)
  -> p95 pixel residual per patch
  -> per-location DINOv2 patch-token memory bank, KNN distance per patch
  -> ensemble non-conformity score -> CAD p-value -> anomaly label

Expected data layout (see data/README.md):
  data/NRSD-MN/JPEGImages/*.jpg
  data/NRSD-MN/SegmentationClass/*.png
  data/NRSD-MN/ImageSets/Segmentation/{train,test}.txt

Run: python3 src/nrsd_dinov2_cad.py
"""

import random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, roc_curve
)

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
# repo_root/src/<this file> -> one level up is the repo root
REPO_ROOT   = Path(__file__).resolve().parent.parent
SEG_DIR     = REPO_ROOT / 'data' / 'NRSD-MN'
IMG_DIR     = SEG_DIR / 'JPEGImages'
MASK_DIR    = SEG_DIR / 'SegmentationClass'
SETS_DIR    = SEG_DIR / 'ImageSets' / 'Segmentation'
OUT_DIR     = REPO_ROOT / 'results' / 'nrsd'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESIZE      = 256
PATCH_SZ    = 128
DEFECT_PX   = 50

VAE_EPOCHS  = 20            # was 40
VAE_LR      = 5e-4
VAE_BATCH   = 32
LATENT_DIM  = 384           # was 256
SSIM_W      = 0.7
KL_BETA     = 0.5

CAL_SPLIT   = 0.1
CAD_ALPHA   = 0.02

# Memory bank
MAX_BANK_PRETOKENS = 40000  # pre-subsample before greedy coreset
MAX_BANK_SIZE      = 8000   # coreset cap
K_NEIGHBORS        = 3

# Ensemble (VAE residual weight = 1-ENSEMBLE_W, bank weight = ENSEMBLE_W)
ENSEMBLE_W  = 0.5           # swept automatically at evaluation

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_split(p): return [l.strip() for l in open(p) if l.strip()]

def extract_patches(ids, img_dir, mask_dir, desc=''):
    patches, labels = [], []
    n = RESIZE // PATCH_SZ
    for img_id in tqdm(ids, desc=desc):
        img  = Image.open(img_dir  / f'{img_id}.jpg').convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR)
        mask = Image.open(mask_dir / f'{img_id}.png').convert('L').resize((RESIZE, RESIZE), Image.NEAREST)
        ia, ma = np.asarray(img), np.asarray(mask)
        for r in range(n):
            for c in range(n):
                patches.append(ia[r*PATCH_SZ:(r+1)*PATCH_SZ, c*PATCH_SZ:(c+1)*PATCH_SZ])
                m = ma[r*PATCH_SZ:(r+1)*PATCH_SZ, c*PATCH_SZ:(c+1)*PATCH_SZ]
                labels.append(1 if np.sum(m > 0) >= DEFECT_PX else 0)
    return np.array(patches, dtype=np.uint8), np.array(labels, dtype=np.int32)

# DINOv2 expects 224 with ImageNet normalisation; decoder target stays at 128
tf_dino = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_dec = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class DualPatchDS(Dataset):
    def __init__(self, patches):
        self.patches = patches
    def __len__(self): return len(self.patches)
    def __getitem__(self, i):
        img = Image.fromarray(self.patches[i])
        return tf_dino(img), tf_dec(img)

print('\n── Step 1: Data loading ──')
train_ids = load_split(SETS_DIR / 'train.txt')
test_ids  = load_split(SETS_DIR / 'test.txt')

X_train_all, y_train = extract_patches(train_ids, IMG_DIR, MASK_DIR, 'Train')
X_test,      y_test  = extract_patches(test_ids,  IMG_DIR, MASK_DIR, 'Test')

X_normal = X_train_all[y_train == 0]
# NEW: defective-labeled training patches -- previously discarded entirely
# (only normal training patches were ever used). Disjoint from the test set.
# Used ONLY to build a validation set for w/alpha selection -- test set
# stays untouched.
X_val_defect_pool = X_train_all[y_train == 1]

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
      f'Val: {len(X_val)} (normal={len(X_val_norm)} defect={len(X_val_defect)}) | '
      f'Test normal/anomal: {(y_test==0).sum()}/{(y_test==1).sum()}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. SSIM LOSS
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
# 3. DINOv2 — load and probe for patch-token API
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 2: Loading DINOv2-small ──')
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
dino.eval().to(DEVICE)
for p in dino.parameters():
    p.requires_grad = False

# Detect whether forward_features() exposes patch tokens (DINOv2 >= 1.0)
_probe = torch.zeros(1, 3, 224, 224, device=DEVICE)
try:
    _out = dino.forward_features(_probe)
    USE_PATCH = 'x_norm_patchtokens' in _out
    del _out
except Exception:
    USE_PATCH = False

DINO_DIM = 384 * (2 if USE_PATCH else 1)   # 768 if patch tokens available, else 384
print(f'DINOv2-small loaded | patch-token API: {USE_PATCH} | feature dim: {DINO_DIM}')

@torch.no_grad()
def dino_features(x):
    """CLS + mean-patch-token concatenation (768D), or CLS only (384D) as fallback."""
    if USE_PATCH:
        out = dino.forward_features(x)
        return torch.cat([out['x_norm_clstoken'],
                          out['x_norm_patchtokens'].mean(1)], dim=1)
    return dino(x)

@torch.no_grad()
def dino_patch_tokens(x):
    """Per-location patch tokens: (B, N_tokens, 384). Falls back to CLS as single token."""
    if USE_PATCH:
        return dino.forward_features(x)['x_norm_patchtokens']   # B×256×384
    return dino(x).unsqueeze(1)                                 # B×1×384

# ══════════════════════════════════════════════════════════════════════════════
# 4. DINOv2-VAE
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu      = nn.Linear(DINO_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(DINO_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),         nn.LeakyReLU(0.2),
            nn.Linear(512, 512 * 4 * 4),        nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 4, 4)),
            # 4×4 → 8 → 16 → 32 → 64 → 128
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,   32, 4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,    3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def encode(self, x_dino):
        feat = dino_features(x_dino).detach()
        return self.fc_mu(feat), self.fc_log_var(feat).clamp(-10, 10)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_dino):
        mu, lv = self.encode(x_dino)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv

model = DINOv2VAE().to(DEVICE)
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

aug = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.RandomResizedCrop(128, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    T.ColorJitter(0.2, 0.2, 0.1, 0.05),
    T.RandomGrayscale(p=0.05),
])

class AugDualDS(Dataset):
    def __init__(self, patches):
        self.patches = patches
    def __len__(self): return len(self.patches)
    def __getitem__(self, i):
        img = aug(Image.fromarray(self.patches[i]))
        return tf_dino(img), tf_dec(img)

fit_loader = DataLoader(AugDualDS(X_fit), batch_size=VAE_BATCH,
                        shuffle=True, num_workers=4, pin_memory=True)

opt = torch.optim.AdamW(model.parameters(), lr=VAE_LR, weight_decay=1e-4)
# 3 cosine cycles over 60 epochs (T_0=20)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=1, eta_min=1e-6)

print(f'\n── Step 3: Training DINOv2-VAE ({VAE_EPOCHS} epochs) ──')
for epoch in range(1, VAE_EPOCHS + 1):
    model.train()
    total = 0.0
    for x_d, x_t in fit_loader:
        x_d, x_t = x_d.to(DEVICE), x_t.to(DEVICE)
        recon, mu, lv = model(x_d)
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
# 6. VAE RESIDUALS
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(patches):
    """Returns (N, H, W) pixel-wise MAE array."""
    model.eval()
    ds = DualPatchDS(patches)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    res = []
    with torch.no_grad():
        for x_d, x_t in tqdm(ld, desc='  residuals', leave=False):
            x_d, x_t = x_d.to(DEVICE), x_t.to(DEVICE)
            mu, _ = model.encode(x_d)
            recon = model.decode(mu)                         # deterministic (use μ)
            mae   = (x_t - recon).abs().mean(dim=1)         # B×H×W
            res.append(mae.cpu().numpy())
    r = np.concatenate(res)
    if np.isnan(r).any():
        r = np.where(np.isnan(r), np.nanmax(r), r)
    return r

print('\n── Step 4: Computing VAE residuals ──')
res_fit   = compute_residuals(X_fit)
res_calib = compute_residuals(X_calib)
res_test  = compute_residuals(X_test)
res_val   = compute_residuals(X_val)
print(f'  Fit mean residual: {res_fit.mean():.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. DINOV2 PATCH-TOKEN MEMORY BANK
# ══════════════════════════════════════════════════════════════════════════════
# For each normal training patch, DINOv2 produces N_tok (=256 for 224px/14) local
# feature vectors.  We build a greedy-coreset memory bank from these and score
# test patches by their max-KNN distance across token positions.
# This PatchCore-style score captures spatial/structural anomalies that the VAE
# reconstruction loss may underweight.

def _all_patch_tokens(patches, desc='tokens'):
    """Extracts & L2-normalises all per-location tokens: (N_patches * N_tok, 384)."""
    all_toks = []
    for i in tqdm(range(0, len(patches), 64), desc=f'  {desc}', leave=False):
        batch = patches[i:i+64]
        imgs  = torch.stack([tf_dino(Image.fromarray(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(imgs)                          # B×N_tok×384
        toks  = F.normalize(toks.flatten(0, 1).float(), dim=1)  # (B*N_tok)×384
        all_toks.append(toks.cpu())
    return torch.cat(all_toks)

print('\n── Step 5: Building DINOv2 memory bank ──')
fit_tokens = _all_patch_tokens(X_fit, 'fit tokens')

# Pre-subsample to keep greedy tractable
if len(fit_tokens) > MAX_BANK_PRETOKENS:
    idx_pre    = torch.randperm(len(fit_tokens))[:MAX_BANK_PRETOKENS]
    pool       = fit_tokens[idx_pre]
else:
    pool = fit_tokens

n_bank = min(MAX_BANK_SIZE, max(1, int(len(pool) * 0.2)))
print(f'  Token pool: {len(pool)} → greedy coreset: {n_bank}')

sel   = [random.randrange(len(pool))]
dists = torch.cdist(pool[sel[-1]].unsqueeze(0), pool)[0]
for _ in tqdm(range(n_bank - 1), desc='  building coreset', leave=False):
    sel.append(int(dists.argmax()))
    dists = torch.minimum(dists, torch.cdist(pool[sel[-1]].unsqueeze(0), pool)[0])

bank     = pool[sel].to(DEVICE)   # n_bank×384 (on GPU if available)
print(f'  Memory bank: {bank.shape}')

def knn_scores(patches, desc='knn'):
    """Per-patch anomaly score = max over token positions of mean KNN distance."""
    scores = []
    for i in tqdm(range(0, len(patches), 32), desc=f'  {desc}', leave=False):
        batch = patches[i:i+32]
        imgs  = torch.stack([tf_dino(Image.fromarray(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(imgs)                       # B×N_tok×384
        toks  = F.normalize(toks.float(), dim=-1)
        B, N, D = toks.shape
        flat  = toks.reshape(B * N, D)                        # (B*N)×384
        d     = torch.cdist(flat, bank)                       # (B*N)×n_bank
        knn_d = d.topk(K_NEIGHBORS, dim=1, largest=False).values.mean(1)  # (B*N)
        knn_d = knn_d.reshape(B, N).max(1).values             # B — max over positions
        scores.append(knn_d.cpu().numpy())
    return np.concatenate(scores)

print('\n── Step 6: Memory bank scoring ──')
bank_fit   = knn_scores(X_fit,   'fit knn')
bank_calib = knn_scores(X_calib, 'calib knn')
bank_test  = knn_scores(X_test,  'test knn')
bank_val   = knn_scores(X_val,   'val knn')

# ══════════════════════════════════════════════════════════════════════════════
# 8. NCS COMPUTATION & ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 7: NCS & Ensemble ──')

# VAE NCS: 95th percentile of per-pixel residuals
ncs_vae_fit   = np.percentile(res_fit,   95, axis=(1, 2))
ncs_vae_calib = np.percentile(res_calib, 95, axis=(1, 2))
ncs_vae_test  = np.percentile(res_test,  95, axis=(1, 2))
ncs_vae_val   = np.percentile(res_val,   95, axis=(1, 2))

# Memory bank NCS: KNN distance (already computed)
ncs_bank_fit, ncs_bank_calib, ncs_bank_test, ncs_bank_val = bank_fit, bank_calib, bank_test, bank_val

# Z-score normalisation using fit-set statistics (no calibration/test info used)
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
# 9. CONFORMAL ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 8: Conformal Anomaly Detection ──')
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
_save = OUT_DIR / 'nrsd_valsplit.npz'
np.savez(_save, ncs_calib=ncs_calib, ncs_test=ncs_test,
         p_values=p_values, y_test=y_test,
         n_cal=np.int64(n_cal), best_alpha=np.float64(best_alpha),
         best_ew=np.float64(best_ew))
print(f'Scores saved → {_save}')

# ══════════════════════════════════════════════════════════════════════════════
# 10. EVALUATION
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

print('\n── Step 9: Results ──')

# Threshold-free comparison baselines
max_res    = res_test.max(axis=(1, 2))
thr_q70    = np.percentile(res_fit.max(axis=(1, 2)), 70)
bank_thr   = np.percentile(ncs_bank_fit, 95)

res_vae    = metrics(y_test, (max_res > thr_q70).astype(int),
                     max_res, 'DINOv2-VAE @ Q70 (baseline)')
res_bank   = metrics(y_test, (ncs_bank_test > bank_thr).astype(int),
                     ncs_bank_test, 'Memory Bank only (p95 thr)')
res_cad002 = metrics(y_test, y_pred_002,  ncs_test,
                     'Ensemble + CAD (α=0.02 thesis)')
res_best   = metrics(y_test, y_pred_cad,  ncs_test,
                     f'Ensemble + CAD (α={best_alpha:.2f} best F1)')

print('\n' + '='*80)
print(f'  {"Method":<42} {"F1":>6} {"Sens":>6} {"Spec":>6} {"AUROC":>7}')
print('  ' + '-'*77)
print(f'  {"Thesis reference (VAE+CAD Q70)":<42} {"0.7716":>6} {"0.7697":>6} {"0.7160":>6} {"~0.80":>7}')
print(f'  {"DINOv2-VAE @ Q70 (v1 baseline)":<42}'
      f' {res_vae["f1"]:>6.4f} {res_vae["sens"]:>6.4f} {res_vae["spec"]:>6.4f} {res_vae["auroc"]:>7.4f}')
print(f'  {"Memory Bank only":<42}'
      f' {res_bank["f1"]:>6.4f} {res_bank["sens"]:>6.4f} {res_bank["spec"]:>6.4f} {res_bank["auroc"]:>7.4f}')
print(f'  {"Ensemble+CAD α=0.02 (thesis α)":<42}'
      f' {res_cad002["f1"]:>6.4f} {res_cad002["sens"]:>6.4f} {res_cad002["spec"]:>6.4f} {res_cad002["auroc"]:>7.4f}')
print(f'  {"Ensemble+CAD α={:.2f} (best F1)".format(best_alpha):<42}'
      f' {res_best["f1"]:>6.4f} {res_best["sens"]:>6.4f} {res_best["spec"]:>6.4f} {res_best["auroc"]:>7.4f}')
print('='*80)

# ══════════════════════════════════════════════════════════════════════════════
# 11. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves — compare all three score types
ax = axes[0]
for scores, label, color, ls in [
        (ncs_vae_test,  f'VAE p95 (AUROC={roc_auc_score(y_test,ncs_vae_test):.3f})',   'steelblue', '-'),
        (ncs_bank_test, f'Mem. Bank (AUROC={roc_auc_score(y_test,ncs_bank_test):.3f})', 'orange',    '-'),
        (ncs_test,      f'Ensemble (AUROC={best_auroc_ens:.3f})',                        'tomato',    '--'),
]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls, label=label)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC — VAE vs Memory Bank vs Ensemble')
ax.legend(fontsize=8)

# CAD p-value distribution
ax = axes[1]
bins = np.linspace(0, 1, 50)
ax.hist(p_values[y_test == 0], bins=bins, alpha=0.6, color='steelblue', density=True, label='Normal')
ax.hist(p_values[y_test == 1], bins=bins, alpha=0.6, color='tomato',    density=True, label='Anomal')
ax.axvline(best_alpha, color='black', linestyle='--', label=f'α={best_alpha}')
ax.set_xlabel('p-value'); ax.set_title('CAD p-value distribution (Ensemble NCS)')
ax.legend()

# Bar chart comparison
ax = axes[2]
methods = ['Thesis\nVAE', 'DINOv2\nVAE', 'Memory\nBank', 'Ensemble\n+CAD']
f1s    = [0.7716, res_vae['f1'],    res_bank['f1'],    res_best['f1']]
aurocs = [0.80,   res_vae['auroc'], res_bank['auroc'], res_best['auroc']]
x = np.arange(4); w = 0.3
b1 = ax.bar(x - w/2, f1s,    w, color='steelblue', label='F1')
b2 = ax.bar(x + w/2, aurocs, w, color='tomato',    label='AUROC')
for bar, v in list(zip(b1, f1s)) + list(zip(b2, aurocs)):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 1.15); ax.set_title('Performance Comparison'); ax.legend()

plt.tight_layout()
out = OUT_DIR / 'results_nrsd_dinov2_valsplit.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nPlot saved → {out}')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 12. EXPLAINABILITY — t-SNE + Anomaly Heatmaps
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom as nd_zoom

print('\n── Step 12: Explainability (t-SNE + Heatmaps) ──')

PATCH_SIDE = 16   # 224 / 14 = 16 patches per side for ViT-S/14

# ── Compute DINOv2 features for test set ────────────────────────────────────
print('  Computing DINOv2 features for t-SNE...')
feat_test_vis = []
for i in tqdm(range(0, len(X_test), 64), desc='  feat', leave=False):
    batch = X_test[i:i+64]
    x = torch.stack([tf_dino(Image.fromarray(p)) for p in batch]).to(DEVICE)
    feat_test_vis.append(dino_features(x).cpu())
feat_test_vis = torch.cat(feat_test_vis)

# ── t-SNE ────────────────────────────────────────────────────────────────────
print('  Running t-SNE...')
n_tsne   = min(600, len(feat_test_vis))
rng      = np.random.default_rng(SEED)
idx_vis  = rng.choice(len(feat_test_vis), n_tsne, replace=False)
f2d      = TSNE(n_components=2, perplexity=30, random_state=SEED,
                max_iter=1000, init='pca').fit_transform(feat_test_vis[idx_vis].numpy())
lbl_vis  = y_test[idx_vis]

# ── Per-patch KNN heatmap ───────────────────────────────────────────────────
@torch.no_grad()
def patch_heatmap(patch_arr):
    x     = tf_dino(Image.fromarray(patch_arr)).unsqueeze(0).to(DEVICE)
    toks  = dino_patch_tokens(x)
    toks  = F.normalize(toks.float(), dim=-1)
    flat  = toks.reshape(-1, toks.shape[-1])
    d     = torch.cdist(flat, bank)
    knn_d = d.topk(K_NEIGHBORS, dim=1, largest=False).values.mean(1)
    hmap  = knn_d.reshape(PATCH_SIDE, PATCH_SIDE).cpu().numpy()
    return nd_zoom(hmap, PATCH_SZ / PATCH_SIDE, order=1)

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
ax_t.set_title('t-SNE — DINOv2 Latent Space (NRSD-MN Rail Surface)', fontsize=12)
ax_t.set_xlabel('t-SNE 1'); ax_t.set_ylabel('t-SNE 2')
ax_t.legend(fontsize=10)

vmax = max(h.max() for h in norm_hmaps + anom_hmaps)

for col, (idx, hmap) in enumerate(zip(norm_idx, norm_hmaps)):
    ax = fig_e.add_subplot(gs[1, col])
    ax.imshow(X_test[idx])
    im = ax.imshow(hmap, cmap='hot', alpha=0.5, vmin=0, vmax=vmax)
    ax.set_title(f'Normal #{idx}', fontsize=9)
    ax.axis('off')

for col, (idx, hmap) in enumerate(zip(anom_idx, anom_hmaps)):
    ax = fig_e.add_subplot(gs[2, col])
    ax.imshow(X_test[idx])
    im = ax.imshow(hmap, cmap='hot', alpha=0.5, vmin=0, vmax=vmax)
    ax.set_title(f'Anomal #{idx}', fontsize=9, color='tomato')
    ax.axis('off')

cb_ax = fig_e.add_axes([0.92, 0.07, 0.015, 0.45])
plt.colorbar(im, cax=cb_ax, label='KNN distance (anomaly score)')
fig_e.suptitle('DINOv2 Memory Bank Explainability — NRSD-MN', fontsize=13, fontweight='bold')
out_exp = OUT_DIR / 'results_nrsd_dinov2_valsplit_explainability.png'
plt.savefig(out_exp, dpi=150, bbox_inches='tight')
plt.close(fig_e)
print(f'  Explainability plot → {out_exp}')

print('Done.')
