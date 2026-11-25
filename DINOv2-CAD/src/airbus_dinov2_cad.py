"""
Airbus Helicopter Accelerometer - DINOv2 + Memory Bank + CAD
------------------------------------------------------------------------
Pipeline:
  1-minute accelerometer sequences -> STFT spectrogram -> split into
  120 sub-images of 64x64 each, so every sub-image is scored on its own
  -> frozen DINOv2-small (ViT-S/14, 768D features, pre-cached)
  -> VAE decoder (64x64 reconstruction)
  -> p95 pixel residual per sub-image
  -> memory bank KNN score per sub-image
  -> ensemble non-conformity score -> CAD p-value per sub-image
  -> a sequence is flagged if the minimum p-value across its 120
     sub-images falls below alpha

Expected data layout (see data/README.md):
  data/Airbus/dftrain.h5
  data/Airbus/dfvalid .h5   (note the space before .h5, matches the
                             original file name from the dataset)
  data/Airbus/dfvalid_groundtruth.csv

Run: python3 src/airbus_dinov2_cad.py
"""

import random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import stft as scipy_stft

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
# repo_root/src/<this file> -> one level up is the repo root
REPO_ROOT   = Path(__file__).resolve().parent.parent
AIRBUS_DIR  = REPO_ROOT / 'data' / 'Airbus'
OUT_DIR     = REPO_ROOT / 'results' / 'airbus'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS          = 1024
WINDOW      = 'hann'
NPERSEG     = 126
NOVERLAP    = 118
N_SUBIMG    = 120
IMG_SZ      = 64

VAE_EPOCHS  = 60
VAE_LR      = 5e-4
VAE_BATCH   = 128
LATENT_DIM  = 384
SSIM_W      = 0.7
KL_BETA     = 0.5

CAL_SPLIT         = 0.1
CAD_ALPHA         = 0.02
MAX_BANK_SIZE     = 8000
MAX_BANK_PRETOKENS= 40000
MAX_SUBIMGS_BANK  = 8000
K_NEIGHBORS       = 3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & STFT
# ══════════════════════════════════════════════════════════════════════════════

def encode_stft(df, desc='STFT'):
    n    = len(df)
    need = N_SUBIMG * IMG_SZ
    out  = np.zeros((n, N_SUBIMG, IMG_SZ, IMG_SZ), dtype=np.float32)
    for i in tqdm(range(n), desc=f'  {desc}', leave=False):
        _, _, Zxx = scipy_stft(df.iloc[i].values.astype(np.float32),
                               fs=FS, window=WINDOW,
                               nperseg=NPERSEG, noverlap=NOVERLAP)
        mag = np.abs(Zxx[:IMG_SZ, :])
        mag = mag[:, :need]
        if mag.shape[1] < need:
            mag = np.pad(mag, ((0, 0), (0, need - mag.shape[1])))
        out[i] = mag.reshape(IMG_SZ, N_SUBIMG, IMG_SZ).transpose(1, 0, 2)
    p99 = np.percentile(out, 99)
    if p99 > 0:
        out = np.clip(out / p99, 0.0, 1.0)
    return out


print('\n── Step 1: Loading Airbus data ──')
X_train_df = pd.read_hdf(AIRBUS_DIR / 'dftrain.h5')
X_test_df  = pd.read_hdf(AIRBUS_DIR / 'dfvalid .h5')
y_df       = pd.read_csv(AIRBUS_DIR / 'dfvalid_groundtruth.csv')
y_test_seq = y_df['anomaly'].values.astype(np.int32)   # 594 sequence-level labels

n_train = len(X_train_df)
n_test  = len(X_test_df)
print(f'Train sequences: {n_train}  (all normal)')
print(f'Test  sequences: {n_test}   anomal={y_test_seq.sum()}  normal={(y_test_seq==0).sum()}')

# NEW: unlike the other three datasets, Airbus's training data is 100% normal --
# there is no unused defective pool sitting in the training files to build a
# validation set from. Every labeled anomaly in this dataset lives in the test
# file. So here the validation set is carved out of the existing test sequences
# themselves (stratified, so both slices keep the 50/50 class balance), w/alpha
# are tuned ONLY on the validation slice, and final metrics are reported ONLY
# on the held-out test slice, which never participates in any tuning decision.
from sklearn.model_selection import train_test_split
VAL_FRAC = 0.3
seq_idx_all       = np.arange(n_test)
test_idx, val_idx = train_test_split(seq_idx_all, test_size=VAL_FRAC,
                                     stratify=y_test_seq, random_state=SEED)
print(f'  Val  slice : {len(val_idx):>4} sequences (anomal={y_test_seq[val_idx].sum()})  ← used for w/alpha tuning')
print(f'  Test slice : {len(test_idx):>4} sequences (anomal={y_test_seq[test_idx].sum()})  ← untouched, final report only')

print('\n── Step 2: STFT encoding ──')
X_train_stft = encode_stft(X_train_df, 'Train STFT')   # (1677, 120, 64, 64)
X_test_stft  = encode_stft(X_test_df,  'Test  STFT')   # ( 594, 120, 64, 64)
del X_train_df, X_test_df

# ── Split at sequence level (preserves temporal correlation within sequences) ──
n_fit_seq    = int((1 - CAL_SPLIT) * n_train)
n_calib_seq  = n_train - n_fit_seq

X_fit_stft   = X_train_stft[:n_fit_seq]    # (1509, 120, 64, 64)
X_calib_stft = X_train_stft[n_fit_seq:]    # ( 168, 120, 64, 64)

# ── Flatten to patch arrays ────────────────────────────────────────────────────
X_fit_sub   = X_fit_stft.reshape(-1, IMG_SZ, IMG_SZ)    # (1509*120, 64, 64)
X_calib_sub = X_calib_stft.reshape(-1, IMG_SZ, IMG_SZ)  # ( 168*120, 64, 64) = 20,160
X_test_sub  = X_test_stft.reshape(-1, IMG_SZ, IMG_SZ)   # ( 594*120, 64, 64) = 71,280

# Labels at patch level: each patch inherits its sequence label
y_test_patch = np.repeat(y_test_seq, N_SUBIMG)   # (71,280,)

print(f'\nPatch counts:')
print(f'  Fit patches  : {len(X_fit_sub):>7}  ({n_fit_seq} seq × {N_SUBIMG})')
print(f'  Calib patches: {len(X_calib_sub):>7}  ({n_calib_seq} seq × {N_SUBIMG})  ← was {n_calib_seq} in v1')
print(f'  Test  patches: {len(X_test_sub):>7}  ({n_test} seq × {N_SUBIMG})')
print(f'  min p-value v1: 1/{n_calib_seq+1} = {1/(n_calib_seq+1):.5f}')
print(f'  min p-value v2: 1/{len(X_calib_sub)+1} = {1/(len(X_calib_sub)+1):.7f}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def subimg_to_pil(arr):
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
    T.RandomCrop(IMG_SZ, padding=4),
    T.ColorJitter(brightness=0.15, contrast=0.15),
])


class SubImageDS(Dataset):
    def __init__(self, subimgs, augment=False):
        self.subimgs = subimgs
        self.augment = augment
    def __len__(self): return len(self.subimgs)
    def __getitem__(self, i):
        img = subimg_to_pil(self.subimgs[i])
        if self.augment:
            img = aug(img)
        return tf_dino(img), tf_dec(img)

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

print('\n── Step 3: Loading DINOv2-small ──')
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
dino.eval().to(DEVICE)
for p in dino.parameters():
    p.requires_grad = False

_probe    = torch.zeros(1, 3, 224, 224, device=DEVICE)
_out      = dino.forward_features(_probe)
USE_PATCH = 'x_norm_patchtokens' in _out
del _out
DINO_DIM  = 384 * (2 if USE_PATCH else 1)
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
# 5. DINOv2-VAE
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu      = nn.Linear(DINO_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(DINO_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),          nn.LeakyReLU(0.2),
            nn.Linear(256, 256 * 4 * 4),         nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,   32, 4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,    3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def encode_feat(self, feat):
        return self.fc_mu(feat), self.fc_log_var(feat).clamp(-10, 10)

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

def cache_features(subimgs, desc='cache'):
    all_feats = []
    for i in tqdm(range(0, len(subimgs), 256), desc=f'  {desc}', leave=False):
        batch = subimgs[i:i+256]
        imgs  = torch.stack([tf_dino(subimg_to_pil(p)) for p in batch]).to(DEVICE)
        feats = dino_features(imgs)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)

print('\n── Step 3b: Pre-caching DINOv2 features ──')
feat_fit   = cache_features(X_fit_sub,   'fit feats')
feat_calib = cache_features(X_calib_sub, 'calib feats')
feat_test  = cache_features(X_test_sub,  'test feats')
print(f'  Cached: fit={feat_fit.shape}  calib={feat_calib.shape}  test={feat_test.shape}')


class CachedFeatDS(Dataset):
    def __init__(self, feats, subimgs):
        self.feats   = feats
        self.subimgs = subimgs
    def __len__(self): return len(self.feats)
    def __getitem__(self, i):
        img = subimg_to_pil(self.subimgs[i])
        img = aug(img)
        return self.feats[i], tf_dec(img)

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

fit_loader = DataLoader(CachedFeatDS(feat_fit, X_fit_sub),
                        batch_size=VAE_BATCH, shuffle=True,
                        num_workers=4, pin_memory=True)

opt = torch.optim.AdamW(model.parameters(), lr=VAE_LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, eta_min=1e-6)

print(f'\n── Step 4: Training DINOv2-VAE ({VAE_EPOCHS} epochs) ──')
epoch_bar = tqdm(range(1, VAE_EPOCHS + 1), desc='  Training', unit='epoch')
for epoch in epoch_bar:
    model.train()
    total = 0.0
    batch_bar = tqdm(fit_loader, desc=f'    Epoch {epoch:3d}/{VAE_EPOCHS}',
                     leave=False, unit='batch')
    for feat, x_t in batch_bar:
        feat, x_t = feat.to(DEVICE), x_t.to(DEVICE)
        recon, mu, lv = model.forward_feat(feat)
        loss = combined_loss(recon, x_t, mu, lv)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        batch_bar.set_postfix(loss=f'{loss.item():.5f}')
    total /= len(fit_loader)
    sch.step()
    epoch_bar.set_postfix(loss=f'{total:.5f}', lr=f'{opt.param_groups[0]["lr"]:.1e}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. VAE RESIDUALS — patch level (no aggregation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(cached_feats, subimgs):
    """Returns (N_patches,) p95 pixel residual — one per patch."""
    model.eval()
    res = []
    with torch.no_grad():
        for i in tqdm(range(0, len(subimgs), 256), desc='  residuals', leave=False):
            feat = cached_feats[i:i+256].to(DEVICE)
            x_t  = torch.stack([tf_dec(subimg_to_pil(s))
                                 for s in subimgs[i:i+256]]).to(DEVICE)
            mu, _  = model.encode_feat(feat)
            recon  = model.decode(mu)
            mae    = (x_t - recon).abs().mean(dim=1)
            res.append(np.percentile(mae.cpu().numpy(), 95, axis=(1, 2)))
    r = np.concatenate(res)
    if np.isnan(r).any():
        r = np.where(np.isnan(r), np.nanmax(r), r)
    return r   # shape: (N_patches,)


print('\n── Step 5: Computing VAE residuals (patch-level) ──')
res_fit_sub   = compute_residuals(feat_fit,   X_fit_sub)    # (n_fit_patches,)
res_calib_sub = compute_residuals(feat_calib, X_calib_sub)  # (20160,)
res_test_sub  = compute_residuals(feat_test,  X_test_sub)   # (71280,)

# ══════════════════════════════════════════════════════════════════════════════
# 8. DINOv2 MEMORY BANK — patch level
# ══════════════════════════════════════════════════════════════════════════════

def _all_patch_tokens(subimgs, desc='tokens'):
    all_toks = []
    for i in tqdm(range(0, len(subimgs), 64), desc=f'  {desc}', leave=False):
        batch = subimgs[i:i+64]
        imgs  = torch.stack([tf_dino(subimg_to_pil(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(imgs)
        toks  = F.normalize(toks.flatten(0, 1).float(), dim=1)
        all_toks.append(toks.cpu())
    return torch.cat(all_toks)


print('\n── Step 6: Building DINOv2 memory bank ──')
if len(X_fit_sub) > MAX_SUBIMGS_BANK:
    bank_idx   = np.random.choice(len(X_fit_sub), MAX_SUBIMGS_BANK, replace=False)
    X_fit_bank = X_fit_sub[bank_idx]
else:
    X_fit_bank = X_fit_sub

fit_tokens = _all_patch_tokens(X_fit_bank, 'fit tokens')
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


def knn_scores_sub(subimgs, desc='knn'):
    """Per-patch KNN score = max over internal DINOv2 patch tokens."""
    scores = []
    for i in tqdm(range(0, len(subimgs), 32), desc=f'  {desc}', leave=False):
        batch = subimgs[i:i+32]
        imgs  = torch.stack([tf_dino(subimg_to_pil(p)) for p in batch]).to(DEVICE)
        toks  = dino_patch_tokens(imgs)
        toks  = F.normalize(toks.float(), dim=-1)
        B, N, D = toks.shape
        flat  = toks.reshape(B * N, D)
        d     = torch.cdist(flat, bank)
        knn_d = d.topk(K_NEIGHBORS, dim=1, largest=False).values.mean(1)
        knn_d = knn_d.reshape(B, N).max(1).values
        scores.append(knn_d.cpu().numpy())
    return np.concatenate(scores)   # (N_patches,)


print('\n── Step 7: Memory bank scoring (patch-level) ──')
bank_fit_sub   = knn_scores_sub(X_fit_sub,   'fit knn')
bank_calib_sub = knn_scores_sub(X_calib_sub, 'calib knn')
bank_test_sub  = knn_scores_sub(X_test_sub,  'test knn')

# ══════════════════════════════════════════════════════════════════════════════
# 9. ENSEMBLE NCS — patch level
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 8: NCS & Ensemble (patch-level) ──')

def znorm(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-9)

vae_fit_n   = znorm(res_fit_sub,   res_fit_sub)
vae_calib_n = znorm(res_calib_sub, res_fit_sub)
vae_test_n  = znorm(res_test_sub,  res_fit_sub)

bank_fit_n   = znorm(bank_fit_sub,   bank_fit_sub)
bank_calib_n = znorm(bank_calib_sub, bank_fit_sub)
bank_test_n  = znorm(bank_test_sub,  bank_fit_sub)

# Sweep ensemble weight using sequence-level AUROC on the VALIDATION slice only
# (aggregate patches → sequence). The test slice is never used here.
print('  Sweeping ensemble weight (on validation slice, not final test slice) ...')
best_auroc_ens, best_ew = 0.0, 0.9
for ew in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ncs_patch = (1 - ew) * vae_test_n + ew * bank_test_n
    # Aggregate to sequence: max NCS patch per sequence
    ncs_seq   = ncs_patch.reshape(n_test, N_SUBIMG).max(axis=1)
    a         = roc_auc_score(y_test_seq[val_idx], ncs_seq[val_idx])
    print(f'    ew={ew:.1f} → val AUROC={a:.4f}')
    if a > best_auroc_ens:
        best_auroc_ens, best_ew = a, ew

print(f'  Best ensemble weight: {best_ew}  (val AUROC={best_auroc_ens:.4f})')

# Patch-level NCS
ncs_calib_patch = (1 - best_ew) * vae_calib_n + best_ew * bank_calib_n  # (20160,)
ncs_test_patch  = (1 - best_ew) * vae_test_n  + best_ew * bank_test_n   # (71280,)

# ══════════════════════════════════════════════════════════════════════════════
# 10. CONFORMAL ANOMALY DETECTION — patch-level calibration
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Step 9: Conformal Anomaly Detection (patch-level) ──')

n_cal = len(ncs_calib_patch)   # 20,160
print(f'  Calibration patches: {n_cal}  (was {n_calib_seq} sequences in v1)')

# p-value for each of the 71,280 test patches
print('  Computing p-values for all test patches ...')
p_values_patch = np.array(
    [(np.sum(ncs_calib_patch >= s) + 1) / (n_cal + 1) for s in tqdm(ncs_test_patch, leave=False)]
)   # (71280,)

# Aggregate to sequence level: min p-value over 120 patches
# (if ANY patch is anomalous, the sequence is flagged)
p_values_seq = p_values_patch.reshape(n_test, N_SUBIMG).min(axis=1)   # (594,)

print(f'  min p-value (patch): {p_values_patch.min():.6f}  (v1 min was ~{1/(n_calib_seq+1):.5f})')

# alpha is swept on the VALIDATION slice's p-values (computed against the SAME
# calibration reference used for the final test slice), NEVER on the held-out
# test slice.
best_f1_cad, best_alpha = 0.0, CAD_ALPHA
for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    y_pred_val = (p_values_seq[val_idx] < alpha).astype(int)
    f1         = f1_score(y_test_seq[val_idx], y_pred_val, zero_division=0)
    print(f'    alpha={alpha:.2f} → val F1={f1:.4f}')
    if f1 > best_f1_cad:
        best_f1_cad, best_alpha = f1, alpha

print(f'  Best alpha: {best_alpha}  (val F1={best_f1_cad:.4f})')

# Final decision applied to the untouched test slice only
y_pred_cad = (p_values_seq[test_idx] < best_alpha).astype(int)

# Save results (all sequence-level arrays here are already restricted to the
# held-out test slice -- final reporting never touches the validation slice)
_save = OUT_DIR / 'airbus_valsplit.npz'
np.savez(_save,
         ncs_calib=ncs_calib_patch,               # patch level (20160,)
         ncs_test=p_values_seq[test_idx],         # sequence level, test slice only
         p_values=p_values_seq[test_idx],
         p_values_patch=p_values_patch,           # full patch level (71280,) — reference only
         y_test=y_test_seq[test_idx],
         n_cal=np.int64(n_cal),
         best_alpha=np.float64(best_alpha),
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

# Sequence-level NCS for ROC (max NCS patch per sequence) -- computed over all
# 594 pooled test sequences, then restricted to the held-out test_idx slice
# ONLY at the point of final reporting below.
ncs_test_seq_all  = ncs_test_patch.reshape(n_test, N_SUBIMG).max(axis=1)
ncs_calib_seq_ref = ncs_calib_patch.reshape(n_calib_seq, N_SUBIMG).max(axis=1)
vae_test_seq_all  = vae_test_n.reshape(n_test, N_SUBIMG).max(axis=1)
bank_test_seq_all = bank_test_n.reshape(n_test, N_SUBIMG).max(axis=1)

y_test_final    = y_test_seq[test_idx]
ncs_test_seq    = ncs_test_seq_all[test_idx]
vae_test_seq    = vae_test_seq_all[test_idx]
bank_test_seq   = bank_test_seq_all[test_idx]

print('\n── Step 10: Results (held-out test slice only) ──')
thr_q99    = np.percentile(vae_fit_n,   99)
bank_thr95 = np.percentile(bank_fit_n,  95)

res_vae    = metrics(y_test_final, (vae_test_seq  > thr_q99).astype(int),
                     vae_test_seq,  'DINOv2-VAE @ Q99')
res_bank   = metrics(y_test_final, (bank_test_seq > bank_thr95).astype(int),
                     bank_test_seq, 'Memory Bank only (p95 thr)')
res_best   = metrics(y_test_final, y_pred_cad, ncs_test_seq,
                     f'Patch-level CAD (α={best_alpha:.2f} best F1)')

print('\n' + '='*80)
print(f'  {"Method":<44} {"F1":>6} {"Sens":>6} {"Spec":>6} {"AUROC":>7}')
print('  ' + '-'*77)
print(f'  {"Thesis reference (Airbus Q99)":<44} {"0.8284":>6} {"0.7071":>6} {"1.0000":>6} {"  N/A":>7}')
print(f'  {"v1 (sequence-level CAD)":<44} {"0.8700":>6} {"0.7700":>6} {"1.0000":>6} {"  N/A":>7}')
print(f'  {"DINOv2-VAE @ Q99":<44}'
      f' {res_vae["f1"]:>6.4f} {res_vae["sens"]:>6.4f} {res_vae["spec"]:>6.4f} {res_vae["auroc"]:>7.4f}')
print(f'  {"Memory Bank only":<44}'
      f' {res_bank["f1"]:>6.4f} {res_bank["sens"]:>6.4f} {res_bank["spec"]:>6.4f} {res_bank["auroc"]:>7.4f}')
print(f'  {"Patch-level CAD α={:.2f}".format(best_alpha):<44}'
      f' {res_best["f1"]:>6.4f} {res_best["sens"]:>6.4f} {res_best["spec"]:>6.4f} {res_best["auroc"]:>7.4f}')
print('='*80)

# ══════════════════════════════════════════════════════════════════════════════
# 12. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Airbus — Patch-level DINOv2+CAD (v2)', fontsize=13, fontweight='bold')

ax = axes[0]
for scores, label, color, ls in [
    (vae_test_seq,  f'VAE (AUROC={roc_auc_score(y_test_final, vae_test_seq):.3f})',   'steelblue', '-'),
    (bank_test_seq, f'Mem.Bank (AUROC={roc_auc_score(y_test_final, bank_test_seq):.3f})', 'orange', '-'),
    (ncs_test_seq,  f'Ensemble (AUROC={roc_auc_score(y_test_final, ncs_test_seq):.3f})', 'tomato', '--'),
]:
    fpr, tpr, _ = roc_curve(y_test_final, scores)
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls, label=label)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC — sequence-level (held-out test slice)'); ax.legend(fontsize=8)

ax = axes[1]
bins = np.linspace(0, 1, 50)
p_values_test_final = p_values_seq[test_idx]
ax.hist(p_values_test_final[y_test_final == 0], bins=bins, alpha=0.6, color='steelblue', density=True, label='Normal')
ax.hist(p_values_test_final[y_test_final == 1], bins=bins, alpha=0.6, color='tomato',    density=True, label='Anomal')
ax.axhline(1.0, color='black', lw=1, ls='--', label='Uniform')
ax.axvline(best_alpha, color='black', ls=':', label=f'α={best_alpha}')
ax.set_xlabel('min p-value per sequence')
ax.set_title(f'p-value dist (n_cal={n_cal} patches)'); ax.legend(fontsize=8)

ax = axes[2]
methods = ['Thesis\n(v1)', 'v1 CAD\n(seq)', 'v2 CAD\n(patch)']
f1s     = [0.8284, 0.87, res_best['f1']]
colors  = ['#94A3B8', '#60A5FA', '#2563EB']
bars = ax.bar(methods, f1s, color=colors, width=0.5)
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_title('F1 comparison'); ax.set_ylabel('F1')

plt.tight_layout()
out = OUT_DIR / 'results_airbus_dinov2_valsplit.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nPlot saved → {out}')
print('Done.')
