"""
ae_lite_colab.py
================
Single-file version for Google Colab.

How to use:
    Split this file into cells exactly at the  # ── CELL N  markers.
    Run Cell 1 first (Drive mount), then cells in order.
    DRIVE_BASE must be set in Cell 1 before this file runs.

Estimated time on Colab T4 GPU:
     5,000 epochs → ~2 min
    10,000 epochs → ~4 min
    50,000 epochs → ~20 min  (recommended for thesis-quality convergence)
"""

# ── CELL 1: Run at the start of every session (Drive + GPU setup) ─────────────
# !nvidia-smi
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# import os, torch
# DRIVE_BASE     = '/content/drive/MyDrive/[04]Projects/BSc-Thesis-project/ae_lite'
# CHECKPOINT_DIR = os.path.join(DRIVE_BASE, 'checkpoints')
# os.makedirs(DRIVE_BASE, exist_ok=True)
# print(f"PyTorch : {torch.__version__}")
# print(f"CUDA    : {torch.cuda.is_available()}")
# print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import random
import math
import time
import os
import matplotlib.pyplot as plt
from scipy.special import erfc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ── CELL 3: CONFIG — edit these before each run ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
CHANNEL    = 'awgn'      # 'awgn' or 'rayleigh'
NUM_EPOCHS = 10_000      # start here; resume and increase later
BATCH_SIZE = 256
LR         = 1e-3        # lower to 1e-4 if loss plateaus after epoch 3000
SNR_MIN    = 0.0
SNR_MAX    = 20.0
k, n       = 8, 16       # do not change — must match model architecture
R          = k / n       # coding rate = 0.5
# ═══════════════════════════════════════════════════════════════════════════════

# Output subfolders — set AFTER config so CHANNEL is defined
# Colab  → DRIVE_BASE/results/colab_t4/  and  DRIVE_BASE/checkpoints/colab_t4/
# Local  → results/local_cpu/            and  checkpoints/local_cpu/
try:
    RESULTS_DIR = os.path.join(DRIVE_BASE, 'results',     'colab_t4')
    CKPT_DIR    = os.path.join(DRIVE_BASE, 'checkpoints', 'colab_t4')
except NameError:
    # DRIVE_BASE not set — running locally, use local subfolders
    RESULTS_DIR = os.path.join('results',     'local_cpu')
    CKPT_DIR    = os.path.join('checkpoints', 'local_cpu')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)
print(f"Results dir    : {RESULTS_DIR}")
print(f"Checkpoint dir : {CKPT_DIR}")

# ── CELL 4: Model definition ──────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(k, 128)
        self.fc2  = nn.Linear(128, 64)
        self.fc3  = nn.Linear(64, 2*n)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x / (x.norm(dim=-1, keepdim=True) / math.sqrt(n) + 1e-8)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(2*n, 64)
        self.fc2     = nn.Linear(64, 128)
        self.fc3     = nn.Linear(128, k)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y):
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        return self.sigmoid(self.fc3(y))


class AWGNChannel(nn.Module):
    def forward(self, tx, snr_db):
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma   = (1.0 / (2.0 * R * snr_lin)) ** 0.5
        return tx + torch.randn_like(tx) * sigma


class RayleighChannel(nn.Module):
    def forward(self, tx, snr_db):
        batch = tx.size(0)
        h_r = torch.randn(batch, 1, device=tx.device) / (2**0.5)
        h_i = torch.randn(batch, 1, device=tx.device) / (2**0.5)
        h_mag = (h_r**2 + h_i**2)**0.5
        re = tx[:, 0::2]; im = tx[:, 1::2]
        faded = torch.zeros_like(tx)
        faded[:, 0::2] = h_r*re - h_i*im
        faded[:, 1::2] = h_r*im + h_i*re
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma   = (1.0 / (2.0 * R * snr_lin)) ** 0.5
        rx = faded + torch.randn_like(faded) * sigma
        return rx / (h_mag + 1e-8)


class AELite(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.channel = AWGNChannel() if CHANNEL == 'awgn' else RayleighChannel()

    def forward(self, bits, snr_db):
        tx = self.encoder(bits)
        rx = self.channel(tx, snr_db)
        return self.decoder(rx)


# ── CELL 5: Training ──────────────────────────────────────────────────────────

model     = AELite().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model   : AELite | Channel: {CHANNEL} | Params: {n_params:,}")
print(f"Epochs  : {NUM_EPOCHS:,}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print("-" * 50)

model.train()
losses = []
t0 = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    snr_db = random.uniform(SNR_MIN, SNR_MAX)
    bits   = torch.randint(0, 2, (BATCH_SIZE, k), device=device).float()

    optimizer.zero_grad()
    loss = criterion(model(bits, snr_db), bits)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 500 == 0:
        elapsed = time.time() - t0
        eta     = (elapsed / epoch) * (NUM_EPOCHS - epoch)
        print(f"Epoch {epoch:6d}/{NUM_EPOCHS}  "
              f"loss: {loss.item():.5f}  "
              f"elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m")

    # Save checkpoint every 2000 epochs to Drive
    # If Colab disconnects, resume from the latest .pt in CKPT_DIR
    if epoch % 2000 == 0:
        ckpt_path = os.path.join(CKPT_DIR, f'ae_lite_{CHANNEL}_epoch{epoch}.pt')
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss'     : loss.item(),
            'channel'  : CHANNEL,
        }, ckpt_path)
        print(f"  checkpoint → {ckpt_path}")

# Save final model
final_path = os.path.join(CKPT_DIR, f'ae_lite_final_{CHANNEL}.pt')
torch.save(model.state_dict(), final_path)
print(f"\nTraining complete.")
print(f"Final loss : {losses[-1]:.5f}")
print(f"Total time : {(time.time()-t0)/60:.1f} min")
print(f"Saved      → {final_path}")

# Training loss plot
w      = max(1, NUM_EPOCHS // 100)
smooth = np.convolve(losses, np.ones(w)/w, mode='valid')
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(losses, alpha=0.2, color='#1D9E75', linewidth=0.8)
ax.plot(smooth, color='#1D9E75', linewidth=1.5, label=f'Smoothed (window={w})')
ax.axhline(np.log(2), color='gray', linestyle='--', linewidth=1,
           label='log(2) = 0.693 — random-guess baseline')
ax.set_xlabel('Epoch'); ax.set_ylabel('BCELoss')
ax.set_title(f'Training loss — AE-Lite {CHANNEL.upper()}')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
loss_plot = os.path.join(RESULTS_DIR, f'ae_lite_training_loss_{CHANNEL}.png')
plt.savefig(loss_plot, dpi=150)
plt.show()
print(f"Loss plot  → {loss_plot}")


# ── CELL 6: BER evaluation ────────────────────────────────────────────────────

# Load the trained model from Drive (safe to re-run independently)
model.load_state_dict(torch.load(final_path, map_location=device))
model.eval()
print(f"Loaded: {final_path}")

SNR_RANGE  = np.arange(0, 21, 2)
N_BITS     = 1_000_000   # increase to 2_000_000 for thesis-quality figures
EVAL_BATCH = 4096

def eval_ber(snr_db):
    errors = 0; total = 0
    n_batches = N_BITS // (EVAL_BATCH * k)
    with torch.no_grad():
        for _ in range(n_batches):
            b       = torch.randint(0, 2, (EVAL_BATCH, k), device=device).float()
            pred    = model(b, snr_db)
            decoded = (pred > 0.5).float()
            errors += (decoded != b).sum().item()
            total  += EVAL_BATCH * k
    return errors / total

print(f"\nEvaluating BER ({N_BITS:,} bits per SNR point)...")
print(f"{'SNR (dB)':>10}  {'BER':>12}  {'Errors':>10}")
print("-" * 38)

ber_ae_awgn = []
for snr in SNR_RANGE:
    ber      = eval_ber(float(snr))
    n_errors = int(ber * N_BITS)
    print(f"{snr:>10.1f}  {ber:>12.2e}  {n_errors:>10,}")
    ber_ae_awgn.append(ber)
    if ber < 1e-6:
        remaining = len(SNR_RANGE) - len(ber_ae_awgn)
        ber_ae_awgn.extend([1e-7] * remaining)
        print("  BER < 1e-6 — stopping early.")
        break

ber_ae_awgn = np.array(ber_ae_awgn)
np.save(os.path.join(RESULTS_DIR, f'ber_ae_{CHANNEL}.npy'), ber_ae_awgn)
np.save(os.path.join(RESULTS_DIR, 'snr_range.npy'), SNR_RANGE)
print(f"\nBER arrays saved → {RESULTS_DIR}/")


# ── CELL 7: BER comparison plot + constellation ───────────────────────────────

snr_fine        = np.arange(0, 21, 0.25)
ber_theory_awgn = np.clip([0.5*erfc(np.sqrt(10**(s/10))) for s in snr_fine], 1e-7, 1)
ber_theory_ray  = np.clip(
    [0.5*(1 - np.sqrt((10**(s/10))/(1+10**(s/10)))) for s in snr_fine], 1e-7, 1)

# BER comparison figure
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogy(snr_fine, ber_theory_awgn, 'k--', lw=1.3, label='BPSK/QPSK theory (AWGN)')
ax.semilogy(snr_fine, ber_theory_ray,  'b--', lw=1.3, label='BPSK theory (Rayleigh, coherent)')
valid = ber_ae_awgn > 1e-7
ax.semilogy(SNR_RANGE[valid], ber_ae_awgn[valid],
            's-', color='#1D9E75', lw=1.6, ms=6,
            label=f'AE-Lite ({CHANNEL.upper()}, simulated)')
ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=13)
ax.set_ylabel('Bit Error Rate (BER)', fontsize=13)
ax.set_title(r'BER vs $E_b/N_0$ — AE-Lite vs BPSK/QPSK  ($k=8$, $n=16$, $R=0.5$)',
             fontsize=12)
ax.set_xlim(0, 20); ax.set_ylim(1e-5, 1)
ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.5)
ax.legend(fontsize=10, loc='lower left')
plt.tight_layout()
ber_plot = os.path.join(RESULTS_DIR, f'ae_lite_ber_comparison_{CHANNEL}.png')
plt.savefig(ber_plot, dpi=200, bbox_inches='tight')
plt.show()
print(f"BER plot → {ber_plot}")

# Constellation figure
all_msgs = torch.zeros(2**k, k)
for i in range(2**k):
    all_msgs[i] = torch.tensor([(i>>b)&1 for b in range(k)], dtype=torch.float32)
with torch.no_grad():
    tx_all = model.encoder(all_msgs.to(device)).cpu()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(tx_all[:, 0].numpy(), tx_all[:, 1].numpy(),
           c=np.arange(2**k), cmap='tab20b', s=18, alpha=0.85)
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.3, label='Unit circle')
ax.set_aspect('equal')
ax.set_xlabel('Re(s₀)'); ax.set_ylabel('Im(s₀)')
ax.set_title(f'Learned constellation — AE-Lite {CHANNEL.upper()}\n'
             f'(256 points = all possible 8-bit messages)')
ax.grid(alpha=0.25); ax.legend(fontsize=9)
plt.tight_layout()
const_plot = os.path.join(RESULTS_DIR, f'ae_lite_constellation_{CHANNEL}.png')
plt.savefig(const_plot, dpi=150)
plt.show()
print(f"Constellation → {const_plot}")


# ── CELL 8: Resume training (use this cell instead of Cell 5 after disconnect) ─
# RESUME_EPOCH = 2000   # epoch number in the checkpoint filename
# EXTRA_EPOCHS = 8000   # how many MORE epochs to run
# LR           = 1e-3   # keep same, or lower to 1e-4 if loss was plateauing
#
# ckpt_path = os.path.join(CKPT_DIR, f'ae_lite_{CHANNEL}_epoch{RESUME_EPOCH}.pt')
# ckpt      = torch.load(ckpt_path, map_location=device)
# model     = AELite().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# model.load_state_dict(ckpt['model'])
# optimizer.load_state_dict(ckpt['optimizer'])
# for state in optimizer.state.values():       # move optimizer state to GPU
#     for kk, v in state.items():
#         if isinstance(v, torch.Tensor): state[kk] = v.to(device)
# print(f"Resumed from epoch {ckpt['epoch']}, loss was {ckpt['loss']:.5f}")
#
# model.train()
# for epoch in range(1, EXTRA_EPOCHS + 1):
#     abs_epoch = ckpt['epoch'] + epoch
#     snr_db    = random.uniform(SNR_MIN, SNR_MAX)
#     bits      = torch.randint(0, 2, (BATCH_SIZE, k), device=device).float()
#     optimizer.zero_grad()
#     loss = criterion(model(bits, snr_db), bits)
#     loss.backward(); optimizer.step()
#     if epoch % 500 == 0:
#         print(f"Epoch {abs_epoch}  loss: {loss.item():.5f}")
#     if epoch % 2000 == 0:
#         p = os.path.join(CKPT_DIR, f'ae_lite_{CHANNEL}_epoch{abs_epoch}.pt')
#         torch.save({'epoch': abs_epoch, 'model': model.state_dict(),
#                     'optimizer': optimizer.state_dict(), 'loss': loss.item(),
#                     'channel': CHANNEL}, p)
#         print(f"  checkpoint → {p}")
# final_path = os.path.join(CKPT_DIR, f'ae_lite_final_{CHANNEL}.pt')
# torch.save(model.state_dict(), final_path)
# print(f"Saved → {final_path}")


# ── CELL 9: Upload to GitHub (run after Cell 7) ───────────────────────────────
# exec(open(os.path.join(DRIVE_BASE, 'upload_to_github.py')).read())
