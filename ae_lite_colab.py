"""
ae_lite_colab.py
================
Single-file version for Google Colab.

Copy the entire contents into one Colab notebook cell and run.
Everything — model definition, training, evaluation, plotting — is here.

Steps:
    1. Run this cell once.
    2. Watch the training loss print every 100 epochs.
    3. When training finishes, the BER plot and constellation appear automatically.

Estimated time on Colab T4 GPU:
    5,000 epochs  ->  ~2 min
   10,000 epochs  ->  ~4 min
   50,000 epochs  ->  ~20 min  (recommended for thesis-quality convergence)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ==============================================================================
# CONFIG — edit these
# ==============================================================================
k           = 8        # bits per block
n           = 16       # complex symbols per block  (R = 0.5)
BATCH_SIZE  = 256
NUM_EPOCHS  = 10_000   # increase to 50_000 for full convergence
LR          = 1e-3
CHANNEL     = 'awgn'   # 'awgn' or 'rayleigh'
SNR_MIN     = 0.0
SNR_MAX     = 20.0
R           = k / n    # coding rate = 0.5


# ==============================================================================
# MODEL
# ==============================================================================

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
        # Power normalisation: average symbol power = 1
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
        batch  = tx.size(0)
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


# ==============================================================================
# TRAINING
# ==============================================================================

model     = AELite().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: AELite | Channel: {CHANNEL} | Params: {n_params:,}")
print(f"Training for {NUM_EPOCHS} epochs ...")
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

    if epoch % 100 == 0:
        print(f"Epoch {epoch:6d}/{NUM_EPOCHS}  loss: {loss.item():.5f}  "
              f"snr: {snr_db:5.1f} dB  time: {time.time()-t0:.0f}s")

print(f"\nTraining complete. Final loss: {losses[-1]:.5f}")
torch.save(model.state_dict(), f'ae_lite_final_{CHANNEL}.pt')
print(f"Saved: ae_lite_final_{CHANNEL}.pt")

# Plot loss curve
fig, ax = plt.subplots(figsize=(8, 3.5))
w = max(1, NUM_EPOCHS // 100)
smooth = np.convolve(losses, np.ones(w)/w, mode='valid')
ax.plot(losses, alpha=0.2, color='#1D9E75')
ax.plot(smooth, color='#1D9E75', linewidth=1.5, label='Loss (smoothed)')
ax.axhline(np.log(2), color='gray', linestyle='--', linewidth=1,
           label='log(2) ≈ 0.693 (random baseline)')
ax.set_xlabel('Epoch'); ax.set_ylabel('BCELoss')
ax.set_title(f'Training loss — AE-Lite ({CHANNEL.upper()})')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'ae_lite_training_loss_{CHANNEL}.png', dpi=150)  # matches ae_lite_train.py
plt.show()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_ber_point(mdl, snr_db, n_bits=1_000_000, batch_size=4096):
    mdl.eval()
    errors = 0; total = 0
    n_batches = n_bits // (batch_size * k)
    with torch.no_grad():
        for _ in range(n_batches):
            bits    = torch.randint(0, 2, (batch_size, k), device=device).float()
            pred    = mdl(bits, snr_db)
            decoded = (pred > 0.5).float()
            errors += (decoded != bits).sum().item()
            total  += batch_size * k
    return errors / total


snr_range  = np.arange(0, 21, 2)
print("\nEvaluating BER...")
print(f"{'SNR':>6}  {'BER':>12}")
print("-" * 22)

ber_ae_awgn = []   # named to match ae_lite_eval.py convention
for snr in snr_range:
    b = evaluate_ber_point(model, float(snr))
    ber_ae_awgn.append(b)
    print(f"{snr:>6.0f}  {b:>12.2e}")

ber_ae_awgn = np.array(ber_ae_awgn)
np.save(f'ber_ae_{CHANNEL}.npy', ber_ae_awgn)   # matches eval: ber_ae_awgn.npy / ber_ae_rayleigh.npy
np.save('snr_range.npy', snr_range)


# ==============================================================================
# FINAL PLOT
# ==============================================================================

snr_fine         = np.arange(0, 21, 0.25)
ber_theory_awgn  = np.clip([0.5*erfc(np.sqrt(10**(s/10))) for s in snr_fine], 1e-7, 1)
ber_theory_ray   = np.clip(
    [0.5*(1 - np.sqrt((10**(s/10))/(1+10**(s/10)))) for s in snr_fine], 1e-7, 1)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogy(snr_fine, ber_theory_awgn,  'k--',  lw=1.3, label='BPSK / QPSK theory (AWGN)')
ax.semilogy(snr_fine, ber_theory_ray,   'b--',  lw=1.3, label='BPSK theory (Rayleigh, coherent)')

valid = ber_ae_awgn > 1e-7
ax.semilogy(snr_range[valid], ber_ae_awgn[valid],
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
plt.savefig(f'ae_lite_ber_comparison_{CHANNEL}.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"BER plot saved: ae_lite_ber_comparison_{CHANNEL}.png")


# ==============================================================================
# CONSTELLATION
# ==============================================================================

all_msgs = torch.zeros(2**k, k)
for i in range(2**k):
    all_msgs[i] = torch.tensor([(i>>b)&1 for b in range(k)], dtype=torch.float32)

with torch.no_grad():
    tx_all = model.encoder(all_msgs.to(device)).cpu()

x_pts = tx_all[:, 0].numpy()
y_pts = tx_all[:, 1].numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_pts, y_pts, c=np.arange(2**k), cmap='tab20b', s=18, alpha=0.85)
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.3, label='Unit circle')
ax.set_aspect('equal')
ax.set_xlabel('Re(s₀)'); ax.set_ylabel('Im(s₀)')
ax.set_title('Learned constellation — first complex symbol\n'
             '(256 points = all possible 8-bit messages)')
ax.grid(alpha=0.25); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'ae_lite_constellation_{CHANNEL}.png', dpi=150)
plt.show()
print(f"Constellation saved: ae_lite_constellation_{CHANNEL}.png")
