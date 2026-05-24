# AE-Lite: ML-Based Autoencoder for End-to-End Wireless Communication

> BSc Thesis · School of Electrical and Computer Engineering · Addis Ababa University  
> **Author:** Henok Belayneh &nbsp;|&nbsp; **Advisor:** Dr. Tsegamlak Terefe &nbsp;|&nbsp; **Year:** 2025–2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15CNNgKw3e4Rrq_EuDNFw_b9P-HDo-66D#scrollTo=CQISb5Ve7n9d)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

AE-Lite is a lightweight, single-carrier autoencoder transceiver trained end-to-end for physical-layer wireless communication. Instead of hand-designing separate modulation, coding, and detection blocks, the system learns an optimal transmitter–receiver pair jointly using a small MLP trained with binary cross-entropy loss.

The model is benchmarked against uncoded BPSK and QPSK over AWGN and Rayleigh flat-fading channels at coding rate **R = k/n = 8/16 = 0.5 bits/complex symbol**.

---

## Architecture

```
Input bits (k=8)
      ↓
Encoder  →  Linear(8,128) → ReLU → Linear(128,64) → ReLU → Linear(64,32) → Power Norm
      ↓
32 real values = 16 complex symbols  [avg power = 1]
      ↓
Channel  →  AWGN:     rx = tx + N(0, σ²)
         →  Rayleigh: rx = (h·tx + noise) / |h|,  h ~ CN(0,1), coherent EQ
      ↓
Decoder  →  Linear(32,64) → ReLU → Linear(64,128) → ReLU → Linear(128,8) → Sigmoid
      ↓
8 bit probabilities → hard decision → decoded bits
```

| Parameter | Value |
|-----------|-------|
| Input bits `k` | 8 |
| Complex symbols `n` | 16 |
| Coding rate `R` | 0.5 bits / complex symbol |
| Encoder hidden layers | 128 → 64 neurons, ReLU |
| Decoder hidden layers | 64 → 128 neurons, ReLU |
| Output activation | Sigmoid |
| Loss function | Binary Cross-Entropy (BCELoss) |
| Optimizer | Adam, lr = 1e-3 |
| Trainable parameters | ~50,000 |
| Training platform | Google Colab T4 GPU |

---

## Repository Structure

```
ae_lite/
├── ae_lite_model.py        # Encoder, Decoder, AWGNChannel, RayleighChannel, AELite
├── ae_lite_train.py        # Training script with CLI args and checkpointing
├── ae_lite_eval.py         # BER evaluation and plotting
├── ae_lite_colab.py        # All-in-one single-cell version for Colab
├── upload_to_github.py     # Auto-uploads results to this repo from Colab
├── results/
│   ├── ae_lite_ber_comparison_awgn.png
│   ├── ae_lite_ber_comparison_rayleigh.png
│   ├── ae_lite_constellation_awgn.png
│   ├── ae_lite_constellation_rayleigh.png
│   ├── ae_lite_training_loss_awgn.png
│   ├── ae_lite_training_loss_rayleigh.png
│   ├── ber_ae_awgn.npy
│   ├── ber_ae_rayleigh.npy
│   └── snr_range.npy
└── checkpoints/
    └── ae_lite_awgn_epochN.pt
```

---

## Results

### BER vs Eb/N0 — AWGN Channel

![BER AWGN](results/ae_lite_ber_comparison_awgn.png)

### BER vs Eb/N0 — Rayleigh Fading Channel

![BER Rayleigh](results/ae_lite_ber_comparison_rayleigh.png)

### Learned Constellation

![Constellation](results/ae_lite_constellation_awgn.png)

> At R = 0.5, AE-Lite matches BPSK/QPSK within ±0.5 dB — consistent with theory.  
> The learned constellation shows the encoder discovers its own modulation geometry without any hand-design.

---

## Quickstart

### Option A — Google Colab (recommended)

Click the badge at the top or open the notebook directly:

```
https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID_HERE
```

1. Runtime → Change runtime type → **T4 GPU**
2. Run Cell 1 (mounts Google Drive, verifies GPU)
3. Run Cell 2 (model definition)
4. Run Cell 3 (training — edit `CHANNEL` and `NUM_EPOCHS` at the top)
5. Run Cell 4 (BER evaluation)
6. Run Cell 5 (plots)

### Option B — Local

```bash
# Clone the repo
git clone https://github.com/henibela/Neural-codes.git
cd ae-lite

# Create and activate environment
conda create -n ae-lite python=3.10
conda activate ae-lite
pip install torch numpy matplotlib scipy

# Smoke test
python ae_lite_model.py

# Train
python ae_lite_train.py --channel awgn --epochs 10000

# Evaluate
python ae_lite_eval.py --model ae_lite_final_awgn.pt --n_bits 1000000 --constellation
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Epochs | 10,000 – 50,000 |
| Batch size | 256 |
| SNR strategy | Random uniform, 0–20 dB per batch |
| Checkpoint interval | Every 2,000 epochs |
| Training time (T4 GPU) | ~4 min / 10k epochs |

Loss starts at ~0.693 (log 2, random-guess baseline) and converges to <0.1 at full training.

---

## References

1. T. J. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," *IEEE Trans. Cogn. Commun. Netw.*, vol. 3, no. 4, pp. 563–575, Dec. 2017.
2. S. Dörner et al., "Deep learning-based communication over the air," *IEEE J. Sel. Topics Signal Process.*, vol. 12, no. 1, pp. 132–143, Feb. 2018.
3. A. Felix et al., "OFDM-autoencoder for end-to-end learning of communication systems," *IEEE SPAWC*, 2018.
4. N. Abdul Haq et al., "BER performance of BPSK and QPSK over Rayleigh channel and AWGN channel," *IJEETC*, vol. 3, no. 2, 2014.

---

## License

MIT License. See `LICENSE` for details.
