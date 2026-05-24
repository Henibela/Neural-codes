"""
ae_lite_train.py
================
Training script for AE-Lite.

Run:
    python ae_lite_train.py                  # AWGN, default settings
    python ae_lite_train.py --channel rayleigh
    python ae_lite_train.py --epochs 10000 --lr 1e-4

All hyperparameters are at the top of this file — easy to change.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import time
import argparse
import matplotlib.pyplot as plt

from ae_lite_model import AELite


# =============================================================================
# HYPERPARAMETERS  — change these to tune your training
# =============================================================================

CONFIG = {
    # --- Architecture ---
    'k'          : 8,      # bits per block
    'n'          : 16,     # complex symbols per block  (R = k/n = 0.5)
    'channel'    : 'awgn', # 'awgn' or 'rayleigh'

    # --- Training ---
    'epochs'     : 10_000, # number of gradient steps
    'batch_size' : 256,    # samples per step
    'lr'         : 1e-3,   # Adam learning rate (start here, lower if unstable)

    # --- SNR strategy ---
    # 'fixed'  : train at one SNR value (snr_fixed_db below)
    # 'random' : sample SNR uniformly from [snr_min_db, snr_max_db] each step
    #            -> trains one model that generalises across the full BER curve
    'snr_strategy' : 'random',
    'snr_fixed_db' : 7.0,
    'snr_min_db'   : 0.0,
    'snr_max_db'   : 20.0,

    # --- Logging / saving ---
    # All outputs go into subfolders — never mixed with source code.
    # local runs → checkpoints/local_cpu/  and  results/local_cpu/
    # Colab runs → managed by upload_to_github.py into colab_t4/ subfolders
    'log_every'        : 100,
    'checkpoint_every' : 500,
    'checkpoint_dir'   : os.path.join('checkpoints', 'local_cpu'),
    'results_dir'      : os.path.join('results',     'local_cpu'),
    'final_model_path' : os.path.join('checkpoints', 'local_cpu', 'ae_lite_final_awgn.pt'),
}


# =============================================================================
# DATA GENERATION
# =============================================================================

def get_batch(batch_size: int, k: int, device: torch.device) -> torch.Tensor:
    """
    Generate a random batch of bit vectors.

    Returns:
        bits : (batch_size, k) tensor of random bits, dtype float32.
               Values are 0.0 or 1.0 — BCELoss requires floats, not ints.
    """
    return torch.randint(0, 2, (batch_size, k), device=device).float()


def sample_snr(cfg: dict) -> float:
    """Return an SNR value in dB according to the configured strategy."""
    if cfg['snr_strategy'] == 'fixed':
        return cfg['snr_fixed_db']
    elif cfg['snr_strategy'] == 'random':
        return random.uniform(cfg['snr_min_db'], cfg['snr_max_db'])
    else:
        raise ValueError(f"Unknown snr_strategy: {cfg['snr_strategy']}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(cfg: dict):
    """
    Main training function.

    The training loop repeats these four steps for every epoch:
        1. Generate a fresh random batch of bits
        2. Forward pass: bits -> model -> predicted probabilities
        3. Compute BCELoss: how wrong are the predictions?
        4. Backward pass + optimizer step: update all weights

    BCELoss (Binary Cross-Entropy):
        Loss = -1/N * sum_i [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
    where y_i is the true bit (0 or 1) and p_i is the predicted probability.
    This penalises confident wrong predictions very heavily, and rewards
    confident correct ones.

    Loss starts around 0.69 (= log(2), which is what you get from random
    guessing on a binary problem) and should fall steadily during training.
    """

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    if device.type == 'cuda':
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --- Model ---
    model = AELite(k=cfg['k'], n=cfg['n'], channel=cfg['channel']).to(device)
    print(f"Model      : {model}")

    # --- Loss function ---
    # BCELoss requires:
    #   - predicted values in (0, 1)  <- Sigmoid in decoder guarantees this
    #   - target values in {0, 1}     <- our bit vectors
    criterion = nn.BCELoss()

    # --- Optimizer ---
    # Adam adapts the learning rate for each parameter individually.
    # Much faster and more stable than plain SGD for this task.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # --- Checkpoint directory ---
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'],     exist_ok=True)

    # --- Training history (for plotting) ---
    history = {'epoch': [], 'loss': [], 'snr_db': []}

    # --- Training loop ---
    print(f"\nTraining   : {cfg['epochs']} epochs, batch={cfg['batch_size']}, "
          f"lr={cfg['lr']}, SNR strategy={cfg['snr_strategy']}")
    print("-" * 55)

    model.train()   # activate training mode
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):

        # Sample SNR for this step
        snr_db = sample_snr(cfg)

        # Fresh random batch every step — infinite dataset
        bits = get_batch(cfg['batch_size'], cfg['k'], device)

        # --- The four core lines ---
        optimizer.zero_grad()          # 1. clear old gradients (MUST be first)
        bits_hat = model(bits, snr_db) # 2. forward pass
        loss = criterion(bits_hat, bits) # 3. compute loss
        loss.backward()                # 4. backpropagate
        optimizer.step()               # 5. update weights

        # --- Logging ---
        loss_val = loss.item()
        history['epoch'].append(epoch)
        history['loss'].append(loss_val)
        history['snr_db'].append(snr_db)

        if epoch % cfg['log_every'] == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:6d}/{cfg['epochs']}  "
                  f"loss: {loss_val:.5f}  "
                  f"snr: {snr_db:5.1f} dB  "
                  f"elapsed: {elapsed:.1f}s")

        # --- Checkpoint ---
        if epoch % cfg['checkpoint_every'] == 0:
            ckpt_path = os.path.join(
                cfg['checkpoint_dir'],
                f"ae_lite_{cfg['channel']}_epoch{epoch}.pt"
            )
            torch.save({
                'epoch'     : epoch,
                'config'    : cfg,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss'      : loss_val,
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

    # --- Save final model ---
    torch.save(model.state_dict(), cfg['final_model_path'])
    print(f"\nFinal model saved: {cfg['final_model_path']}")
    print(f"Total time : {(time.time()-t0)/60:.1f} min")

    # --- Plot training loss curve ---
    _plot_loss(history, cfg)

    return model, history


def _plot_loss(history: dict, cfg: dict):
    """Plot and save the training loss curve."""
    # Smooth the loss with a 50-step rolling average for readability
    losses = np.array(history['loss'])
    window = min(50, len(losses) // 10 or 1)
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], losses, alpha=0.25, color='#1D9E75', linewidth=0.8)
    ax.plot(history['epoch'][window-1:], smoothed, color='#1D9E75', linewidth=1.5,
            label=f'Loss (smoothed, window={window})')
    ax.axhline(y=np.log(2), color='gray', linestyle='--', linewidth=1,
               label='log(2) ≈ 0.693  (random-guess baseline)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCELoss')
    ax.set_title(f'Training loss — AE-Lite ({cfg["channel"].upper()}, '
                 f'SNR strategy: {cfg["snr_strategy"]})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(cfg['results_dir'], f'ae_lite_training_loss_{cfg["channel"]}.png')
    plt.savefig(path, dpi=150)
    print(f"Loss curve saved: {path}")
    plt.show()


# =============================================================================
# RESUME FROM CHECKPOINT
# =============================================================================

def resume_training(checkpoint_path: str, extra_epochs: int = 5000):
    """
    Resume training from a saved checkpoint.

    Args:
        checkpoint_path : path to the .pt checkpoint file
        extra_epochs    : how many more epochs to run
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg  = ckpt['config']
    start_epoch = ckpt['epoch']

    print(f"Resuming from epoch {start_epoch}, loss was {ckpt['loss']:.5f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = AELite(k=cfg['k'], n=cfg['n'], channel=cfg['channel']).to(device)
    model.load_state_dict(ckpt['model'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    optimizer.load_state_dict(ckpt['optimizer'])

    # Update config for the continuation run
    cfg['epochs'] = extra_epochs
    # Move optimizer state to correct device
    for state in optimizer.state.values():
        for k_, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k_] = v.to(device)

    criterion = nn.BCELoss()
    model.train()

    print(f"Running {extra_epochs} more epochs...")
    for epoch in range(1, extra_epochs + 1):
        snr_db = sample_snr(cfg)
        bits = get_batch(cfg['batch_size'], cfg['k'], device)
        optimizer.zero_grad()
        loss = criterion(model(bits, snr_db), bits)
        loss.backward()
        optimizer.step()
        if epoch % cfg['log_every'] == 0:
            print(f"  Epoch {start_epoch + epoch}  loss: {loss.item():.5f}")

    torch.save(model.state_dict(), cfg['final_model_path'])
    print(f"Saved: {cfg['final_model_path']}")
    return model


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AE-Lite')
    parser.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--epochs',  type=int,   default=CONFIG['epochs'])
    parser.add_argument('--lr',      type=float, default=CONFIG['lr'])
    parser.add_argument('--batch',   type=int,   default=CONFIG['batch_size'])
    parser.add_argument('--snr',     default='random', choices=['random', 'fixed'])
    args = parser.parse_args()

    CONFIG['channel']      = args.channel
    CONFIG['epochs']       = args.epochs
    CONFIG['lr']           = args.lr
    CONFIG['batch_size']   = args.batch
    CONFIG['snr_strategy'] = args.snr
    CONFIG['final_model_path'] = os.path.join('checkpoints', 'local_cpu', f'ae_lite_final_{args.channel}.pt')

    train(CONFIG)
