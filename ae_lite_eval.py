"""
ae_lite_eval.py
===============
Evaluation script: generates BER vs Eb/N0 curves for AE-Lite
and overlays BPSK / QPSK theoretical references.

Run:
    python ae_lite_eval.py                              # AWGN model
    python ae_lite_eval.py --model ae_lite_final_rayleigh.pt --channel rayleigh
    python ae_lite_eval.py --n_bits 5000000            # more bits -> better low-BER estimate
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os

from scipy.special import erfc
from ae_lite_model import AELite


# =============================================================================
# THEORETICAL BER FORMULAS
# =============================================================================

def ber_bpsk_awgn(snr_db: float) -> float:
    """
    Theoretical BER for BPSK over AWGN.
    BER = 0.5 * erfc( sqrt(Eb/N0) )

    BPSK and Gray-coded QPSK have identical BER vs Eb/N0 — they differ
    only in spectral efficiency (bits per Hz), not in noise performance.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_lin))


def ber_bpsk_rayleigh(snr_db: float) -> float:
    """
    Theoretical BER for BPSK over Rayleigh flat fading (coherent detection).
    BER = 0.5 * (1 - sqrt( gamma / (1 + gamma) ))
    where gamma = Eb/N0 in linear.

    Rayleigh fading produces a characteristic "BER floor" at high SNR —
    the curve falls much more slowly than AWGN because deep fades still
    occasionally wipe out entire blocks.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * (1.0 - np.sqrt(snr_lin / (1.0 + snr_lin)))


# =============================================================================
# EVALUATION LOOP
# =============================================================================

def evaluate_ber(model: AELite,
                 snr_db: float,
                 n_bits: int = 1_000_000,
                 batch_size: int = 4096,
                 device: torch.device = torch.device('cpu')) -> float:
    """
    Estimate BER at a single SNR point.

    Procedure:
        1. Generate n_bits random bits (in batches to fit memory)
        2. Pass through the model at the given snr_db
        3. Hard-decision decode: probability > 0.5  ->  bit = 1
        4. Count mismatches against the original bits
        5. BER = errors / total_bits

    Why so many bits?
        To measure BER = 1e-5 reliably you need ~100 errors at that SNR,
        so you need 100 / 1e-5 = 10M bits ideally.  1M is the minimum
        for a rough estimate.  Use 5M+ for thesis-quality plots.

    Args:
        model      : trained AELite model in eval() mode
        snr_db     : Eb/N0 in dB for this measurement
        n_bits     : total bits to test  (must be divisible by batch_size * k)
        batch_size : process this many messages at a time
        device     : cpu or cuda

    Returns:
        BER estimate (float in [0, 1])
    """
    k = model.k
    n_messages = n_bits // k
    n_batches  = n_messages // batch_size

    total_bits  = 0
    error_bits  = 0

    with torch.no_grad():   # no gradient tracking needed for evaluation
        for _ in range(n_batches):
            # Ground truth: random bits
            bits = torch.randint(0, 2, (batch_size, k), device=device).float()

            # Model prediction: probabilities in (0, 1)
            probs = model(bits, snr_db)

            # Hard decision: threshold at 0.5
            # prob > 0.5  ->  decoded bit = 1
            # prob <= 0.5 ->  decoded bit = 0
            bits_decoded = (probs > 0.5).float()

            # Count bit errors
            errors = (bits_decoded != bits).sum().item()

            error_bits  += errors
            total_bits  += batch_size * k

    return error_bits / total_bits if total_bits > 0 else float('nan')


def sweep_ber(model: AELite,
              snr_range: np.ndarray,
              n_bits: int = 1_000_000,
              batch_size: int = 4096,
              device: torch.device = torch.device('cpu'),
              stop_below: float = 1e-6) -> np.ndarray:
    """
    Evaluate BER across a range of SNR values.

    Args:
        snr_range  : array of Eb/N0 values in dB
        n_bits     : bits per SNR point
        stop_below : stop sweeping if BER falls below this threshold
                     (avoids wasting time at very high SNR)

    Returns:
        ber_array : array of BER values, same length as snr_range
    """
    ber_list = []
    print(f"\nEvaluating BER across {len(snr_range)} SNR points "
          f"({n_bits:,} bits each)...")
    print(f"{'SNR (dB)':>10}  {'BER':>12}  {'Errors':>10}")
    print("-" * 36)

    for snr_db in snr_range:
        ber = evaluate_ber(model, float(snr_db), n_bits, batch_size, device)
        n_errors = int(ber * n_bits)
        print(f"{snr_db:>10.1f}  {ber:>12.2e}  {n_errors:>10}")
        ber_list.append(ber)
        if ber < stop_below:
            # Pad remaining points with the threshold value
            remaining = len(snr_range) - len(ber_list)
            ber_list.extend([stop_below] * remaining)
            print(f"  BER < {stop_below:.0e} — stopping early.")
            break

    return np.array(ber_list)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_ber_comparison(snr_range_disc: np.ndarray,
                        ber_ae_awgn: np.ndarray,
                        ber_ae_rayleigh: np.ndarray = None,
                        save_path: str = 'ae_lite_ber_comparison_awgn.png'):
    """
    Plot BER vs Eb/N0 for AE-Lite vs BPSK/QPSK theory.

    Produces a publication-ready figure matching the style of your
    Month 2-3 baseline plots (semilogy, grid, legend).

    Args:
        snr_range_disc    : SNR points where AE-Lite was evaluated (dB)
        ber_ae_awgn       : AE-Lite BER array over AWGN
        ber_ae_rayleigh   : AE-Lite BER array over Rayleigh (optional)
        save_path         : output file path
    """
    # Smooth theory curves at fine resolution
    snr_fine = np.arange(0, 21, 0.25)
    ber_theory_awgn     = np.array([ber_bpsk_awgn(s) for s in snr_fine])
    ber_theory_rayleigh = np.array([ber_bpsk_rayleigh(s) for s in snr_fine])

    # Clip to avoid log(0) issues
    ber_theory_awgn     = np.clip(ber_theory_awgn, 1e-7, 1)
    ber_theory_rayleigh = np.clip(ber_theory_rayleigh, 1e-7, 1)
    ber_ae_awgn         = np.clip(ber_ae_awgn, 1e-7, 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # --- Theory lines (dashed) ---
    ax.semilogy(snr_fine, ber_theory_awgn,
                'k--', linewidth=1.3, label='BPSK / QPSK theory (AWGN)')
    ax.semilogy(snr_fine, ber_theory_rayleigh,
                'b--', linewidth=1.3, label='BPSK theory (Rayleigh, coherent)')

    # --- AE-Lite AWGN result ---
    valid = ber_ae_awgn > 1e-7
    ax.semilogy(snr_range_disc[valid], ber_ae_awgn[valid],
                's-', color='#1D9E75', linewidth=1.6, markersize=6,
                label='AE-Lite (AWGN, simulated)')

    # --- AE-Lite Rayleigh result (if provided) ---
    if ber_ae_rayleigh is not None:
        ber_ae_rayleigh = np.clip(ber_ae_rayleigh, 1e-7, 1)
        valid_r = ber_ae_rayleigh > 1e-7
        ax.semilogy(snr_range_disc[valid_r], ber_ae_rayleigh[valid_r],
                    'D-', color='#E8593C', linewidth=1.6, markersize=6,
                    label='AE-Lite (Rayleigh, simulated)')

    # --- Formatting ---
    ax.set_xlabel(r'$E_b / N_0$ (dB)', fontsize=13)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=13)
    ax.set_title(
        r'BER vs $E_b/N_0$ — AE-Lite vs BPSK / QPSK  ($k=8$, $n=16$, $R=0.5$)',
        fontsize=12
    )
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.show()


# =============================================================================
# CONSTELLATION VISUALISATION
# =============================================================================

def plot_constellation(model: AELite, save_path: str = 'ae_lite_constellation_awgn.png'):
    """
    Visualise the learned constellation — the pattern of transmitted symbols.

    For each of the 2^k = 256 possible 8-bit messages, encode it and plot
    the first two real values (= Re and Im of the first complex symbol).

    A good AE-Lite constellation will show points spread across the unit
    circle with near-equal spacing — it discovers its own modulation scheme.
    Compare to BPSK (2 points) or QPSK (4 points on the unit circle).
    """
    k = model.k
    # All 256 possible messages
    all_messages = torch.zeros(2**k, k)
    for i in range(2**k):
        bits = [(i >> b) & 1 for b in range(k)]
        all_messages[i] = torch.tensor(bits, dtype=torch.float32)

    with torch.no_grad():
        tx = model.encode_only(all_messages)   # (256, 32)

    # Plot first complex symbol: (Re, Im) = (tx[:,0], tx[:,1])
    x = tx[:, 0].numpy()
    y = tx[:, 1].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(x, y, c=np.arange(2**k), cmap='tab20', s=20, alpha=0.8)
    # Draw unit circle for reference (avg power = 1 per symbol)
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.3,
            label='Unit circle (avg power)')
    ax.set_aspect('equal')
    ax.set_xlabel('Re(s₀)')
    ax.set_ylabel('Im(s₀)')
    ax.set_title('Learned constellation — first complex symbol of AE-Lite\n'
                 '(each point = one of the 256 possible 8-bit messages)')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Constellation saved: {save_path}")
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AE-Lite BER')
    parser.add_argument('--model',    default=os.path.join('checkpoints', 'local_cpu', 'ae_lite_final_awgn.pt'),
                        help='Path to trained model .pt file')
    parser.add_argument('--channel',  default='awgn',
                        choices=['awgn', 'rayleigh'],
                        help='Channel the model was trained on')
    parser.add_argument('--model_r',  default=None,
                        help='Optional: path to Rayleigh-trained model '
                             '(to overlay both curves on one plot)')
    parser.add_argument('--n_bits',   type=int, default=1_000_000,
                        help='Bits per SNR evaluation point (more = more accurate)')
    parser.add_argument('--snr_step', type=float, default=2.0,
                        help='SNR step size in dB')
    parser.add_argument('--constellation', action='store_true',
                        help='Also plot the learned constellation')
    args = parser.parse_args()

    # --- Output directory ---
    results_dir = os.path.join('results', 'local_cpu')
    os.makedirs(results_dir, exist_ok=True)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- SNR range ---
    snr_range = np.arange(0, 21, args.snr_step)

    # --- Load AWGN model ---
    model_awgn = AELite(channel='awgn')
    model_awgn.load_state_dict(
        torch.load(args.model, map_location='cpu')
    )
    model_awgn.eval()
    model_awgn.to(device)
    print(f"Loaded: {args.model}  ({model_awgn.count_parameters():,} params)")

    # --- Evaluate over AWGN channel ---
    ber_awgn = sweep_ber(model_awgn, snr_range,
                         n_bits=args.n_bits, device=device)

    # --- Optionally evaluate Rayleigh model ---
    ber_rayleigh = None
    if args.model_r and os.path.exists(args.model_r):
        model_ray = AELite(channel='rayleigh')
        model_ray.load_state_dict(torch.load(args.model_r, map_location='cpu'))
        model_ray.eval()
        model_ray.to(device)
        print(f"Loaded Rayleigh model: {args.model_r}")
        ber_rayleigh = sweep_ber(model_ray, snr_range,
                                 n_bits=args.n_bits, device=device)

    # --- Plot ---
    plot_ber_comparison(snr_range, ber_awgn, ber_rayleigh,
                        save_path=os.path.join(results_dir, f'ae_lite_ber_comparison_{args.channel}.png'))

    # --- Constellation ---
    if args.constellation:
        plot_constellation(model_awgn,
                           save_path=os.path.join(results_dir, f'ae_lite_constellation_{args.channel}.png'))

    # --- Save results as numpy ---
    np.save(os.path.join(results_dir, 'ber_ae_awgn.npy'),  ber_awgn)
    np.save(os.path.join(results_dir, 'snr_range.npy'),     snr_range)
    if ber_rayleigh is not None:
        np.save(os.path.join(results_dir, 'ber_ae_rayleigh.npy'), ber_rayleigh)
    print(f"Results saved to: {results_dir}/")
