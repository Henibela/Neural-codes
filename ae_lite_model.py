"""
ae_lite_model.py
================
AE-Lite: Lightweight Autoencoder for End-to-End Wireless Communication
Thesis: ML-Based Autoencoder for End-to-End Wireless Communication
Author: Henok Belayneh | Advisor: Dr. Tsegamlak Terefe
Addis Ababa University

Architecture (from design specification):
  Encoder: Linear(8,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,32) -> Power Norm
  Channel:  AWGN or Rayleigh flat fading
  Decoder: Linear(32,64) -> ReLU -> Linear(64,128) -> ReLU -> Linear(128,8) -> Sigmoid

Coding rate R = k/n = 8/16 = 0.5 bits per complex symbol.
The 32 real outputs of the encoder represent n=16 complex baseband symbols (real + imag pairs).
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# ENCODER
# =============================================================================

class Encoder(nn.Module):
    """
    The transmitter side of the autoencoder.

    Takes a block of k=8 raw bits (as floats 0.0 / 1.0) and maps them
    to n=16 complex baseband symbols, represented as 2n=32 real numbers.

    After the final linear layer, a power normalisation step enforces
    the constraint that the average transmitted power per complex symbol
    equals 1.  Without this the network can "cheat" by making its signal
    arbitrarily large, which makes any BER-vs-SNR comparison meaningless.

    Layer sizes follow the design spec exactly:
        Input  :  8  (k bits)
        Hidden1: 128  neurons + ReLU
        Hidden2:  64  neurons + ReLU
        Output :  32  real values  (= 2n, n=16 complex symbols)
    """

    def __init__(self, k: int = 8, n: int = 16):
        """
        Args:
            k : number of input bits per message block  (default 8)
            n : number of complex symbols per block      (default 16)
                Output dimension will be 2*n real numbers.
        """
        super().__init__()

        self.k = k          # input bits
        self.n = n          # complex symbols out  (output = 2n reals)

        # --- Fully-connected layers ---
        # fc1: 8 inputs -> 128 neurons
        # Why 128?  Wide first layer gives the network enough "workspace"
        # to learn a rich feature mapping from 8 binary inputs.
        self.fc1 = nn.Linear(k, 128)

        # fc2: 128 -> 64
        # Compression step.  Forces the network to distil the most
        # important features — classic autoencoder bottleneck behaviour.
        self.fc2 = nn.Linear(128, 64)

        # fc3: 64 -> 32  (= 2n real values representing n complex symbols)
        # No activation after this layer; power normalisation comes next.
        self.fc3 = nn.Linear(64, 2 * n)

        # Shared ReLU activation.
        # ReLU(x) = max(0, x).  Applied after fc1 and fc2 to introduce
        # non-linearity — without it, stacking linear layers is still just
        # one linear transformation no matter how many layers you add.
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : (batch_size, k) tensor of bits  — dtype float32, values 0. or 1.

        Returns:
            tx : (batch_size, 2n) tensor of power-normalised transmitted symbols.
        """
        # --- Layer 1 ---
        # x shape:  (batch, 8)
        x = self.relu(self.fc1(x))   # -> (batch, 128)

        # --- Layer 2 ---
        x = self.relu(self.fc2(x))   # -> (batch, 64)

        # --- Output layer (no activation yet) ---
        x = self.fc3(x)              # -> (batch, 32)

        # --- Power normalisation ---
        # Goal: enforce that the average power per complex symbol = 1.
        #
        # The 32 outputs are arranged as [Re(s0), Im(s0), Re(s1), Im(s1), ...]
        # The total signal energy in a block of n complex symbols is:
        #     E = sum_i ( Re(si)^2 + Im(si)^2 )  =  ||x||^2
        #
        # We want the *average* energy per symbol = 1, i.e. E/n = 1.
        # So we scale: x_norm = x  /  ( ||x|| / sqrt(n) )
        #
        # x.norm(dim=-1, keepdim=True) computes the L2 norm along the last
        # dimension for every sample in the batch, keeping shape (batch, 1)
        # so that division broadcasts correctly across the 32 columns.
        #
        # Adding 1e-8 prevents division by zero on pathological initialisation.
        tx = x / (x.norm(dim=-1, keepdim=True) / math.sqrt(self.n) + 1e-8)

        return tx   # -> (batch, 32),  avg symbol power = 1 guaranteed


# =============================================================================
# CHANNEL  (AWGN and Rayleigh)
# =============================================================================

class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Adds complex Gaussian noise to the transmitted signal.  Because the
    signal is represented as 32 real numbers (not complex objects), noise
    is simply a real Gaussian tensor of the same shape.

    Noise standard deviation:
        sigma = sqrt( 1 / (2 * R * SNR_linear) )

    where R = k / n = 0.5 is the coding rate and SNR_linear = 10^(Eb/N0_dB / 10).

    This formula comes directly from the definition of Eb/N0:
        Eb/N0 = (signal energy per bit) / (noise spectral density)
    Rearranging for the noise variance gives the formula above.
    """

    def __init__(self, R: float = 0.5):
        """
        Args:
            R : coding rate = k/n  (default 0.5)
        """
        super().__init__()
        self.R = R

    def forward(self, tx: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Args:
            tx     : (batch, 2n) power-normalised transmitted signal
            snr_db : Eb/N0 in dB

        Returns:
            rx : (batch, 2n) received signal = tx + noise
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        # Noise std dev from the Eb/N0 definition
        sigma = (1.0 / (2.0 * self.R * snr_linear)) ** 0.5
        # torch.randn_like creates a tensor of the same shape and device as tx
        # filled with samples from N(0,1), then scaled by sigma -> N(0, sigma^2)
        noise = torch.randn_like(tx) * sigma
        return tx + noise


class RayleighChannel(nn.Module):
    """
    Rayleigh flat-fading channel with coherent detection.

    Each transmitted block is multiplied by a complex fading coefficient
    h ~ CN(0, 1)  (circularly symmetric complex Gaussian).

    In the real-valued representation used here (32 real numbers for 16
    complex symbols), the fading is applied per-symbol pair:
        [Re(r_i), Im(r_i)] = h_real * [Re(s_i), Im(s_i)]
                            - h_imag * [Im(s_i), -Re(s_i)]
                            + noise

    For simplicity (and consistency with your Month 2-3 baseline) we
    draw one scalar fading coefficient per sample in the batch (i.e.
    flat fading — the entire block sees the same channel realisation).

    Coherent detection: the receiver is assumed to know |h|, and the
    received signal is equalised by dividing by |h| before decoding.
    This is the standard assumption in your baseline Rayleigh simulations.
    """

    def __init__(self, R: float = 0.5):
        super().__init__()
        self.R = R

    def forward(self, tx: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Args:
            tx     : (batch, 2n) power-normalised transmitted signal
            snr_db : Eb/N0 in dB  (before fading)

        Returns:
            rx_eq : (batch, 2n) equalised received signal
        """
        batch = tx.size(0)
        device = tx.device

        # Fading coefficients: h = h_r + j*h_i,  h ~ CN(0,1)
        # Real and imaginary parts each ~ N(0, 1/2)
        h_r = torch.randn(batch, 1, device=device) / (2 ** 0.5)  # (batch, 1)
        h_i = torch.randn(batch, 1, device=device) / (2 ** 0.5)  # (batch, 1)
        # |h|^2  — used for equalisation
        h_mag2 = h_r ** 2 + h_i ** 2                              # (batch, 1)
        h_mag  = h_mag2 ** 0.5                                     # (batch, 1)

        # Apply fading to real and imaginary parts of each symbol.
        # tx is arranged as [Re0, Im0, Re1, Im1, ...]
        re = tx[:, 0::2]   # (batch, n) — real parts
        im = tx[:, 1::2]   # (batch, n) — imaginary parts

        # Complex multiplication: (h_r + j*h_i)(re + j*im)
        #   = h_r*re - h_i*im  +  j*(h_r*im + h_i*re)
        faded_re = h_r * re - h_i * im   # (batch, n)
        faded_im = h_r * im + h_i * re   # (batch, n)

        # Interleave back to (batch, 2n)
        faded = torch.zeros_like(tx)
        faded[:, 0::2] = faded_re
        faded[:, 1::2] = faded_im

        # Add AWGN noise
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma = (1.0 / (2.0 * self.R * snr_linear)) ** 0.5
        noise = torch.randn_like(faded) * sigma
        rx = faded + noise

        # Coherent equalisation: divide by |h| to undo fading amplitude
        # This gives the receiver a clean (but still noisy) signal.
        rx_eq = rx / (h_mag + 1e-8)

        return rx_eq


# =============================================================================
# DECODER
# =============================================================================

class Decoder(nn.Module):
    """
    The receiver side of the autoencoder.

    Takes the noisy (and possibly faded) received signal — 32 real numbers —
    and maps it back to estimated probabilities for the original k=8 bits.

    Architecture mirrors the encoder (expand back from narrow to wide):
        Input  :  32  (= 2n noisy received values)
        Hidden1:  64  neurons + ReLU
        Hidden2: 128  neurons + ReLU
        Output :   8  sigmoid units  (one probability per input bit)

    The Sigmoid output squashes each value to (0, 1), which is required
    by BCELoss (Binary Cross-Entropy Loss) used during training.
    Output[i] ≈ 1.0 means the decoder is confident bit i was 1.
    Output[i] ≈ 0.0 means the decoder is confident bit i was 0.
    """

    def __init__(self, k: int = 8, n: int = 16):
        """
        Args:
            k : number of output bit probabilities  (default 8)
            n : number of complex symbols received  (input = 2n reals)
        """
        super().__init__()

        self.k = k
        self.n = n

        # Decoder is a mirror of the encoder (narrow -> wide -> output)
        # fc1: 32 inputs -> 64 neurons
        self.fc1 = nn.Linear(2 * n, 64)

        # fc2: 64 -> 128
        self.fc2 = nn.Linear(64, 128)

        # fc3: 128 -> 8  (one output per input bit)
        self.fc3 = nn.Linear(128, k)

        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Sigmoid is ONLY applied at the final layer.
        # Why not ReLU?  Because ReLU output can exceed 1, but BCELoss
        # requires inputs strictly in (0,1).  Sigmoid guarantees this.

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            y : (batch_size, 2n) noisy received signal

        Returns:
            probs : (batch_size, k) estimated bit probabilities in (0, 1)
        """
        # --- Layer 1 ---
        y = self.relu(self.fc1(y))      # -> (batch, 64)

        # --- Layer 2 ---
        y = self.relu(self.fc2(y))      # -> (batch, 128)

        # --- Output layer with Sigmoid ---
        probs = self.sigmoid(self.fc3(y))   # -> (batch, 8)

        return probs


# =============================================================================
# FULL AE-LITE MODEL  (Encoder + Channel + Decoder)
# =============================================================================

class AELite(nn.Module):
    """
    Complete end-to-end autoencoder transceiver.

    Wraps the Encoder, a chosen channel model, and the Decoder into one
    nn.Module that can be trained with a single optimizer call.

    The channel is included inside forward() so that PyTorch's autograd
    can differentiate through the entire pipeline.  Gradients flow back
    through the noise addition (which is just an addition — differentiable)
    into the decoder and then into the encoder.  Both are updated together
    in one backward() call — this is the "end-to-end" in end-to-end learning.

    Usage:
        model = AELite(channel='awgn')
        bits_hat = model(bits, snr_db=7.0)   # training
        bits_hat = model(bits, snr_db=12.0)  # evaluation at different SNR
    """

    def __init__(self, k: int = 8, n: int = 16, channel: str = 'awgn'):
        """
        Args:
            k       : bits per block   (default 8)
            n       : complex symbols  (default 16)  — coding rate R = k/n = 0.5
            channel : 'awgn' or 'rayleigh'
        """
        super().__init__()

        self.k = k
        self.n = n
        self.R = k / n   # coding rate = 0.5

        self.encoder = Encoder(k=k, n=n)
        self.decoder = Decoder(k=k, n=n)

        if channel == 'awgn':
            self.channel = AWGNChannel(R=self.R)
        elif channel == 'rayleigh':
            self.channel = RayleighChannel(R=self.R)
        else:
            raise ValueError(f"Unknown channel '{channel}'. Choose 'awgn' or 'rayleigh'.")

        self.channel_name = channel

    def forward(self, bits: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Full forward pass: encode -> channel -> decode.

        Args:
            bits   : (batch_size, k) float tensor of input bits (0.0 or 1.0)
            snr_db : Eb/N0 in dB for this forward pass

        Returns:
            bits_hat : (batch_size, k) estimated bit probabilities in (0, 1)
        """
        # Step 1: Encode bits to transmitted symbols
        tx = self.encoder(bits)          # (batch, 2n),  avg power = 1

        # Step 2: Pass through channel (adds noise, or fading + noise)
        rx = self.channel(tx, snr_db)   # (batch, 2n),  noisy received signal

        # Step 3: Decode received signal to bit probability estimates
        bits_hat = self.decoder(rx)      # (batch, k),   values in (0, 1)

        return bits_hat

    def encode_only(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Run only the encoder.  Useful for visualising the learned constellation.

        Args:
            bits : (batch_size, k) float tensor

        Returns:
            tx : (batch_size, 2n) transmitted symbols (power-normalised)
        """
        with torch.no_grad():
            return self.encoder(bits)

    def decode_only(self, rx: torch.Tensor) -> torch.Tensor:
        """
        Run only the decoder.  Useful for offline evaluation.

        Args:
            rx : (batch_size, 2n) received signal

        Returns:
            bits_hat : (batch_size, k) estimated bit probabilities
        """
        with torch.no_grad():
            return self.decoder(rx)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"AELite(k={self.k}, n={self.n}, R={self.R}, "
            f"channel='{self.channel_name}', "
            f"params={self.count_parameters():,})"
        )


# =============================================================================
# QUICK SMOKE TEST  (run this file directly: python ae_lite_model.py)
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("AE-Lite Model Smoke Test")
    print("=" * 55)

    for ch in ['awgn', 'rayleigh']:
        print(f"\n--- Channel: {ch.upper()} ---")
        model = AELite(channel=ch)
        print(model)

        # Fake batch of 64 messages, each 8 random bits
        bits = torch.randint(0, 2, (64, 8)).float()
        snr_db = 7.0

        bits_hat = model(bits, snr_db)

        print(f"  Input  shape : {list(bits.shape)}")
        print(f"  Output shape : {list(bits_hat.shape)}")
        print(f"  Output range : [{bits_hat.min():.4f}, {bits_hat.max():.4f}]"
              f"  (must be inside (0,1))")

        # Check power normalisation
        tx = model.encode_only(bits)
        # Average power per complex symbol = mean of (re^2 + im^2) over all symbols and batch
        re = tx[:, 0::2]
        im = tx[:, 1::2]
        avg_power = (re**2 + im**2).mean().item()
        print(f"  Avg symbol power : {avg_power:.4f}  (must be ~1.0)")

    print("\nSmoke test passed." if True else "")
    print("=" * 55)
