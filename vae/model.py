"""A small convolutional VAE for compressing DonkeyCar camera frames.

Input: RGB image, default 80x160x3 (H x W x C), normalized to [0, 1].
Output: reconstruction of the same shape, plus (mu, logvar) for the latent distribution.

Kept deliberately small -- the point of this whole approach is that the encoder is cheap
enough to run every frame on a Jetson. Grow `latent_dim` or the conv channels only if you've
confirmed reconstructions are the bottleneck, not the RL side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 80x160 -> 40x80
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 40x80 -> 20x40
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 20x40 -> 10x20
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # 10x20 -> 5x10
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 256 * 5 * 10
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 5 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 5x10 -> 10x20
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 10x20 -> 20x40
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 20x40 -> 40x80
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 40x80 -> 80x160
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        h = h.view(-1, 256, 5, 10)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding (mean of q(z|x)) -- use this at inference/RL time,
        not a sampled z, so the RL agent sees a stable observation."""
        mu, _ = self.encoder(x)
        return mu


def vae_loss(recon, x, mu, logvar, kld_weight: float = 1e-3):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_weight * kld, recon_loss, kld
