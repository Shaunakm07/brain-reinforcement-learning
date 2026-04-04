"""
Small CNN image generator: latent vector → RGB image.

Architecture (DCGAN-style decoder):

    z ∈ R^latent_dim
    → Linear(latent_dim, 512×7×7)
    → Reshape (512, 7, 7)
    → ConvTranspose2d × 5
    → (3, 224, 224), values in [-1, 1] via Tanh

Why DCGAN-style?
    Simple, no extra dependencies, fast to train. The transposed convolution
    ladder doubles the spatial resolution at each step (7→14→28→56→112→224),
    producing images at V-JEPA2's native input resolution (224×224).

Converting output to [0, 1] for the V-JEPA2 pipeline:
    image_01 = (generator(z) + 1) / 2

Channel widths (512→256→128→64→32→3) trade off capacity against memory.
Reducing them speeds up training at the cost of image complexity.
"""

import torch
import torch.nn as nn


class BrainGenerator(nn.Module):
    """DCGAN-style generator mapping latent vectors to 224×224 RGB images."""

    def __init__(self, latent_dim: int = 256, base_channels: int = 512):
        """
        Args:
            latent_dim: dimensionality of the input noise vector
            base_channels: number of channels in the first convolutional layer;
                           subsequent layers halve this at each step
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 7 * 7),
            nn.ReLU(inplace=True),
        )

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        c = base_channels
        self.conv = nn.Sequential(
            _block(c,     c // 2),   # 7  → 14
            _block(c // 2, c // 4),  # 14 → 28
            _block(c // 4, c // 8),  # 28 → 56
            _block(c // 8, c // 16), # 56 → 112
            # Final layer: no BatchNorm, Tanh output
            nn.ConvTranspose2d(c // 16, 3, kernel_size=4, stride=2,
                               padding=1, bias=True),   # 112 → 224
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) noise vector

        Returns:
            image: (B, 3, 224, 224) float tensor in [-1, 1]
        """
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 7, 7)
        return self.conv(x)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample n images, returning (n, 3, 224, 224) in [0, 1]."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        images_tanh = self(z)
        return (images_tanh + 1) / 2
