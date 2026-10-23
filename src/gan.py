"""
GAN-based augmentation for illicit transaction embeddings.

A trained GraphSAGE model produces node embeddings.
The GAN learns the distribution of illicit training embeddings
and generates synthetic examples for classifier-head training.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, hidden=128):
        super().__init__()

        if noise_dim <= 0 or embed_dim <= 0:
            raise ValueError("noise_dim and embed_dim must be positive")

        self.noise_dim = noise_dim
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gan(
    real_embeddings,
    noise_dim=32,
    epochs=500,
    lr=2e-4,
    batch_size=64,
    device="cpu",
    verbose=True,
):
    """
    Train a GAN on real illicit transaction embeddings.

    Returns:
        G: Trained generator
        D: Trained discriminator
        history: Recorded training losses
    """

    if real_embeddings.ndim != 2:
        raise ValueError("real_embeddings must have shape [n_samples, embed_dim]")

    if real_embeddings.shape[0] < 2:
        raise ValueError("At least two real embeddings are required")

    if not torch.isfinite(real_embeddings).all():
        raise ValueError("real_embeddings contains NaN or infinity")

    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")

    if epochs < 1:
        raise ValueError("epochs must be positive")

    if noise_dim < 1:
        raise ValueError("noise_dim must be positive")

    embed_dim = real_embeddings.shape[1]

    G = Generator(noise_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)

    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
    )

    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
    )

    bce = nn.BCEWithLogitsLoss()

    real = real_embeddings.detach().to(device)

    n = real.shape[0]
    bs = min(batch_size, n)

    ones = torch.ones(bs, device=device)
    zeros = torch.zeros(bs, device=device)

    history = []

    G.train()
    D.train()

    for epoch in range(1, epochs + 1):

        # ---------------------------------
        # 1. Train discriminator
        # ---------------------------------

        real_batch = real[
            torch.randint(0, n, (bs,), device=device)
        ]

        noise = torch.randn(bs, noise_dim, device=device)
        fake_batch = G(noise).detach()

        real_logits = D(real_batch)
        fake_logits = D(fake_batch)

        d_loss = (
            bce(real_logits, ones)
            + bce(fake_logits, zeros)
        )

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---------------------------------
        # 2. Train generator
        # ---------------------------------

        # Freeze discriminator parameters.
        for parameter in D.parameters():
            parameter.requires_grad_(False)

        noise = torch.randn(bs, noise_dim, device=device)
        generated = G(noise)

        g_loss = bce(D(generated), ones)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # Unfreeze discriminator for next epoch.
        for parameter in D.parameters():
            parameter.requires_grad_(True)

        # ---------------------------------
        # 3. Record training progress
        # ---------------------------------

        if epoch % 100 == 0 or epoch == epochs:

            record = {
                "epoch": epoch,
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
            }

            history.append(record)

            if verbose:
                print(
                    f"gan epoch {epoch:4d} | "
                    f"d_loss {d_loss.item():.4f} | "
                    f"g_loss {g_loss.item():.4f}"
                )

    return G, D, history


@torch.no_grad()
def generate_synthetic_embeddings(generator, n_samples, noise_dim=None):
    """
    Generate synthetic illicit transaction embeddings.
    """

    if n_samples < 0:
        raise ValueError("n_samples cannot be negative")

    if noise_dim is None:
        noise_dim = generator.noise_dim

    if noise_dim != generator.noise_dim:
        raise ValueError(
            f"Expected noise_dim={generator.noise_dim}, "
            f"received {noise_dim}"
        )

    generator.eval()

    device = next(generator.parameters()).device

    noise = torch.randn(
        n_samples,
        noise_dim,
        device=device,
    )

    return generator(noise)