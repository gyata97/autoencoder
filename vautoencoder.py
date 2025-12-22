import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Simple VAE for 28x28 grayscale images (e.g., MNIST).
    Encoder -> latent mean/logvar -> reparameterize -> decoder.
    """

    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid(),  # bounds output to [0, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction + KL divergence losses summed over all elements and batch.
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence between learned latent distribution and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


if __name__ == "__main__":
    # Example training loop skeleton; plug in your DataLoader and device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalAutoencoder(latent_dim=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    raise SystemExit(
        "Provide a DataLoader (train_loader) and call train_vae(model, optimizer, train_loader, device)."
    )


def train_vae(
    model: VariationalAutoencoder,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 10,
) -> None:
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, _ in train_loader:
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
