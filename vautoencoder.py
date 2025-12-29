import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for 28x28 grayscale images.
    """

    def __init__(self, latent_size=None):
        super().__init__()
        if latent_size is None:
            latent_size = settings.VAE_LATENT_SIZE
        self.encoder_layers = nn.Sequential(
            nn.Linear(settings.IMAGE_FLATTENED_SIZE, settings.VAE_ENCODER_HIDDEN_SIZE),
            nn.ReLU(True)
        )
        self.to_mu = nn.Linear(settings.VAE_ENCODER_HIDDEN_SIZE, latent_size)
        self.to_logvar = nn.Linear(settings.VAE_ENCODER_HIDDEN_SIZE, latent_size)

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_size, settings.VAE_DECODER_HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(settings.VAE_DECODER_HIDDEN_SIZE, settings.IMAGE_FLATTENED_SIZE),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x):
        hidden = self.encoder_layers(x)
        mu = self.to_mu(hidden)
        logvar = self.to_logvar(hidden)
        return mu, logvar

    def reparam_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_layers(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam_sample(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar

def vae_loss(reconstructed, original, mean, logvar):
    reconstruction_loss = F.binary_cross_entropy(reconstructed, original, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_div

def train(model, opt, loader, device, num_epochs=None):
    if num_epochs is None:
        num_epochs = settings.VAE_NUM_EPOCHS
    model.to(device)
    for ep in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:
            x = x.view(-1, settings.IMAGE_FLATTENED_SIZE).to(device)
            opt.zero_grad()
            rec, mu, logvar = model(x)
            loss = vae_loss(rec, x, mu, logvar)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {ep+1}/{num_epochs}: Loss: {epoch_loss/len(loader.dataset):.4f}")

if __name__ == "__main__":
    dev = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
    model = VariationalAutoencoder(latent_size=settings.VAE_LATENT_SIZE).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.VAE_LEARNING_RATE)

    raise SystemExit("Insert your DataLoader and call train(model, optimizer, train_loader, dev)")
