# Autoencoder & Variational Autoencoder

This project provides implementations for both **Autoencoder (AE)** and **Variational Autoencoder (VAE)** models.

- **Autoencoder (AE):**
  - A deterministic neural network that learns to encode input data into a lower-dimensional latent representation and decode it back to reconstruct the original input.

- **Variational Autoencoder (VAE):**
  - A probabilistic generative model that encodes inputs as distributions in the latent space. The VAE learns both the mean and variance for each dimension, and samples from these distributions for decoding, enabling generation of novel data.

Both models are tailored for 28x28 grayscale images and can be used for representation learning or generative tasks.
