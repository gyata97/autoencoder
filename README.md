# Autoencoder, Variational Autoencoder, and AlexNet

This project provides implementations for **Autoencoder (AE)**, **Variational Autoencoder (VAE)**, and **AlexNet** models.

- **Autoencoder (AE):**
  - A deterministic neural network that learns to encode input data into a lower-dimensional latent representation and decode it back to reconstruct the original input.

- **Variational Autoencoder (VAE):**
  - A probabilistic generative model that encodes inputs as distributions in the latent space. The VAE learns both the mean and variance for each dimension, and samples from these distributions for decoding, enabling generation of novel data.

- **AlexNet:**
  - A convolutional neural network originally designed for large-scale image classification, adapted here for 28x28 or single-channel images. AlexNet extracts hierarchical image features using multiple convolutional and pooling layers, followed by fully-connected layers for classification tasks.

All models in this repository are tailored for 28x28 grayscale images and can be used for representation learning, generative tasks, or classification.
