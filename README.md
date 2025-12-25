# Autoencoder, Variational Autoencoder, and AlexNet

This project provides implementations for **Autoencoder (AE)**, **Variational Autoencoder (VAE)**, and **AlexNet** models.

- **Autoencoder (AE):**
  - A deterministic neural network that learns to encode input data into a lower-dimensional latent representation and decode it back to reconstruct the original input.

- **Variational Autoencoder (VAE):**
  - A probabilistic generative model that encodes inputs as distributions in the latent space. The VAE learns both the mean and variance for each dimension, and samples from these distributions for decoding, enabling generation of novel data.

- **AlexNet:**
  - A convolutional neural network originally designed for large-scale image classification, adapted here for 28x28 or single-channel images. AlexNet extracts hierarchical image features using multiple convolutional and pooling layers, followed by fully-connected layers for classification tasks.

All models in this repository are tailored for 28x28 grayscale images and can be used for representation learning, generative tasks, or classification.

## Contributing

We welcome contributions to this project! To contribute:

1. Fork this repository and clone it to your local machine.
2. Create a new branch for your feature or bugfix:  
   ```
   git checkout -b my-feature
   ```
3. Make your changes and commit them with clear commit messages.
4. Push your branch to your forked repository.
5. Open a pull request describing your changes and their purpose.

Please ensure your code adheres to the project's coding style and includes relevant tests or documentation updates. If you have questions or need help, feel free to open an issue.
