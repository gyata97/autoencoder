# Autoencoder, Variational Autoencoder, and AlexNet

This project provides implementations for **Autoencoder (AE)**, **Variational Autoencoder (VAE)**, and **AlexNet** models.

- **Autoencoder (AE):**
  - A deterministic neural network that learns to encode input data into a lower-dimensional latent representation and decode it back to reconstruct the original input.

- **Variational Autoencoder (VAE):**
  - A probabilistic generative model that encodes inputs as distributions in the latent space. The VAE learns both the mean and variance for each dimension, and samples from these distributions for decoding, enabling generation of novel data.

- **AlexNet:**
  - A convolutional neural network originally designed for large-scale image classification, adapted here for 28x28 or single-channel images. AlexNet extracts hierarchical image features using multiple convolutional and pooling layers, followed by fully-connected layers for classification tasks.

All models in this repository are tailored for 28x28 grayscale images and can be used for representation learning, generative tasks, or classification.

## Computing Correlation Between GPU Number and Data Size

An important aspect of optimizing deep learning workflows is to understand how the number of GPUs used for training correlates with the size of the dataset. This can guide decisions about resource allocation and parallelization.

To compute the correlation between GPU number and data size in your experiments:

1. **Collect Data:**  
   Track the number of GPUs (`gpu_count`) and corresponding dataset size (`data_size`, for example, number of images) across multiple training runs.

2. **Prepare Data:**  
   Organize your results as pairs:  
   ```
   gpu_count, data_size
   1, 10000
   2, 20000
   4, 40000
   ...
   ```

3. **Compute Correlation in Python:**  
   Use libraries like NumPy or pandas:
   ```python
   import numpy as np

   gpu_counts = np.array([1, 2, 4])
   data_sizes = np.array([10000, 20000, 40000])
   correlation = np.corrcoef(gpu_counts, data_sizes)[0, 1]
   print(f"Correlation: {correlation:.4f}")
   ```
   Or, with pandas:
   ```python
   import pandas as pd

   df = pd.DataFrame({'gpu_count': [1, 2, 4], 'data_size': [10000, 20000, 40000]})
   print(df.corr().loc['gpu_count', 'data_size'])
   ```

4. **Interpret the Results:**  
   The correlation coefficient ranges from -1 to 1. Values close to 1 indicate a strong positive linear relationship—i.e., you tend to use more GPUs for larger data sizes.

Feel free to adapt this process for more complex scenarios, such as measuring correlations with training time or model accuracy.

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
