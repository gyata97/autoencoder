# Autoencoder, Variational Autoencoder, and AlexNet

This project provides implementations for **Autoencoder (AE)**, **Variational Autoencoder (VAE)**, and **AlexNet** models.

- **Autoencoder (AE):**
  - A neural network that learns to encode input data into a lower-dimensional latent vector and reconstructs the input from it.

- **Variational Autoencoder (VAE):**
  - A probabilistic generative model that encodes inputs as distributions in the latent space, enabling sampling and novel data generation.

- **AlexNet:**
  - A convolutional neural network, adapted here for 28x28 or single-channel images, suitable for classification tasks.

All models in this repository are tailored for 28x28 grayscale images and useful for representation learning, generative modeling, and classification.

---

## How to Run the Neural Nets

All models can be run using the `main.py` script. You can specify which model to run and various options from the command line.

### 1. Requirements

- Python 3.7+
- PyTorch

Install dependencies (if needed):

```bash
pip install torch
```

### 2. Running a Model

The main entry point is `main.py`. Use the `--model` argument to select the neural net:

- `alexnet`   — AlexNet classifier
- `ae`        — Autoencoder
- `vae`       — Variational Autoencoder

#### Example: Dummy Forward Pass

You can quickly check the models with a dummy input using the `--test-forward` flag:

```bash
# Run AlexNet on random data
python main.py --model alexnet --test-forward

# Run Autoencoder on random data (28x28 grayscale)
python main.py --model ae --test-forward

# Run Variational Autoencoder
python main.py --model vae --test-forward
```

This will print the output shape for a batch of random images. By default, batch size is 64 and input size is 28x28.

#### Additional Options

- `--batch-size`: Batch size for dummy data (default 64)
- `--classes`: Number of output classes for classifier models (default 10)
- `--channels`: Number of input channels (default 1 for grayscale)

Example:

```bash
python main.py --model alexnet --batch-size 128 --channels 1 --classes 10 --test-forward
```

For actual training or evaluation, you’ll need to implement or connect a dataset loader and training loop, as this project focuses on model definitions and architecture.

---

## Computing Correlation Between GPU Number and Data Size

To analyze how the number of GPUs correlates with dataset size:

1. Collect pairs of (`gpu_count`, `data_size`) from your experiments.
2. Organize as:

   ```
   gpu_count, data_size
   1, 10000
   2, 20000
   4, 40000
   ...
   ```

3. Compute correlation in Python:

   ```python
   import numpy as np
   gpu_counts = np.array([1, 2, 4])
   data_sizes = np.array([10000, 20000, 40000])
   correlation = np.corrcoef(gpu_counts, data_sizes)[0, 1]
   print(f"Correlation: {correlation:.4f}")
   ```

   Or with pandas:

   ```python
   import pandas as pd
   df = pd.DataFrame({'gpu_count': [1, 2, 4], 'data_size': [10000, 20000, 40000]})
   print(df.corr().loc['gpu_count', 'data_size'])
   ```

4. Interpret the correlation coefficient: values close to 1 indicate a strong positive relationship (using more GPUs for bigger datasets).

Adapt this process for correlations with training time or model accuracy as needed.

---

## Contributing

We welcome contributions!

1. Fork this repository and clone it.
2. Create a branch for your changes:
   ```
   git checkout -b my-feature
   ```
3. Make changes and commit with clear messages.
4. Push your branch to your fork.
5. Open a pull request describing your changes.

Please ensure your code is consistent with the project’s style and, where applicable, includes tests or documentation updates. Questions? Open an issue!
