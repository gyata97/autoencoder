"""
Settings file containing all dynamic variables for the autoencoder project.
"""

# Image Configuration
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
IMAGE_FLATTENED_SIZE = IMAGE_SIZE * IMAGE_SIZE  # 784 for 28x28

# Training Configuration
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PRINT_INTERVAL = 100

# Model Configuration
NUM_CLASSES = 10  # For classifier models (AlexNet)

# Autoencoder Configuration
AE_ENCODER_LAYERS = [128, 64, 32]
AE_DECODER_LAYERS = [64, 128, IMAGE_FLATTENED_SIZE]

# Variational Autoencoder Configuration
VAE_LATENT_SIZE = 20
VAE_ENCODER_HIDDEN_SIZE = 400
VAE_DECODER_HIDDEN_SIZE = 400
VAE_LEARNING_RATE = 1e-3
VAE_NUM_EPOCHS = 10

# AlexNet Configuration
ALEXNET_DEFAULT_CLASSES = 10
ALEXNET_DEFAULT_CHANNELS = 1

# Device Configuration
DEVICE = "cuda"  # Will be set to "cpu" if CUDA is not available

