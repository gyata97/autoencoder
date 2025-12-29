import argparse
import torch
from alexnet import AlexNet
import sys
import settings

def main():
    parser = argparse.ArgumentParser(description="Run neural nets: AlexNet, AE, VAE")
    parser.add_argument('--model', type=str, required=True, choices=['alexnet', 'ae', 'vae'],
                        help='Model to run: alexnet | ae | vae')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE, help='Batch size for dummy forward pass')
    parser.add_argument('--classes', type=int, default=settings.NUM_CLASSES, help='Number of output classes (for classifier models)')
    parser.add_argument('--channels', type=int, default=settings.IMAGE_CHANNELS, help='Number of input channels')
    parser.add_argument('--test-forward', action='store_true',
                        help='If set, runs a dummy forward pass instead of training')
    args = parser.parse_args()

    if args.model == "alexnet":
        model = AlexNet(num_classes=args.classes, in_channels=args.channels)
        print("Loaded AlexNet.")
    elif args.model == "ae":
        from autoencoder import Autoencoder
        model = Autoencoder()
        print("Loaded Autoencoder.")
    elif args.model == "vae":
        from vautoencoder import VariationalAutoencoder
        model = VariationalAutoencoder(latent_size=settings.VAE_LATENT_SIZE)
        print("Loaded VAE.")
    else:
        print("Unknown model.")
        sys.exit(1)

    if args.test_forward:
        # Make a dummy forward pass with random input for 28x28 images
        x = torch.randn(args.batch_size, args.channels, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"Dummy forward output shape: {y.shape}")

if __name__ == "__main__":
    main()

