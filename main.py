import argparse
import torch
from alexnet import AlexNet
import sys

def main():
    parser = argparse.ArgumentParser(description="Run neural nets: AlexNet, AE, VAE")
    parser.add_argument('--model', type=str, required=True, choices=['alexnet', 'ae', 'vae'],
                        help='Model to run: alexnet | ae | vae')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for dummy forward pass')
    parser.add_argument('--classes', type=int, default=10, help='Number of output classes (for classifier models)')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--test-forward', action='store_true',
                        help='If set, runs a dummy forward pass instead of training')
    args = parser.parse_args()

    if args.model == "alexnet":
        model = AlexNet(num_classes=args.classes, in_channels=args.channels)
        print("Loaded AlexNet.")
    elif args.model == "ae":
        # Assume autoencoder.py exists and contains class Autoencoder
        from autoencoder import Autoencoder
        model = Autoencoder(in_channels=args.channels)
        print("Loaded Autoencoder.")
    elif args.model == "vae":
        # Assume vae.py exists and contains class VAE
        from vae import VAE
        model = VAE(in_channels=args.channels)
        print("Loaded VAE.")
    else:
        print("Unknown model.")
        sys.exit(1)

    if args.test_forward:
        # Make a dummy forward pass with random input for 28x28 images
        x = torch.randn(args.batch_size, args.channels, 28, 28)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"Dummy forward output shape: {y.shape}")

if __name__ == "__main__":
    main()

