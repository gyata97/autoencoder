import torch
from alexnet import AlexNet
import sys
import settings


from cli import get_args

def main():
    args = get_args()

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

