import argparse
import settings


def get_args():
    parser = argparse.ArgumentParser(description="Run neural nets: AlexNet, AE, VAE")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["alexnet", "ae", "vae"],
        help="Model to run: alexnet | ae | vae",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.BATCH_SIZE,
        help="Batch size for dummy forward pass",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=settings.NUM_CLASSES,
        help="Number of output classes (for classifier models)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=settings.IMAGE_CHANNELS,
        help="Number of input channels",
    )
    parser.add_argument(
        "--test-forward",
        action="store_true",
        help="If set, runs a dummy forward pass instead of training",
    )
    return parser.parse_args()

