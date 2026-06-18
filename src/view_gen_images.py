import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import load_diffusion_model


def display_ddpm_images(
    images,
    timesteps=(1000, 750, 500, 350, 250, 100, 50, 1)
):
    """
    Displays images generated with DDPM method.
    Images have shape [num_images, num_timesteps, H, W, C].
    Each image is displayed on a row at the timesteps specified by `timesteps`.
    """

    titles = [f"t={t}" for t in timesteps]

    cols = len(timesteps)
    rows = images.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = np.atleast_2d(axes)

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(images[i, timesteps[j] - 1])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9, pad=4)
            axes[i, j].axis("off")

    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()


def display_ddim_images(images):
    """
    Displays images generated with the DDIM method.
    Images have shape [num_images, H, W, C].
    They are displayed on a square grid. For example, 64 images
    are displayed on a 8x8 grid.
    """

    num_images = len(images)
    grid = math.ceil(math.sqrt(num_images))

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2))
    axes = np.atleast_2d(axes)

    for idx in range(grid * grid):
        i, j = divmod(idx, grid)
        if idx < num_images:
            axes[i, j].imshow(images[idx])
        axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def sample_and_view_images(
    model_dir,
    method="ddim",
    num_images=500,
    num_steps=100
):
    """
    Samples diffusion model and displays generated images.
    
    Arguments:
        model_dir (str): directory where the model files are (config, weights).
        method (str): sampling method, either 'ddpm' or 'ddim'.
        num_images (int): number of images to generate.
        num_steps (int): number of DDIM steps (unused with 'ddpm' method).
    """

    if method.lower() not in ("ddpm", "ddim"):
        raise ValueError("Sampling method should be either 'ddpm' or 'ddim'")

    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Generate images
    print(f"\nGenerating {num_images} images using {method} sampling method")

    if method == "ddpm":
        images = model.ddpm_sampling(num_images, keep_all_images=True)

        # Rescale generated images from [-1, 1] to [0, 1] for matplotlib
        images = (images.numpy() + 1) / 2.0
        images = np.clip(images, 0, 1)

        batch_size = 8  # Display 8 x timesteps grid of images
        for i in range(0, len(images), batch_size):
            display_ddpm_images(images[i:i+batch_size])

    else:
        images = model.ddim_sampling(num_images, num_steps=num_steps, eta=0)

        # Rescale generated images from [-1, 1] to [0, 1] for matplotlib
        images = (images.numpy() + 1) / 2.0
        images = np.clip(images, 0, 1)

        batch_size = 64   # Display 8x8 grid of images
        for i in range(0, len(images), batch_size):
            display_ddim_images(images[i:i+batch_size])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_dir",
        help="Directory where the diffusion model files are (config, weights)",
        required=True,
        type=str
    )    
    parser.add_argument(
        "--method",
        help="Sampling method, either 'ddpm' or 'ddim'",
        type=str,
        default="ddim"
    )
    parser.add_argument(
        "--num_images",
        help="Number of images to generate",
        type=int,
        default=500
    )
    parser.add_argument(
        "--num_steps",
        help="Number of DDIM steps",
        type=int,
        default=100
    )

    args = parser.parse_args()

    sample_and_view_images(
        args.model_dir,
        method=args.method,
        num_images=args.num_images,
        num_steps=args.num_steps
    )
