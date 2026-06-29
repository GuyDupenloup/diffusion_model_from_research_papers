# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import matplotlib.pyplot as plt
import numpy as np
from model_utils import load_diffusion_model


def display_ddpm_images(images, timesteps=None):
    """
    Displays images generated with DDPM method.
    Images have shape [num_images, num_timesteps, H, W, C].
    Each image is displayed on a row at the timesteps specified by `timesteps`.
    """
    # Display images at t = 999, 899, 799 ... 99, 0
    if timesteps is None:
        timesteps = list(range(999, -1, -100)) + [0]

    titles = [f"t={t}" for t in timesteps]

    cols = len(timesteps)
    rows = images.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = np.atleast_2d(axes)

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(images[i, timesteps[j]], cmap="gray")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9, pad=4)
            axes[i, j].axis("off")

    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()


def display_ddim_images(images, rows, cols):
    """
    Displays images generated with the DDIM method.
    Images have shape [num_images, H, W, C].
    They are displayed on a square grid. For example, 64 images
    are displayed on a 8x8 grid.
    """
    if images.shape[0] != rows*cols:
        raise ValueError(f"Expecting {rows*cols} images, received {images.shape[0]}")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))  # e.g. (6, 6)
    # fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.atleast_2d(axes)

    n = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(images[n], cmap="gray")
            n += 1
            axes[i, j].axis("off")

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()


def sample_and_view_images(
    model_dir,
    method="ddim",
    num_screens=10,
    num_steps=50,
    eta=0
):
    """
    Samples diffusion model and displays generated images.
    
    Arguments:
        model_dir (str): Directory where the model files are (config, weights).
        num_screens (str): Number of screens to display.
        method (str): Sampling method, either 'ddpm' or 'ddim'.
        num_steps (int): Number of DDIM steps.
        eta (float): DDIM eta parameter (0: deterministics, 1: DDPM-like).

    If method is:
        'ddpm': Each screen shows 8 images at timesteps 999, 899 ... 99, 0.
        'ddim': Each screen shows a grid of 12 x 12 images.
    """

    # Load diffusion model
    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Generate images
    print(f"\nGenerating {num_images} images using {method} sampling method")

    if method == "ddpm":
        # Generate 8 images per screen
        batch_size = 8
        num_images = num_screens * batch_size

        print(f"Generating {num_images} using DDPM sampling")
        images = model.ddpm_sampling(num_images, keep_all_images=True)

        # Rescale generated images from [-1, 1] to [0, 1] for matplotlib
        images = (images.numpy() + 1) / 2.0
        images = np.clip(images, 0, 1)

        for i in range(0, len(images), batch_size):
            display_ddpm_images(images[i:i+batch_size])

    if method == "ddim":
        # Generate 144 images per screen
        batch_size = 144
        num_images = num_screens * batch_size

        print(f"Generating {num_images} using DDIM sampling with {num_steps} steps (eta={eta}")
        images = model.ddim_sampling(num_images, num_steps=num_steps, eta=eta)

        # Rescale generated images from [-1, 1] to [0, 1] for matplotlib
        # images = (images.numpy() + 1) / 2.0
        images = (images.numpy() + 1) / 2.0
        images = np.clip(images, 0, 1)

        for i in range(0, len(images), batch_size):
            display_ddim_images(images[i:i+batch_size], rows=12, cols=12)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        help="Directory where the diffusion model files are (model config, EMA weights)",
        required=True,
        type=str
    )    
    parser.add_argument(
        "--method",
        help="Sampling method, either 'ddpm' or 'ddim'. Default: 'ddim'",
        choices=["ddpm", "ddim"],
        type=str
    )
    parser.add_argument(
        "--num_screens",
        help="Number of screens to display. Default: 10",
        type=int
    )
    parser.add_argument(
        "--num_steps",
        help="Number of DDIM steps. Default: 50",
        type=int
    )
    parser.add_argument(
        "--eta",
        help="DDIM eta parameter (0: deterministic, 1: DDPM-like). Default: 0",
        type=float
    )

    args = parser.parse_args()

    sample_and_view_images(
        args.model_dir,
        method=args.method,
        num_screens=args.num_screens,
        num_steps=args.num_steps,
        eta=args.eta
    )
