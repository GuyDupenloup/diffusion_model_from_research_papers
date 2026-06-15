import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import load_diffusion_model


def sample_and_view_images(
    model_dir,
    method="ddim",
    num_images=500,
    num_steps=100
):

    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    if method not in ("ddpm", "ddim"):
        raise ValueError("Sampling method should be 'ddpm' or 'ddim'")
    print(f"\nUsing {method} sampling method")

    if method == "ddpm":
        images = model.ddpm_sampling(num_images, keep_all_images=True)
    else:
        images = model.ddim_sampling(num_images, num_steps=num_steps, eta=0)

    # Rescale generated images from [-1, 1] to [0, 1] for matplotlib
    images = (images.numpy() + 1) / 2.0
    images = np.clip(images, 0, 1)

    if method == "ddpm":
        for i in range(num_images):
            _, axes = plt.subplots(1, 6, figsize=(14, 5))
            ax = 0
            for t in (0, 19, 49, 99, 499, 999):    
                axes[ax].imshow(images[i, t])
                axes[ax].set_title(f"t={t}")
                ax += 1
            plt.tight_layout()
            plt.show()

    elif method == "ddim":
        for i in range(num_images):
            plt.imshow(images[i])
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
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
