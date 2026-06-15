import os
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import load_diffusion_model


def sample_images(
    model_dir,
    method,
    num_images,
    ddim_num_steps=100,
    ddim_eta=0
):

    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    print(f"\n{method} Sampling")
    if method == "ddpm":
        images = model.ddpm_sampling(num_images, keep_all_images=True)
    elif method == "ddim":
        images = model.ddim_sampling(num_images, num_steps=ddim_num_steps, eta=ddim_eta)
    else:
        raise ValueError("Sampling method should be 'ddpm' or 'ddim'")

    return images


def view_images(images, method):

    # Rescale to [0, 1] for matplotlib
    images = (images + 1) / 2
    images = np.clip(images, 0.0, 1.0)

    if method == "ddpm":
        for b in range(images.shape[0]):
            fig, axes = plt.subplots(1, 6, figsize=(14, 5))
            i = 0
            for t in (0, 19, 49, 99, 499, 999):    
                axes[i].imshow(images[b, t])
                axes[i].set_title(f"t={t+1}")
                i += 1
            plt.tight_layout()
            plt.show()

    elif method == "ddim":
        b = 0
        while True:
            if len(b) < 4:
                break
            fig, axes = plt.subplots(1, 4, figsize=(14, 5))
            for i in range(0, len(b), 4):



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
    
#     parser.add_argument(
#         "--model_dir",
#         help="Directory where the diffusion model files are",
#         required=True,
#         type=str
#     )    
#     parser.add_argument(
#         "--sampling_batch_size",
#         help="Number of images generated at once",
#         type=int,
#         default=250
#     )
#     parser.add_argument(
#         "--num_steps",
#         help="DDIM number of steps parameter",
#         type=int,
#         default=100
#     )
#     parser.add_argument(
#         "--eta",
#         help="DDIM eta parameter",
#         type=float,
#         default=0
#     )

#     args = parser.parse_args()

#     evaluate_cifar10_fid(
#         args.model_dir,
#         args.sampling_batch_size,
#         args.num_steps,
#         eta=0
#     )
