# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import time
from datetime import timedelta
import math
import random
import numpy as np
import tensorflow as tf
from utils.model_utils import load_diffusion_model
from utils.fid_utils import get_inception_activations, compute_acts_fid


def compute_fid_score(model_dir, method, num_steps=None, eta=None, batch_size=None):
    """
    Generates images with the diffusion model and computes the FID score.
 
    Samples a batch of images from the diffusion model using DDIM, then
    computes the FID score against the provided real images by extracting
    Inception v3 activations from both sets and comparing their distributions.
 
    Arguments:
        model_dir (str):
            Directory where the model files are (config, EMA weights).
        method (str):
            Sampling method, either 'ddpm' or 'ddim'.
        num_steps (int):
            Number of DDIM steps to use for generation.
        eta (float):
            DDIM eta parameter. Fully deterministic sampling if eta=0, 
            DDPM-like if eta=1.
         batch_size (int):
            Sampling batch size (number of images generated per sampling).

    Returns:
        FID score as a Python float. Lower is better.
    """

    # Load the model
    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Get the training set images
    (real_images, _), _ = tf.keras.datasets.mnist.load_data()
    real_images = np.expand_dims(real_images, axis=-1)

    if num_images is not None:
        random.shuffle(real_images)
        real_images = real_images[:num_images]

    num_images = real_images.shape[0]

    if method == "ddpm":
        print(f"\nGenerating {num_images} images using DDPM sampling")
    else:
        print(f"\nGenerating {num_images} images using DDIM sampling with {num_steps} steps")

    num_batches = math.ceil(num_images / batch_size)
    gen_images = []

    start_time = time.time()
    for i in range(num_batches):
        print(f"batch {i+1}/{num_batches}")

        current_batch_size = min(batch_size, num_images - i*batch_size)
        if method == "ddpm":
            img = model.ddpm_sampling(current_batch_size)
        else:
            img = model.ddim_sampling(current_batch_size, num_steps=num_steps, eta=eta)

        # Crop the images from 32x32 to 28x28
        img = img[:, 2:30, 2:30, :]

        gen_images.append(img.numpy())


    # Get a single numpy array with shape (num_images, H, W, C)
    gen_images = np.concatenate(gen_images, axis=0)

    elapsed = time.time() - start_time
    print(f"Generation time: {str(timedelta(seconds=int(elapsed)))}")

    # Rescale the generated images from [-1, 1] to [0, 255]
    gen_images = (gen_images + 1) * 127.5
    gen_images = gen_images.clip(0, 255).astype(np.uint8)

    # Create inception V3 model
    inception = tf.keras.applications.InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3)
    )

    print("\nExtracting activations for real images")
    real_acts = get_inception_activations(real_images, inception)

    print("Extracting activations for generated images")
    gen_acts = get_inception_activations(gen_images, inception)

    fid_score = compute_acts_fid(real_acts, gen_acts)

    print(f"\n\nFID score: {fid_score:.2f}")

    return fid_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )  
    parser.add_argument(
        "--model_dir",
        help="Directory where the model files are (config, EMA weights)",
        required=True,
        type=str
    )   
    parser.add_argument(
        "--method",
        help="Sampling method, either 'ddpm' or 'ddim'",
        choices=["ddpm", "ddim"],
        type=str,
        default="ddim"
    )
    parser.add_argument(
        "--num_steps",
        help="DDIM number of sampling steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--eta",
        help="DDIM eta parameter (deterministic if eta=0, DDPM-like if eta=1)",
        type=float,
        default=0
    )
    parser.add_argument(
        "--batch_size",
        help="Size of sampling batches",
        type=int,
        default=2000
    )
    args = parser.parse_args()

    compute_fid_score(
        model_dir=args.model_dir,
        method=args.method,
        num_steps=args.num_steps,
        eta=args.eta,
        batch_size=args.batch_size
    )
