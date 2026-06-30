# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import time
from datetime import timedelta
import math
import random
import numpy as np
import tensorflow as tf
from utils.model_utils import load_diffusion_model
from utils.fid_utils import get_inception_activations, compute_acts_fid


def compute_fid_score(
    model_dir,
    method="ddim",
    num_steps=50,
    eta=0,
    batch_size=2000,
    save_dir=None
):
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
        save_dir (int):
            Directory where to save batches of generated images to numpy arrays.

    Returns:
        FID score as a Python float. Lower is better.
    """

    # Load the model
    print(f">> Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Get the training set images
    (real_images, _), _ = tf.keras.datasets.mnist.load_data()
    real_images = np.expand_dims(real_images, axis=-1)

    if num_images is not None:
        random.shuffle(real_images)
        real_images = real_images[:num_images]

    num_images = real_images.shape[0]

    if method == "ddpm":
        print(f">> Generating {num_images} images using DDPM sampling")
    else:
        print(f">> Generating {num_images} images using DDIM sampling with {num_steps} steps")

    # Create directory where to save images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

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
        img = img.numpy()

        # Crop images from 32x32 to 28x28
        img = img[:, 2:30, 2:30, :]

        if save_dir:
            fn = os.path.join(save_dir, f"images_{i}.npy")
            print(f"Saving images to {fn}")
            np.save(fn, img.numpy())

        gen_images.append(img)

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

    print(">> Extracting activations for real and generated images")
    real_acts = get_inception_activations(real_images, inception)
    gen_acts = get_inception_activations(gen_images, inception)

    print(">> Computing FID")
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
        help="Sampling method, either 'ddpm' or 'ddim' (Default: 'ddim')",
        choices=["ddpm", "ddim"],
        type=str
    )
    parser.add_argument(
        "--num_steps",
        help="DDIM number of sampling steps (Default: 50)",
        type=int
    )
    parser.add_argument(
        "--eta",
        help="DDIM eta parameter (Default: 0)",
        type=float,
        default=0
    )
    parser.add_argument(
        "--batch_size",
        help="Size of sampling batches (Default: 2000)",
        type=int
    )
    parser.add_argument(
        "--save_dir",
        help="Directory where to save images as they get created",
        type=str
    )
    args = parser.parse_args()

    compute_fid_score(
        model_dir=args.model_dir,
        method=args.method,
        num_steps=args.num_steps,
        eta=args.eta,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
