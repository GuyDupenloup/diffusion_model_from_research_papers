import time
from datetime import timedelta
import math
import random
import argparse
from scipy import linalg
import numpy as np
import tensorflow as tf
from utils import load_diffusion_model


def get_inception_activations(images, inception, batch_size=64):
    """
    Returns (N, 2048) pool3 features from Inception v3.
 
    Preprocesses the input images (resizing, channel conversion, normalization)
    and runs them through the Inception v3 model to extract pooled activations.
 
    Args:
        images:
            Array of images with shape (N, H, W, C), uint8 or float.
            Grayscale images (C=1) are automatically converted to RGB.
        inception:
            A Keras Inception v3 model with `include_top=False` and
            `pooling='avg'`, returning (N, 2048) activations.
        batch_size (int):
            Number of images to process per batch. Defaults to 64.
 
    Returns:
        Numpy array of shape (N, 2048) containing the pool3 activations.
    """

    def preprocess(x):
        x = tf.cast(x, tf.float32)
        # Convert grayscale to RGB by repeating the channel 3 times
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = tf.image.resize(x, [299, 299])
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x.set_shape([299, 299, 3])
        return x

    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return inception.predict(dataset, verbose=1)


def compute_acts_fid(acts1, acts2, eps=1e-6):
    """
    Computes the Fréchet Inception Distance (FID) between two sets of activations.
 
    FID measures the similarity between two distributions of Inception v3
    activations by fitting multivariate Gaussians and computing the Fréchet
    distance between them. Lower scores indicate greater similarity.
 
    Args:
        acts1:
            Numpy array of shape (N, D) containing activations for the
            first set of images (e.g. real images).
        acts2:
            Numpy array of shape (N, D) containing activations for the
            second set of images (e.g. generated images).
        eps:
            Small value added to the diagonal of each covariance matrix
            for numerical stability. Defaults to 1e-6.
 
    Returns:
        FID score as a Python float. Lower is better.
 
    Raises:
        ValueError: If the matrix square root produces an imaginary component
            that exceeds the tolerance threshold, which typically indicates
            too few samples were provided (at least 2048 are recommended).
    """
        
    mu1, sigma1 = acts1.mean(axis=0), np.cov(acts1, rowvar=False)
    mu2, sigma2 = acts2.mean(axis=0), np.cov(acts2, rowvar=False)

    # Stabilize near-singular covariance matrices
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(
                f"Imaginary component in sqrtm result is too large.\n"
                f"Try using more samples (at least 2048)."
            )
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_score(model, real_images, num_steps, batch_size, eta):
    """
    Generates images with the diffusion model and computes the FID score.
 
    Samples a batch of images from the diffusion model using DDIM, then
    computes the FID score against the provided real images by extracting
    Inception v3 activations from both sets and comparing their distributions.
 
    Args:
        model:
            Keras diffusion model. Returns images in the range [-1, 1].
        real_images:
            NumPy array of shape (N, H, W, C) containing real images
            in the range [0, 255] (uint8).
        num_steps (int):
            Number of DDIM denoising steps to use during generation.
        batch_size (int):
            Sampling batch size (number of images generated per sampling).
        eta (float):
            DDIM eta parameter. Fully deterministic sampling if eta=0, 
            DDPM-like if eta=1.
 
    Returns:
        FID score as a Python float. Lower is better.
    """

    num_images = real_images.shape[0]
    print(f"\nGenerating {num_images} images using {num_steps} steps")

    num_batches = math.ceil(num_images / batch_size)

    start_time = time.time()
    gen_images = []
    for i in range(num_batches):
        print(f"batch {i+1}/{num_batches}")
        current_batch_size = min(batch_size, num_images - i*batch_size)

        img = model.ddim_sampling(
            current_batch_size,
            num_steps=num_steps,
            eta=eta
        )
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


def main(
    dataset,
    model_dir,
    num_images=10000,
    num_steps=100,
    eta=0,
    batch_size=500
):
    """
    Loads a trained diffusion model and evaluates its FID score on a dataset.
 
    Loads the model from the given directory, fetches the training set of the
    specified dataset, optionally subsamples it, and computes the FID score
    between real and generated images.
 
    Arguments:
        dataset (str):
            Name of the dataset to evaluate against, either 'mnist' or 'cifar10'.
        model_dir (str):
            Path to the directory where the model files are (config, weights).
        num_images (int):
            Number of real and generated images to use when computing the FID score.
            Defaults to 10000.
        num_steps (int):
            Number of DDIM denoising steps. Defaults to 100.
        eta (float):
            DDIM eta parameter. Fully deterministic sampling if eta=0, DDPM-like
            if eta=1. Defaults to 0.
        batch_size (int):
            Sampling batch size (number of images generated per sampling). Defaults to 500.
    """

    if dataset not in ("mnist", "cifar10"):
        raise ValueError("The dataset name should be 'mnist' or 'cifar10'")
    
    # Load the model
    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Get the training set images
    if dataset == "mnist":
        (real_images, _), _ = tf.keras.datasets.mnist.load_data()
        real_images = np.expand_dim(real_images, axis=-1)
    else:
        (real_images, _), _ = tf.keras.datasets.cifar10.load_data()

    if num_images is not None:
        random.shuffle(real_images)
        real_images = real_images[:num_images]

    fid_score = compute_fid_score(
        model,
        real_images,
        num_steps,
        batch_size,
        eta
    )

    print(f"\n\nFID score: {fid_score:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
    parser.add_argument(
        "--dataset",
        help="Dataset name, either 'mnist' or 'cifar10'",
        required=True,
        type=str
    )   
    parser.add_argument(
        "--model_dir",
        help="Directory where the model files are (config, weights)",
        required=True,
        type=str
    )
    parser.add_argument(
        "--num_images",
        help="Number of real/generated images used to compute FID score",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--num_steps",
        help="Number of DDIM steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--eta",
        help="DDIM eta parameter (deterministic if 0)",
        type=float,
        default=0
    )
    parser.add_argument(
        "--batch_size",
        help="Model sampling batch size",
        type=int,
        default=500
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.model_dir,
        num_images=args.num_images,
        num_steps=args.num_steps,
        eta=args.eta,
        batch_size=args.batch_size
    )
