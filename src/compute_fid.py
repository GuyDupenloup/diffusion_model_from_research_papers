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
    """Returns (N, 2048) pool3 features from Inception v3."""
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
    Evaluate FID score
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
    num_steps=100,
    num_images=50000,
    batch_size=500,
    eta=0
):
    
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


from google.colab import drive
drive.mount("/content/drive")

main(
    dataset="mnist",
    model_dir="/content/drive/MyDrive/mnist/checkpoints/epoch_500",
    num_steps=100,
    num_images=10000,
    batch_size=500,
    eta=0
)

# main(
#     dataset="cifar10",
#     model_dir="/content/drive/MyDrive/cifar10/checkpoints/epoch_500",
#     num_steps=100,
#     num_images=50000,
#     batch_size=500,
#     eta=0
# )
