# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
import tensorflow as tf
from scipy import linalg


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
