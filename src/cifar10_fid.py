import time
from datetime import timedelta
import argparse
import numpy as np
import tensorflow as tf
from scipy import linalg
from utils import load_diffusion_model


def get_inception_activations(images, inception, batch_size=64):
    """Returns (N, 2048) pool3 features from Inception v3."""
    def preprocess(x):
        x = tf.image.resize(tf.cast(x, tf.float32), [299, 299])
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x.set_shape([299, 299, 3])  # restore shape info for Keras
        return x

    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return inception.predict(dataset, verbose=1)


def compute_fid(acts1, acts2):
    mu1, sigma1 = acts1.mean(axis=0), np.cov(acts1, rowvar=False)
    mu2, sigma2 = acts2.mean(axis=0), np.cov(acts2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def evaluate_cifar10_fid(
    model_dir,
    sampling_batch_size,
    num_steps=100,
    eta=0
):

    # Load the model
    print(f"Loading diffusion model from directory {model_dir}")
    model = load_diffusion_model(model_dir, ema_net_only=True)

    # Get the CIFAR-10 dataset images
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    real_images = np.concatenate([x_train, x_test], axis=0) 
    num_images = real_images.shape[0]

    # Generate the same number of images
    print(f"\nGenerating {num_images} images")
    start_time = time.time()

    gen_images = []
    num_batches = -(-num_images // sampling_batch_size)  # ceiling division
    for i in range(num_batches):
        current_batch_size = min(sampling_batch_size, num_images - i * sampling_batch_size)
        img = model.ddim_sampling(current_batch_size, num_steps=num_steps, eta=eta)
        gen_images.append(img.numpy())

    gen_images = np.concatenate(gen_images, axis=0)[:num_images]

    elapsed = time.time() - start_time
    print(f"Generation time: {str(timedelta(seconds=int(elapsed)))}")

    # Rescale the generated images from [-1, 1] to [0, 255]
    gen_images = (gen_images + 1) * 127.5
    gen_images = gen_images.clip(0, 255).astype(np.uint8)

    inception = tf.keras.applications.InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3)
    )
    print("\nExtracting activations for real images")
    real_acts = get_inception_activations(real_images, inception)

    print("Extracting activations for generated images")
    gen_acts = get_inception_activations(gen_images, inception)

    fid_score = compute_fid(real_acts, gen_acts)

    print("\nFID score:", fid_score)

    return fid_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        help="Directory where the diffusion model files are",
        required=True,
        type=str
    )    
    parser.add_argument(
        "--sampling_batch_size",
        help="Number of images generated at once",
        type=int,
        default=250
    )
    parser.add_argument(
        "--num_steps",
        help="DDIM number of steps parameter",
        type=int,
        default=100
    )
    parser.add_argument(
        "--eta",
        help="DDIM eta parameter",
        type=float,
        default=0
    )

    args = parser.parse_args()

    evaluate_cifar10_fid(
        args.model_dir,
        args.sampling_batch_size,
        args.num_steps,
        eta=0
    )
