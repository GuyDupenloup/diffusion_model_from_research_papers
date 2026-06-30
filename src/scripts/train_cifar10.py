# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import tensorflow as tf
from diffusion_model import DiffusionModel
from utils import print_trainable_variables, SaveCheckpointCallback, load_checkpoint_weights


def create_data_loader(x, batch_size):
    """
    Creates a tf.data.Dataset to load CIFAR-10 images (labels are discarded)
    Rescales from [0, 255] to [-1.0, 1.0]
    """
    def preprocess(x):
        x = tf.cast(x, tf.float32)/127.5 - 1.0
        return x

    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_model(output_dir, epochs, resume_dir=None, resume_epoch=0):

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = np.load(os.path.join(output_dir, "dataset", "cifar10_x_train.npy"))
    y_train = np.load(os.path.join(output_dir, "dataset", "cifar10_y_train.npy"))

    # Create data loader for training set images
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = create_data_loader(x_train, batch_size=128)

    # Create diffusion model
    print(">> Creating diffusion model")
    model = DiffusionModel({
        "u_net": {
            "image_size": 32,
            "image_channels": 3,
            "base_channels": 128,
            "channel_multiplier": (1, 2, 2, 2),
            "num_resnet_blocks": 2,
            "attn_resolutions": (16,),
            "dropout_rate": 0.1
        },
        "data_augment": {
            "random_flip": True
        }
    })

    print_trainable_variables(model, params_only=True)
    
    # The model handles loss function and metrics.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4)
    )

    if resume_dir:
        print(f"Resuming training from directory {resume_dir}")
        load_checkpoint_weights(resume_dir, model)

    # Set up callbacks
    callbacks = [
        SaveCheckpointCallback(
            os.path.join(output_dir, "checkpoints"),
            period=50,
            epoch_offset=resume_epoch
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dir, "metrics.csv"),
            append=True
        )
    ]

    # Train model
    print(">> Starting training")
    start_time = timer()
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    end_time = timer()
    train_run_time = int(end_time - start_time)
    print(">> Training runtime: " + str(timedelta(seconds=train_run_time))) 

    # Save the config file and the two models (U-Net and EMA)
    model.save(os.path.join(output_dir, "trained_model"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
    parser.add_argument(
        "--output_dir",
        help="Directory where to save training output files (model config, trained weights)",
        required=True,
        type=str
    )   
    parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        required=True,
        type=int
    )
    parser.add_argument(
        "--resume_dir",
        help="Directory to resume training from",
        type=str
    )
    parser.add_argument(
        "--resume_epoch",
        help="Epoch number starting from the beginning of the training",
        type=int,
        default=0
    )

    args = parser.parse_args()
    train_model(args.output_dir, args.epochs, args.resume_dir, args.epoch_offset)
