# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import tensorflow as tf
from models.diffusion_model import DiffusionModel
from utils.model_utils import print_trainable_variables
from utils.train_utils import SaveCheckpointCallback, load_checkpoint_weights


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


def train_model(
    output_dir,
    epochs,
    save_period,
    resume_from=None
):
    """
    Trains a model, saving checkpoints at regular intervals
    as the training progresses.
    If interrupted, training can be resumed from a saved checkpoint
    (model and optimizer weights are restored).

    Arguments:
        output_dir (str): Root directory under which all the directories 
                          and files created during the training are saved.
        epochs (int): Number of training epochs.
        save_period (int): Checkpoint saving period in epochs. If set to 0,
                           no checkpoints are saved.
        resume_from (str): Checkpoint directory to resume the training from
                           (instead of training from scratch)
    
    Output directory structure:
    --------------------------
        `output_dir`
              ├── checkpoints
              |       ├── checkpoint_0
              |       |       └── u_net.weights.h5, ema_net.weights.h5, optimizer_weights.npz
              |       └── checkpoint_1
              |               └── u_net.weights.h5, ema_net.weights.h5, optimizer_weights.npz
              ├── trained_model
              |       └── model_config.json, u_net.weights.h5, ema_net.weights.h5
              └── metrics.csv
    """

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

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

    # The model takes care of the loss function.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4)
    )
    
    # Load checkpoint if resuming a training
    if resume_from:
        chk_epoch = int(os.path.basename(resume_from)[11:])
        print(f">> Loading checkpoint {resume_from}")
        load_checkpoint_weights(resume_from, model)

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dir, "metrics.csv"),
            append=True
        )
    ]
    if save_period > 0:
        callbacks.append(
            SaveCheckpointCallback(
                os.path.join(output_dir, "checkpoints"),
                period=save_period,
                epoch_offset=chk_epoch if resume_from else 0,
            )
        )

    # Train model
    if not resume_from:
      print(">> Starting training")
    else:
      print(f">> Resuming training at epoch {chk_epoch+1}")

    start_time = timer()
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    end_time = timer()
    train_run_time = int(end_time - start_time)
    print(">> Training runtime: " + str(timedelta(seconds=train_run_time))) 

    # Save model config file, and U-Net and EMA weights
    dirpath = os.path.join(output_dir, "trained_model")
    print(f">> Saving trained model in directory {dirpath}")
    model.save(dirpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="Directory where to save training output files",
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
        "--save_period",
        help="Checkpoint saving period in epochs",
        required=True,
        type=int
    )
    parser.add_argument(
        "--resume_from",
        help="Checkpoint directory to resume training from",
        type=str
    )
    args = parser.parse_args()
    
    train_model(
       args.output_dir,
       args.epochs,
       args.save_period,
       resume_from=args.resume_from
    )
