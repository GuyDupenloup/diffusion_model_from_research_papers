# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import numpy as np
import tensorflow as tf


class SaveCheckpointCallback(tf.keras.callbacks.Callback):
    """
    Callback that periodically saves model and optimizer weights during training.

    The following files are saved:
        - u_net.weights.h5          U-Net weights
        - ema_net.weights.h5        EMA model weights
        - optimizer_weights.npz     Optimizer weights

    Checkpoint directories are named using the pattern `epoch_XXXX`, 
    where `XXXX` is the epoch number.

    Arguments:
        save_dir (str): Directory where checkpoint files are saved.
        period (int): Save interval in epochs.
        epoch_offset (int): Offset added to the epoch number when
            creating checkpoint directory names. This is useful when
            resuming training from a previous checkpoint.
    """

    def __init__(self, save_dir, period=1, epoch_offset=0):
        super().__init__()
        self.save_dir = save_dir
        self.period = period
        self.epoch_offset = epoch_offset

        # Create checkpoints root directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:

            # Create checkpoint directory
            epoch_dir = os.path.join(self.save_dir, f"checkpoint_{epoch+1+self.epoch_offset}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save U-net and EMA-net weights
            self.model.u_net.save_weights(os.path.join(epoch_dir, "u_net.weights.h5"))
            self.model.ema_net.save_weights(os.path.join(epoch_dir, "ema_net.weights.h5"))

            # Save optimizer weights
            opt_weights = [var.numpy() for var in self.model.optimizer.variables]
            np.savez(os.path.join(epoch_dir, "optimizer_weights.npz"), *opt_weights)
 
            print(f"\nsaved checkpoint to directory {epoch_dir}")


def load_checkpoint_weights(dirpath, model):
    """
    Loads model and optimizer weights from a checkpoint directory
    saved by SaveCheckpointCallback.

    Arguments:
        dirpath (str): Checkpoint directory containing the weight files.
        model (keras.Model): Model into which the weights will be loaded.

    The directory must contain the following files:
        - u_net.weights.h5          U-Net weights
        - ema_net.weights.h5        EMA model weights
        - optimizer_weights.npz     Optimizer weights
    """

    # Check that the checkpoint directory exists
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Unable fo find checkpoint directory {dirpath}")

    # Build optimizer (train step with dummy inputs)
    H, W, C = model.image_shape
    dummy_images = tf.zeros((1, H, W, C), dtype=tf.float32)
    model.train_step(dummy_images)

    # Get optimizer weights and set them on optimizer
    fn = os.path.join(dirpath, "optimizer_weights.npz")
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Unable to find optimizer weights file {fn}")
    with np.load(fn) as data:
        opt_weights = [data[k] for k in data.files]
        
    for var, weights in zip(model.optimizer.variables, opt_weights):
        var.assign(weights)

    # Load U-Net weights
    fn = os.path.join(dirpath, "u_net.weights.h5")
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Unable to find U-Net model weights {fn}")
    model.u_net.load_weights(fn)

    # Load EMA weights
    fn = os.path.join(dirpath, "ema_net.weights.h5")
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Unable to find EMA model weights {fn}")
    model.ema_net.load_weights(fn)
