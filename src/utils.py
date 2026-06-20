# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
from tabulate import tabulate
import numpy as np
import tensorflow as tf
from diffusion_model import DiffusionModel


def restore_model_optimizer(dirpath, model):
        
    # Load optimizer config file
    fn = os.path.join(dirpath, "optimizer_config.json")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find optimizer config file {fn}")
    with open(fn) as f:
        optimizer_config = json.load(f)

    # Create optimizer and compile
    optimizer = tf.keras.optimizers.deserialize({
        "class_name": optimizer_config["name"],
        "config": optimizer_config
    })
    model.compile(optimizer=optimizer)

    # Build optimizer
    H, W, C = model.image_shape
    dummy_images = tf.zeros((1, H, W, C), dtype=tf.float32)
    model.train_step(dummy_images)

    # Load optimizer weights file
    fn = os.path.join(dirpath, "optimizer_weights.npy")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find optimizer weights file {fn}")
    optimizer_weights = np.load(
        os.path.join(dirpath, "optimizer_weights.npy"),
        allow_pickle=True
    )

    # Set weights on optimizer
    for var, saved in zip(optimizer.variables, optimizer_weights):
        var.assign(saved)


def load_diffusion_model(dirpath, ema_net_only=False):
    """
    Creates a diffusion model from the following files in directory `dir_path`:
        - Model configuration:  "config.json"
        - U-net model:  "u_net.keras"
        - EMA network:  "ema_net.keras"

    These files are created when calling the `save()` method of a diffusion model.
    """

    if not os.path.isdir(dirpath):
        raise ValueError(f"Unable to find diffusion model directory {dirpath}")

    # Load the configuration file
    fn = os.path.join(dirpath, "model_config.json")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find diffusion model configuration file {fn}")
    with open(fn, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Create the diffusion model
    model = DiffusionModel(config)

    # Load U-Net weights
    if not ema_net_only:
        fn = os.path.join(dirpath, "u_net.weights.h5")
        if not os.path.isfile(fn):
            raise ValueError(f"Unable to find U-Net model weights {fn}")
        model.u_net.load_weights(fn)

    # Load EMA weights
    fn = os.path.join(dirpath, "ema_net.weights.h5")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find EMA model weights {fn}")
    model.ema_net.load_weights(fn)

    return model


def print_trainable_variables(model, params_only=False):
    """
    Prints the trainable variables of a model (name, shape, number of parameters)
    """

    if not params_only:
        print("\n" + "=" * 80)
        print(f"  Trainable variables")
        print("=" * 80 + "\n")

    headers = ["Variable", "Shape", "#Params"]
    data = []
    total_params = 0

    for var in model.trainable_variables:
        num_params = int(np.prod(var.shape))
        total_params += num_params
        data.append([var.name, var.shape, f"{num_params:,.0f}"])

    if not params_only:
        print(tabulate(data, headers=headers, tablefmt="pipe", colalign=("left", "center", "right")))
    print(f"Trainable parameters: {total_params:,.0f}")


class SaveCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, period=1, save_optimizer=False):
        super().__init__()
        self.save_dir = save_dir
        self.period = period
        self.save_optimizer = save_optimizer

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:

            # Create the checkpoint directory
            epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save the U-Net and EMA weights
            self.model.u_net.save_weights(os.path.join(epoch_dir, "u_net.weights.h5"))
            self.model.ema_net.save_weights(os.path.join(epoch_dir, "ema_net.weights.h5"))

            if self.save_optimizer:
                # Save optimizer config
                optimizer_config = self.model.optimizer.get_config()
                with open(os.path.join(epoch_dir, "optimizer_config.json"), "w") as f:
                    json.dump(optimizer_config, f)

                # Save optimizer weights
                optimizer_weights = [v.numpy() for v in self.model.optimizer.variables]
                np.save(
                    os.path.join(epoch_dir, "optimizer_weights.npy"),
                    optimizer_weights,
                    allow_pickle=True
                )

            print(f"\nSaved checkpoint for epoch {epoch+1} to {epoch_dir}")
