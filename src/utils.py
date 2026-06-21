# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
from tabulate import tabulate
import numpy as np
import tensorflow as tf
from diffusion_model import DiffusionModel


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


def load_diffusion_model(dirpath, ema_net_only=False):
    """
    Creates a diffusion model from the following files in directory `dir_path`:
        - Model configuration:  "config.json"
        - U-net model:  "u_net.keras"
        - EMA network:  "ema_net.keras"

    These files are created when calling the `save()` method of a diffusion model.
    """

    # Check that the model directory exist
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Unable to find diffusion model directory {dirpath}")

    # Load the configuration file
    fn = os.path.join(dirpath, "model_config.json")
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Unable to find diffusion model configuration file {fn}")
    with open(fn, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Create the diffusion model
    model = DiffusionModel(config)

    # Load U-Net weights
    if not ema_net_only:
        fn = os.path.join(dirpath, "u_net.weights.h5")
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"Unable to find U-Net model weights {fn}")
        model.u_net.load_weights(fn)

    # Load EMA weights
    fn = os.path.join(dirpath, "ema_net.weights.h5")
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Unable to find EMA model weights {fn}")
    model.ema_net.load_weights(fn)

    return model


def save_model_optimizer(dirpath, model):

    # Create the output directory if it does not exist
    os.makedirs(dirpath, exist_ok=True)
    
    # Save optimizer config
    config = model.optimizer.get_config()
    with open(os.path.join(dirpath, "optimizer_config.json"), "w") as f:
        json.dump(config, f)

    # Save optimizer weights
    weights = [v.numpy() for v in model.optimizer.variables]
    np.savez(
        os.path.join(dirpath, "optimizer_weights.npz"),
        *weights
    )


def restore_model_optimizer(dirpath, model):

    # Check that the optimizer directory exists
    if not os.path.isdir(dirpath):
        raise ValueError(f"Unable to find diffusion model directory {dirpath}")
    
    # Load optimizer config file
    fn = os.path.join(dirpath, "optimizer_config.json")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find optimizer configuration file {fn}")
    with open(fn) as f:
        config = json.load(f)

    # Create optimizer and compile the model
    optimizer = tf.keras.optimizers.deserialize({
        "class_name": config["name"],
        "config": config
    })
    model.compile(optimizer=optimizer)

    # Build optimizer (train step with dummy inputs)
    H, W, C = model.image_shape
    dummy_images = tf.zeros((1, H, W, C), dtype=tf.float32)
    model.train_step(dummy_images)

    # Load optimizer weights
    fn = os.path.join(dirpath, "optimizer_weights.npz")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find optimizer config file {fn}")
    data = np.load(fn)
    weights = [data[f"arr_{i}"] for i in range(len(data))]

    # Set weights on optimizer
    for var, saved in zip(optimizer.variables, weights):
        var.assign(saved)


class SaveCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, period=1):
        super().__init__()
        self.save_dir = save_dir
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:

            # Create checkpoint directory
            epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:04d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save model and optimizer
            self.model.save(epoch_dir)
            save_model_optimizer(epoch_dir, self.model)

            print(f"\nSaved checkpoint for epoch {epoch+1} to {epoch_dir}")
