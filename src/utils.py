# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
from tabulate import tabulate
import numpy as np
import tensorflow as tf
from scipy import linalg
from diffusion_model import DiffusionModel


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
    fn = os.path.join(dirpath, "config.json")
    if not os.path.isfile(fn):
        raise ValueError(f"Unable to find diffusion model configuration file {fn}")
    with open(fn, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Create the diffusion model
    model = DiffusionModel(config)

    # Use dummy inputs to build the model

    
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


class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, period=1):
        super().__init__()
        self.save_dir = save_dir
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            self.model.u_net.save_weights(os.path.join(epoch_dir, "u_net.weights.h5"))
            self.model.ema_net.save_weights(os.path.join(epoch_dir, "ema_net.weights.h5"))
            print(f"\nSaved weights for epoch {epoch+1} to {epoch_dir}")
