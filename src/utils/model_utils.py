# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
from tabulate import tabulate
import numpy as np
from diffusion_model import DiffusionModel


def print_trainable_variables(model, params_only=False):
    """
    Prints the trainable variables of a model (name, shape, number of parameters).
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

    # Check that the model directory exists
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
