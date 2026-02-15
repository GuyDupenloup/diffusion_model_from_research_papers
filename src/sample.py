# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from diffusion_model import DiffusionModel


def load_ema_model(model_dir):

    # Check that the model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'unable to find model directory {model_dir}')

    # Check that the config and EMA model files are present
    config_filepath = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(os.path.join(model_dir, 'config.json')):
        raise FileNotFoundError(f'unable to find configuration file {config_filepath}')

    ema_filepath = os.path.join(model_dir, 'ema_net.keras')
    if not os.path.isfile(os.path.join(model_dir, 'ema_net.keras')):
        raise FileNotFoundError(f'unable to find EMA model file {ema_filepath}')

    # Read JSON config file
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Create diffusion model
    model = DiffusionModel(config)

    # Load EMA model
    ema_filepath = os.path.join(model_dir, 'ema_net.keras')
    print(f'>> Loading EMA network {ema_filepath}')
    model.load_ema_net(ema_filepath)
 
    return model


def sample_model(
    model_dir,
    num_samples=32,
    samples_filepath='samples.npy',
    method='ddim',
    num_steps=100,
    eta=0.0
):

    # Load the EMA model
    model = load_ema_model(model_dir)

    if method == 'ddpm':
        samples = model.ddpm_sampling(num_samples)
    else:
        samples = model.ddim_sampling(num_samples, num_steps=num_steps, eta=eta)

    samples = np.array(samples, np.float32)
    np.save(samples_filepath, samples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_dir',
        help='Directory where the .json config file and .keras EMA model file are',
        type=str
    )
    parser.add_argument(
        '--num_samples',
        help="Number of samples to generate",
        type=int,
        default=32
    )
    parser.add_argument(
        '--samples_filepath',
        help='Generated samples .npy file',
        type=str,
        default='samples.npy'
    )
    parser.add_argument(
        '--method',
        help="Sampling method: 'ddpm' or 'ddim'",
        type=str,
        default='ddim'
    )
    parser.add_argument(
        '--num_steps',
        help='DDIM method: number of sampling steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--eta',
        help='DDIM method: stochasticity factor (deterministic if eta=0.0)',
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    if args.method not in ('ddpm', 'ddim'):
        raise ValueError("Argument `method` must be set to 'ddpm' or 'ddim'")

    sample_model(
        args.model_dir,
        num_samples=args.num_samples,
        samples_filepath=args.samples_filepath,
        method=args.method,
        num_steps=args.num_steps,
        eta=args.eta
    )
