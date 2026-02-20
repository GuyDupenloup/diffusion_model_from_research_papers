# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import numpy as np
from utils import load_diffusion_model


def sample_model(
    model_dir,
    num_samples=32,
    samples_filepath='samples.npy',
    method='ddim',
    num_steps=100,
    eta=0.0
):

    # Load the diffusion model and EMA network
    model = load_diffusion_model(model_dir, ema_net_only=True)
    print(f'>> Loaded diffusion model from directory {model_dir}')

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
