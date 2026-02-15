# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
import numpy as np
from diffusion_model import DiffusionModel


def restore_model(model_dir):

    # Read the JSON config file
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Create diffusion model
    model = DiffusionModel(config)

    # Load EMA model
    model.load_ema_net(os.path.join(model_dir, 'ema_net.keras'))
 
    return model


model_dir = 'train_output/trained_model'

model = restore_model(model_dir)
samples = model.ddpm_sampling(20)

samples = np.array(samples, np.float32)
np.save('samples.np', samples)
