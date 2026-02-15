# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import shutil
from tabulate import tabulate
import numpy as np
import tensorflow as tf


def print_trainable_variables(model, params_only=False):
    """
    Prints the trainable variables of a model (name, shape, number of parameters)
    """

    if not params_only:
        print('\n' + '=' * 80)
        print(f"  Trainable variables")
        print('=' * 80 + '\n')

    headers = ['Variable', 'Shape', '#Params']
    data = []
    total_params = 0

    for var in model.trainable_variables:
        num_params = int(np.prod(var.shape))
        total_params += num_params
        data.append([var.name, var.shape, f'{num_params:,.0f}'])

    if not params_only:
        print(tabulate(data, headers=headers, tablefmt='pipe', colalign=('left', 'center', 'right')))
    print(f'Trainable parameters: {total_params:,.0f}')


class SaveCheckpoint(tf.keras.callbacks.Callback):
    """
    Callback that saves U-Net and EMA models at regular epoch intervals.
    Models are saved to .keras files to save the optimizer.
    """

    def __init__(self, dirpath, basename='checkpoint', period=1, offset=0, overwrite=False):
        """
        Arguments:
            dirpath: Path to the directory where to save models.
            basename: Prefix to use for the model filenames.
            period: Epoch interval between two saves.
            offset: Integer added to the epoch number (useful when resuming a training).
            overwrite: If True, Overwrite the directory if it already exists.
                       Otherwise, raise an error.

        The paths to the model files follow these patterns:
            f'{basename}_u_net_{epoch + offset + 1}.keras'
            f'{basename}_ema_net_{epoch + offset + 1}.keras'
        """

        super().__init__()
        self.dirpath = dirpath
        self.basename = basename
        self.period = period
        self.offset = offset
        self.pattern = os.path.join(dirpath, basename)

        if os.path.isdir(dirpath):
            if not overwrite:
                print(f'Unable to save checkpoints in {dirpath}. Directory already exists.')
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            # Save the U-Net
            fn = f'{self.basename}_u_net_{epoch + self.offset + 1}.keras'
            self.model.u_net.save(os.path.join(self.dirpath, fn))

            # Save the EMA network
            fn = f'{self.basename}_ema_net_{epoch + self.offset + 1}.keras'
            self.model.ema_net.save(os.path.join(self.dirpath, fn))
