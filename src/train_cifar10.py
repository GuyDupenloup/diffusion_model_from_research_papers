# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
from diffusion_model import DiffusionModel
from utils import SaveCheckpoint


def create_data_loader(x, batch_size):
    """
    Creates a tf.data.Dataset to load images (labels are discarded)
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


def train_model(output_dir):

    # Load CIFAR-10
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    
    # Create data loader for training set images
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = create_data_loader(x_train, batch_size=128)

    # Create diffusion model
    model = DiffusionModel({
        'u_net': {
            'image_size': 32,
            'image_channels': 3,
            'base_channels': 128,
            'channel_multiplier': (1, 2, 2, 2),
            'num_resnet_blocks': 1,
            'attn_resolutions': (16,),
            'dropout_rate': 0.1
        },
        'beta_schedule': {
            'timesteps': 1000
        }
    })
    
    # Create the output dir if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # resume_from_epoch = 0
    resume_from_epoch = 45

    if resume_from_epoch > 0:
        print('>> Resuming from epoch', resume_from_epoch)
        print('checkpoints_dir:', checkpoints_dir)
        checkpoints_dir = os.path.join(output_dir, f'checkpoints_{resume_from_epoch}')
        if not os.path.isdir(checkpoints_dir):
            raise ValueError("Can't find checkpoints directory", checkpoints_dir)

        fn = os.path.join(checkpoints_dir, f'checkpoint_u_net_{resume_from_epoch}.keras')
        model.u_net = tf.keras.models.load_model(fn)
        print('Loaded', fn)

        fn = os.path.join(checkpoints_dir, f'checkpoint_ema_net_{resume_from_epoch}.keras')
        model.ema_net = tf.keras.models.load_model(fn)
        print('>> Loaded', fn)

    # Don't pass a loss function, the model handles it.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4)
    )

    # Set up callbacks
    callbacks = [
        SaveCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            basename='checkpoint',
            period=5,
            offset=resume_from_epoch,
            overwrite=True
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dir, 'metrics.csv')
        )
    ]

    # Train model
    print('>> Starting training')
    start_time = timer()
    model.fit(
        train_ds,
        epochs=100,
        callbacks=callbacks,
    )
    end_time = timer()
    train_run_time = int(end_time - start_time)
    print('>> Training runtime: ' + str(timedelta(seconds=train_run_time))) 

    # Save the config file and the two models (U-Net and EMA)
    model.save(os.path.join(output_dir, 'trained_model'), overwrite=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help='Directory where to save training output files (model config, checkpoint, etc.)',
        type=str,
        default='./train_output'
    )

    args = parser.parse_args()
    train_model(args.output_dir)
