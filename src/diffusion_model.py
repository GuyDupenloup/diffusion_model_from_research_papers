# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import shutil
import math
import json
import tensorflow as tf
from u_net import UNet


# @tf.keras.utils.register_keras_serializable()
class DiffusionModel(tf.keras.models.Model):
    """
    Diffusion model, including methods to:
        - train the U-Net
        - Update weights into the EMA network.
        """

    def __init__(self, diffusion_config, name=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.model_config = self.get_model_config(diffusion_config)

        # Create U-Net model
        cfg = self.model_config['u_net']

        image_size = cfg['image_size']
        image_channels = cfg['image_channels']
        self.image_shape = (image_size, image_size, image_channels)
        self.data_augment = self.model_config['data_augment']

        self.u_net = UNet(
            image_size=image_size,
            image_channels=image_channels,
            base_channels=cfg['base_channels'],
            channel_multiplier=cfg['channel_multiplier'],
            num_resnet_blocks=cfg['num_resnet_blocks'],
            attn_resolutions=cfg['attn_resolutions'],
            dropout_rate=cfg['dropout_rate'],
            name='u_net'
        )

        # Build U-Net using dummy inputs
        dummy_data = (
            tf.random.uniform((2,) + self.image_shape),   # Images
            tf.constant([0, 1], dtype=tf.int32)           # Diffusion times
        )
        _ = self.u_net(dummy_data)

        # Create a copy of the U-Net for EMA
        self.ema_decay = self.model_config['ema']['decay']

        self.ema_net = tf.keras.models.clone_model(self.u_net)
        _ = self.ema_net(dummy_data)
        weights = self.u_net.get_weights()
        self.ema_net.set_weights(weights)

        # Loss metric tracker for training
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        # Generate beta values using cosine schedule
        cfg = self.model_config['beta_schedule']
        self.timesteps = cfg['timesteps']
        self.betas, self.alpha_bar = self.cosine_beta_schedule(
            self.timesteps, s=cfg['s'], beta_min=cfg['beta_min'], beta_max=cfg['beta_max']
        )

        # Precompute all sampling coefficients
        alphas_cumprod = self.alpha_bar[1:]
        alphas_cumprod_prev = self.alpha_bar[:-1]
        alphas = 1.0 - self.betas
        
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * tf.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod - 1.0)
        self.posterior_log_variance_clipped = tf.math.log(
            tf.maximum(self.posterior_variance, 1e-20)
        )

    def get_model_config(self, diffusion_cfg):
        """
        Completes the configuration passed in argument
        using default values wherever needed.
        """

        u_net_defaults = {
            'image_size': 32,
            'image_channels': 3,
            'base_channels': 128,
            'channel_multiplier': (1, 2, 2, 2),
            'num_resnet_blocks': 2,
            'attn_resolutions': (16,),
            'dropout_rate': 0.0
        }
        data_augment_defaults = {
            'random_flip': False
        }
        beta_schedule_defaults = {
            'timesteps': 1000,
            's': 0.008,
            'beta_min': 1e-5,
            'beta_max': 0.999
        }
        ema_defaults = {
            'decay': 0.999
        }

        u_net = diffusion_cfg.get('u_net', {})
        data_augment = diffusion_cfg.get('data_augment', {})
        beta_schedule = diffusion_cfg.get('beta_schedule', {})
        ema = diffusion_cfg.get('ema', {})
        
        config = {
            'u_net': {**u_net_defaults, **(u_net or {})},
            'data_augment': {**data_augment_defaults, **(data_augment or {})},
            'beta_schedule': {**beta_schedule_defaults, **(beta_schedule or {})},
            'ema': {**ema_defaults, **(ema or {})}
        }

        return config

    def save(self, dirpath, overwrite=True):
        """
        Save the model configuration directory to a JSON file,
        and the U-Net and EMA network in .keras files.

        Arguments:
            dirpath: Path to the directory where to save the configuration and model files.
            overwrite: If True, the directory is overwritten if it already exists.
                       Otherwise, an error is raised.
        """

        if os.path.isdir(dirpath):
            if not overwrite:
                raise ValueError(f'Unable to save diffusion model. Directory {dirpath} already exists.')
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)

        # Save configuration to JSON file
        with open(os.path.join(dirpath, 'config.json'), 'w') as f:
            json.dump(self.model_config, f, indent=4)

        # Save U-Net to keras file
        self.u_net.save(os.path.join(dirpath, 'u_net.keras'))

        # Save EMA network to keras file
        self.ema_net.save(os.path.join(dirpath, 'ema_net.keras'))

    def load_u_net(self, filepath):
        """
        Loads an EMA model file in the diffusion model.
        """
        if not os.path.isfile(filepath):
                raise FileNotFoundError(f'Unable to find U-Net model file {filepath}')
        self.u_net = tf.keras.models.load_model(
                filepath,
                custom_objects=self.u_net.custom_objects
            )

    def load_ema_net(self, filepath):
        """
        Loads a U-Net model file in the diffusion file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'Unable to find EMA model file {filepath}')
        self.ema_net = tf.keras.models.load_model(
            filepath,
            custom_objects=self.ema_net.custom_objects
        )
 
    def cosine_beta_schedule(self, timesteps, s, beta_min, beta_max):
        """
        Generates beta values using a cosine schedule.

        Arguments:
            timesteps: Number of diffusion steps.
            s: 
            beta_min, beta_max: Valid beta value range (used for clipping).
        """

        # t in [0, T]
        steps = timesteps + 1
        t = tf.linspace(0.0, timesteps, steps)

        # cosine alpha_bar           ==> this alpha_bar(t)  ??
        alpha_bar = tf.cos(
            ((t / timesteps) + s) / (1.0 + s) * (math.pi / 2)
        ) ** 2

        # Normalize so alpha_bar[0] = 1
        alpha_bar = alpha_bar / alpha_bar[0]

        # Derive betas and ensure that they are in specified range
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = tf.clip_by_value(betas, beta_min, beta_max)

        return betas, alpha_bar

    def update_ema_weights(self):
        """
        Gets the weights from the U-Net and integrates them in the EMA network.
        """

        # Linearly interpolate between online weights and EMA weights
        for weight, ema_weight in zip(self.u_net.trainable_variables, 
                                    self.ema_net.trainable_variables):
            ema_weight.assign(
                self.ema_decay * ema_weight + (1.0 - self.ema_decay) * weight
            )

    def train_step(self, images):
        """
        Runs a training step for an input batch of images.
        """

        # Randomly flip images horizontally
        if self.data_augment['random_flip']:
            images = tf.image.random_flip_left_right(images)

        batch_size = tf.shape(images)[0]
        alpha_bar = tf.math.cumprod(1 - self.betas)

        # Sample t
        t = tf.random.uniform((batch_size,), minval=0, maxval=self.timesteps, dtype=tf.int32)

        # Sample Gaussian noise
        noises = tf.random.normal(tf.shape(images))

        # Get alpha_bar_t
        alpha_bar_t = tf.gather(alpha_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

        # Create noisy image x_t
        noisy_images = (
            tf.sqrt(alpha_bar_t) * images +
            tf.sqrt(1.0 - alpha_bar_t) * noises
        )

        # Predict noises using the U-Net and calculate the loss
        with tf.GradientTape() as tape:
            pred_noises = self.u_net([noisy_images, t], training=True)
            loss = tf.reduce_mean(tf.square(noises - pred_noises))

        gradients = tape.gradient(loss, self.u_net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.u_net.trainable_weights))

        # Update the EMA weights
        self.update_ema_weights()

        # Update the loss tracker
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def ddpm_sampling(self, num_samples):
        
        alphas = 1.0 - self.betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.concat([[1.0], alphas_cumprod[:-1]], axis=0)
        
        # Precompute posterior coefficients (like Ho's code)
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef1 = (
            self.betas * tf.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        
        # Coefficients for predicting x_0 from noise
        sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod - 1.0)
        
        batch_shape = (num_samples,) + self.image_shape
        images = tf.random.normal(batch_shape)
        
        for t in reversed(range(self.timesteps)):
            
            t_tensor = tf.fill((num_samples,), t)
            predicted_noise = self.ema_net((images, t_tensor), training=False)
            
            # Step 1: Predict x_0 from noise (like Ho's predict_start_from_noise)
            x_recon = (
                sqrt_recip_alphas_cumprod[t] * images - 
                sqrt_recipm1_alphas_cumprod[t] * predicted_noise
            )
            # x_recon = tf.clip_by_value(x_recon, -1.0, 1.0)
            
            # Step 2: Compute posterior mean (like Ho's q_posterior)
            model_mean = (
                posterior_mean_coef1[t] * x_recon + 
                posterior_mean_coef2[t] * images
            )
            
            # Step 3: Add noise (like Ho's p_sample)
            if t > 0:
                noise = tf.random.normal(batch_shape)
                model_log_variance = tf.math.log(posterior_variance[t])
                images = model_mean + tf.exp(0.5 * model_log_variance) * noise
            else:
                images = model_mean
        
        images = tf.clip_by_value(images, 0.0, 1.0)

        return images

    def ddim_sampling(self, num_samples, num_steps=100, eta=0.0):

        batch_shape = (num_samples,) + self.image_shape
        images = tf.random.normal(batch_shape)

        # ---- FIXED TIMESTEP SCHEDULE ----
        steps = tf.linspace(0.0, tf.cast(self.timesteps - 1, tf.float32), num_steps)
        steps = tf.cast(steps, tf.int32)
        steps = tf.reverse(steps, axis=[0])  # go from T -> 0

        for i in range(num_steps):

            t = steps[i]
            t_tensor = tf.fill((num_samples,), t)

            # Predict noise with EMA model
            eps = self.ema_net((images, t_tensor), training=False)

            # Previous timestep
            if i < num_steps - 1:
                t_prev = steps[i + 1]
            else:
                t_prev = tf.constant(0, dtype=tf.int32)

            t_prev_tensor = tf.fill((num_samples,), t_prev)

            alpha_bar_t = tf.gather(self.alpha_bar, t_tensor)
            alpha_bar_prev = tf.gather(self.alpha_bar, t_prev_tensor)

            # Reshape for broadcasting
            alpha_bar_t = tf.reshape(alpha_bar_t, [num_samples, 1, 1, 1])
            alpha_bar_prev = tf.reshape(alpha_bar_prev, [num_samples, 1, 1, 1])

            # Predict X0 and clip for stability
            x0_pred = (
                images - tf.sqrt(1.0 - alpha_bar_t) * eps
            ) / tf.sqrt(alpha_bar_t)
            x0_pred = tf.clip_by_value(x0_pred, -1.0, 1.0)

            # Compute sigma (stochasticity)
            if i < num_steps - 1:
                sigma = eta * tf.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
                noise = tf.random.normal(batch_shape) if eta > 0 else tf.zeros(batch_shape)
            else:
                sigma = 0.0
                noise = tf.zeros(batch_shape)

            # ---- STANDARD DDIM UPDATE (corrected) ----
            dir_xt = tf.sqrt(1.0 - alpha_bar_prev) * eps

            images = (
                tf.sqrt(alpha_bar_prev) * x0_pred +
                dir_xt +
                sigma * noise
            )

        return images
