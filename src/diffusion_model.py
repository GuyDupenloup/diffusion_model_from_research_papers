# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import math
import json
import numpy as np
import tensorflow as tf
from u_net import UNet


class DiffusionModel(tf.keras.models.Model):
    """
    Diffusion model, including methods to:
        - Create the U-Net and EMA networks
        - Train the U-Net and update the EMA network
        - Save the configuration, U-Net and EMA weights to files
        - Sample the EMA model using DDPM and DDIM sampling methods

    The model configuration is passed as a dictionary, as shown below 
    with the default parameter values:
        {
            # U-Net configuration parameters
            "u_net": {
                "image_size": 32,
                "image_channels": 3,
                "base_channels": 128,
                "channel_multiplier": [1, 2, 2, 2],
                "num_resnet_blocks": 2,
                "attn_resolutions": (16,)
                "dropout_rate": 0.0
            },

            # Input images augmentation
            "data_augment": {
                "random_flip": False
            },

            # Linear beta schedule parameters
            "beta_schedule": {
                "timesteps": 1000,
                "beta_start": 0.008,
                "beta_end": 1e-05,
            },

            # Moving average parameters for EMA network
            "ema": {
                "decay": 0.9999
            }
        }

    """

    def __init__(self, diffusion_config, name=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.model_config = self.complete_model_config(diffusion_config)

        # Create U-Net model
        cfg = self.model_config["u_net"]

        image_size = cfg["image_size"]
        image_channels = cfg["image_channels"]
        self.image_shape = (image_size, image_size, image_channels)
        self.data_augment = self.model_config["data_augment"]

        self.u_net = UNet(
            image_size=image_size,
            image_channels=image_channels,
            base_channels=cfg["base_channels"],
            channel_multiplier=cfg["channel_multiplier"],
            num_resnet_blocks=cfg["num_resnet_blocks"],
            attn_resolutions=cfg["attn_resolutions"],
            dropout_rate=cfg["dropout_rate"],
            name="u_net"
        )

        # Build U-Net using dummy inputs
        dummy_data = (
            tf.random.uniform((2,) + self.image_shape),   # Images
            tf.constant([0, 1], dtype=tf.int32)           # Diffusion times
        )
        _ = self.u_net(dummy_data)

        # Create the EMA model (easier than cloning the U-Net
        # as it avoids model serialization issues)
        self.ema_net = UNet(
            image_size=image_size,
            image_channels=image_channels,
            base_channels=cfg["base_channels"],
            channel_multiplier=cfg["channel_multiplier"],
            num_resnet_blocks=cfg["num_resnet_blocks"],
            attn_resolutions=cfg["attn_resolutions"],
            dropout_rate=cfg["dropout_rate"],
            name="ema_net"
        )
        _ = self.ema_net(dummy_data)
        self.ema_net.set_weights(self.u_net.get_weights())
        self.ema_decay = self.model_config["ema"]["decay"]

        # Loss metric tracker for training
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # Generate beta values using linear schedule
        cfg = self.model_config["beta_schedule"]
        self.timesteps = cfg["timesteps"]
        self.betas, self.alpha_bar = self.linear_beta_schedule(
            self.timesteps, cfg["beta_start"], cfg["beta_end"]
        )

    def complete_model_config(self, diffusion_cfg):
        """
        Completes the model configuration passed in argument using default values.
        """

        # U-Net configuration parameters
        u_net_defaults = {
            "image_size": 32,
            "image_channels": 3,
            "base_channels": 128,
            "channel_multiplier": (1, 2, 2, 2),
            "num_resnet_blocks": 2,
            "attn_resolutions": (16,),
            "dropout_rate": 0.1
        }

        # Input images augmentation
        data_augment_defaults = {
            "random_flip": False
        }

        # Beta schedule parameters
        beta_schedule_defaults = {
            "timesteps": 1000,
            "beta_start": 1e-4,
            "beta_end": 0.02
        }

        # Moving average parameters for EMA network
        ema_defaults = {
            "decay": 0.9999
        }

        u_net = diffusion_cfg.get("u_net", {})
        data_augment = diffusion_cfg.get("data_augment", {})
        beta_schedule = diffusion_cfg.get("beta_schedule", {})
        ema = diffusion_cfg.get("ema", {})
        
        config = {
            "u_net": {**u_net_defaults, **(u_net or {})},
            "data_augment": {**data_augment_defaults, **(data_augment or {})},
            "beta_schedule": {**beta_schedule_defaults, **(beta_schedule or {})},
            "ema": {**ema_defaults, **(ema or {})}
        }

        return config


    def save(self, dirpath, ema_net_only=False):
        """
        Saves the following files to the `dirpath` directory:
            - model_config.json         Model configuration dict
            - u_net.weights.h5          U-net model weights
            - ema_net.weights.h5        EMA model weights

        The diffusion model can be recreated from these three files.

        The directory is created if it does not exist. If it does and 
        contains files with the names used to save the model, they 
        will be overwritten.

        Arguments:
            dirpath (str): Path to the directory where to save files.
        """

        # Create the directory where files will be saved
        os.makedirs(dirpath, exist_ok=True)

        # Save model configuration to JSON file
        with open(os.path.join(dirpath, "model_config.json"), "w") as f:
            json.dump(self.model_config, f, indent=4)

        # Save the weights of the U-Net
        if not ema_net_only:
            self.u_net.save_weights(os.path.join(dirpath, "u_net.weights.h5"))

        # Save the weights of the EMA network
        self.ema_net.save_weights(os.path.join(dirpath, "ema_net.weights.h5"))


    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        """
        Generates beta values using a linear schedule,
        as in the original DDPM paper by Ho et al.

        Arguments:
            timesteps (int): Number of diffusion steps (T).
            beta_start (float): Starting beta value.
            beta_end (float): Ending beta value.

        Returns:
            betas: Linearly spaced beta values from 0 to timesteps.
                   A tensor with shape (timesteps).
            alpha_bar: Cumulative products of (1 - betas).
                   A tensor with shape (timesteps). 
        """

        # Linearly spaced betas from beta_start to beta_end.
        betas = tf.linspace(beta_start, beta_end, timesteps)

        # Derive alpha_bar (cumulative products of (1 - beta_t))
        alphas = 1.0 - betas
        alpha_bar = tf.math.cumprod(alphas)

        return betas, alpha_bar


    def update_ema_weights(self):
        """
        Gets the weights from the U-Net and update EMA's moving averages.
        """

        # Linearly interpolate between online weights and EMA weights
        for weight, ema_weight in zip(self.u_net.trainable_variables, 
                                    self.ema_net.trainable_variables):
            ema_weight.assign(
                self.ema_decay * ema_weight + (1.0 - self.ema_decay) * weight
            )


    def train_step(self, images):
        """
        Performs a single training step for the DDPM model.

        This method:
            - Applies data augmentation (random horizontal flip if enabled).
            - Samples timesteps and Gaussian noise.
            - Corrupts input images to create noisy versions.
            - Predicts noise using the U-Net and computes MSE loss.
            - Updates the U-Net weights via gradient descent.
            - Updates the EMA (Exponential Moving Average) of the U-Net weights.
            - Logs the loss for tracking.

        Arguments:
            images (4D tensor): A batch of input images.

        Returns:
            A dictionary providing the current loss value, e.g. {'loss': 0.1})
        """

        # Randomly flip images horizontally
        if self.data_augment["random_flip"]:
            images = tf.image.random_flip_left_right(images)

        batch_size = tf.shape(images)[0]

        # Sample t
        t = tf.random.uniform((batch_size,), minval=0, maxval=self.timesteps, dtype=tf.int32)

        # Sample Gaussian noise
        noises = tf.random.normal(tf.shape(images))

        # Get alpha_bar_t
        alpha_bar_t = tf.gather(self.alpha_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

        # Create noisy image x_t
        noisy_images = (
            tf.sqrt(alpha_bar_t) * images +
            tf.sqrt(1.0 - alpha_bar_t) * noises
        )

        # Predict noises using the U-Net and calculate the MSE loss
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


    def ddpm_sampling(self, num_samples, keep_all_images=False):
        """
        Implements the sampling algorithm from the DDPM paper (Ho et al.),
        using the EMA-averaged U-Net weights for denoising.

        Arguments:
            num_samples(int) : Number of images to generate.
            keep_all_images (bool):
                If False, only the final denoised images (t=0) are returned.
                If True, all intermediate images (t=T-1 to t=0) are returned
                (useful for viewing the denoising process).

        Returns:
            A tensor with shape:
                (num_samples, H, W, C) if keep_all_images=False.
                (num_samples, T, H, W, C) if keep_all_images=True.
        """

        alphas = 1.0 - self.betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.concat([[1.0], alphas_cumprod[:-1]], axis=0)
        sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod - 1.0)

        # Precompute posterior variance and mean coefficients for the reverse process
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef1 = (
            self.betas * tf.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Initialize random noise for the starting point (t = T)
        batch_shape = (num_samples,) + self.image_shape
        images = tf.random.normal(batch_shape)

        # If keeping all images, prepare a tensor to store intermediate results
        if keep_all_images:
            H, W, C = self.image_shape
            all_images = tf.zeros((self.timesteps, num_samples, H, W, C), dtype=tf.float32)

        for t in reversed(range(self.timesteps)):

            # Predict noise with EMA network at current timestep
            t_tensor = tf.fill((num_samples,), t)
            predicted_noise = self.ema_net((images, t_tensor), training=False)
            
            # Reconstruct x_0 from the current noisy image and predicted noise
            x_recon = (
                sqrt_recip_alphas_cumprod[t] * images - 
                sqrt_recipm1_alphas_cumprod[t] * predicted_noise
            )
            x_recon = tf.clip_by_value(x_recon, -1, 1)

            # Compute the mean of the reverse process distribution
            model_mean = (
                posterior_mean_coef1[t] * x_recon + 
                posterior_mean_coef2[t] * images
            )
        
            # Add noise for timesteps > 0
            if t > 0:
                noise = tf.random.normal(batch_shape)
                model_log_variance = tf.math.log(posterior_variance[t])
                images = model_mean + tf.exp(0.5 * model_log_variance) * noise
            else:
                images = model_mean
        
            # Store intermediate images if required
            if keep_all_images:
                images_expanded = tf.expand_dims(images, axis=0)
                all_images = tf.tensor_scatter_nd_update(
                    all_images, [[t]], images_expanded
                )

        # Return all images or just the final ones
        if keep_all_images:
            return tf.transpose(all_images, perm=[1, 0, 2, 3, 4])
        else:
            return images
        

    def ddim_sampling(self, num_samples, num_steps=100, eta=0.0):
        """
        Implements the sampling algorithm from the DDIM paper (Song et al.),
        using the EMA-averaged U-Net weights for denoising.

        Arguments:
            num_samples (int): Number of samples to generate.
            num_steps (int): Number of timesteps used during the reverse diffusion process.
            eta (float): 
                Controls the stochasticity of the sampling process:
                    - eta = 0: purely deterministic
                    - eta > 0: some stochasticity
                    - eta = 1: same behavior as DDPM sampling
        """

        batch_shape = (num_samples,) + self.image_shape
        images = tf.random.normal(batch_shape)

        # Generate denoising timesteps using quadratic subsequence (Song et al.)
        steps = np.array([
            int((i / num_steps) ** 2 * self.timesteps)
            for i in range(num_steps)
        ])
        steps = np.clip(steps, 0, self.timesteps - 1)
        steps = tf.cast(steps[::-1], tf.int32)  # go from T -> 0

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

            x0_pred = tf.clip_by_value(x0_pred, -1, 1)

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

            # DDIM update
            dir_xt = tf.sqrt(
                1.0 - alpha_bar_prev - sigma**2
            ) * eps

            images = (
                tf.sqrt(alpha_bar_prev) * x0_pred +
                dir_xt +
                sigma * noise
            )

        return images
