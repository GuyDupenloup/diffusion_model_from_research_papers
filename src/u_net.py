# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import math
import numpy as np
import tensorflow as tf


def default_init(scale):
    return tf.initializers.variance_scaling(
        scale=1e-10 if scale == 0 else scale,
        mode='fan_avg',
        distribution='uniform'
)


@tf.keras.utils.register_keras_serializable()
class NiNLayer(tf.keras.layers.Layer):
    """
    Network-in-Network (NIN) layer for U-Net shortcut connections and self-attention block.
    Implemented as a 1x1 convolution operation.
    
    Layer arguments:
        num_units (integer): Number of units/channels.
        init_scale (float): Scale factor for weight initialization.

    Layer call() method:
        Inputs: Feature map, a 4D tensor.
        Returns: Resampled feature map, a 4D tensor.
    """
            
    def __init__(self, num_units, init_scale=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_units = num_units
        self.init_scale = init_scale

        self.conv2d_1x1 = tf.keras.layers.Conv2D(
            num_units,
            kernel_size=1,
            kernel_initializer=default_init(init_scale),
            padding='same'
        )
        
    def call(self, x):
        return self.conv2d_1x1(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_units': self.num_units,
            'init_scale': self.init_scale
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim

        # Time embedding dense layers
        self.dense0 = tf.keras.layers.Dense(
            embedding_dim, kernel_initializer=default_init(scale=1.), name='dense0'
        )
        self.dense1 = tf.keras.layers.Dense(
            embedding_dim, kernel_initializer=default_init(scale=1.), name='dense1'
        )


    def get_timestep_embedding(self, timesteps):
        """
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """

        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)

        emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)

        if self.embedding_dim % 2 == 1:  # zero pad
            emb = tf.pad(emb, [[0, 0], [0, 1]])
            
        return emb


    def call(self, timesteps):

        temb = self.get_timestep_embedding(timesteps)

        temb = self.dense0(temb)
        temb = tf.nn.silu(temb)
        temb = self.dense1(temb)

        return temb

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class ResamplingLayer(tf.keras.layers.Layer):
    """
    Resampling layer.

    In the down-path of the U-Net, the layer decreases the resolution by a factor a 2
    using a 3x3 convolution with strides=2.

    In the up-path, it increases the resolution by a factor of 2. The feature map is first
    resized using the nearest neighbors method that duplicates rows and columns.
    Then, a 3x3 convolution is applied to smooth the results.

    The layer preserves the number of channels.

    Layer arguments:
        down_path (boolean): If True (False), the layer is in the down-path (up-path) of U-Net.
        channels (integer): Number of channels.

    Layer call() method:
        Inputs: Feature map, a 4D tensor.
        Returns: Resampled feature map, a 4D tensor.
    """

    def __init__(self, downsample, channels, with_conv=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.downsample = downsample
        self.channels = channels
        self.with_conv = with_conv

        if downsample:
            if with_conv:
                self.down_layer = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding='same')
            else:
                self.down_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        else:
            self.up_layer= tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
            if with_conv:
                self.up_conv = tf.keras.layers.Conv2D(channels, kernel_size=3, padding='same')

    def call(self, x):
        if self.downsample:
            x = self.down_layer(x)
        else:
            x = self.up_layer(x)
            if self.with_conv:
                x = self.up_conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'down_path': self.down_path,
            'channels': self.channels
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class ResnetBlock(tf.keras.layers.Layer):
    """
    Residual block.

        The block performs:
            1. Group normalization + non-linearity
            2. 3x3 convolution
            3. Time embeddings addition
            4. Group normalization + non-linearity
            5. Dropout
            6. 3x3 convolution
            7. Block inputs addition (skip connection)

        A NIN layer (1x1 conv) is inserted in the input/output skip connection
        if the numbers of input and output channels are different (otherwise,
        it is a straight connection).

        Layer arguments:
            output_channels (integer): Number of channels the block outputs.
            dropout_rate (float): Dropout rate for the dropout layer.

        Layer call() method:
            Inputs:
                x: feature map, a 4D tensor.
                time_emb: time embeddings, a tensor with shape (B, C).
            Returns:
                Output feature map, a 4D tensor.
        """

    def __init__(self, output_channels, dropout_rate=0.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.output_channels = output_channels
        self.dropout_rate = dropout_rate

        self.norm1 = tf.keras.layers.GroupNormalization(name='norm1')

        self.conv1 = tf.keras.layers.Conv2D(
            output_channels,
            kernel_size=3, 
            kernel_initializer=default_init(scale=1.), 
            padding='same',
            name='conv1'
        )

        self.temb_proj = tf.keras.layers.Dense(
            output_channels, kernel_initializer=default_init(scale=1.), name='temb_proj'
        )

        self.norm2 = tf.keras.layers.GroupNormalization(name='norm2')
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')

        self.conv2 = tf.keras.layers.Conv2D(
            output_channels,
            kernel_size=3,  
            kernel_initializer=default_init(scale=0.), 
            padding='same',
            name='conv2'
        )

        # NIN layer required if the numbers of input and output channels
        # are different. Handled by the build() method of the layer.
        self.shortcut = None


    def build(self, input_shape):
        if input_shape[-1] != self.output_channels:
                self.shortcut = NiNLayer(self.output_channels)
        super().build(input_shape)


    def call(self, x, time_emb, training=None):

        # Save input feature map for residual connection
        h = x

        h = self.norm1(h, training=training)
        h = tf.nn.silu(h)
        h = self.conv1(h)

        # Add in timestep embedding
        time_emb = tf.nn.silu(time_emb)
        time_emb = time_emb[:, None, None, :]   # Broadcast to (B, H, W, C)
        h += self.temb_proj(time_emb)

        h = self.norm2(h, training=training)
        h = tf.nn.silu(h)
        h = self.dropout_layer(h, training=training)
        h = self.conv2(h)

        if self.shortcut is not None:
            # The numbers of input/output channels are different.
            x = self.shortcut(x)

        # Residual connection
        h = x + h

        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class AttentionBlock(tf.keras.layers.Layer):
    """
    Self-attention block with spatial dot-product attention.

        The block performs:
        1. Group normalization
        2. Query, Key, Value projections (1x1 conv)
        3. Scaled dot-product attention across spatial dimensions
        4. Output projection (1x1 conv)
        5. Residual connection

    Layer arguments:
        channels (int): Number of input and output channels (unchanged by the layer).

    Layer call() method:
        Inputs: Feature map, a 4D tensor.
        Returns: Feature map, a 4D tensor with the same shape as the input tensor.
    """

    def __init__(self, channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.channels = channels

        self.norm = tf.keras.layers.GroupNormalization(name='norm')
        self.q = NiNLayer(channels, name='q')
        self.k = NiNLayer(channels, name='k')
        self.v = NiNLayer(channels, name='v')
        self.proj_out = NiNLayer(channels, init_scale=0., name='proj_out')
    

    def call(self, x, training=None):

        B, H, W, C = tf.unstack(tf.shape(x))

        # Normalize input
        h = self.norm(x, training=training)
        
        # Compute query, key, value projections
        q = self.q(h)  # [B, H, W, C]
        k = self.k(h)  # [B, H, W, C]
        v = self.v(h)  # [B, H, W, C]
      
        # Compute attention weights
        # einsum 'bhwc,bHWc->bhwHW' computes dot product between each spatial position
        w = tf.einsum('bhwc,bHWc->bhwHW', q, k)
        
        # Scale by sqrt(C) for stability (standard attention scaling)
        w = w * (tf.cast(C, tf.float32) ** (-0.5))
            
        # Reshape to apply softmax over all spatial positions
        w = tf.reshape(w, [B, H, W, H * W])
        w = tf.nn.softmax(w, axis=-1)
        w = tf.reshape(w, [B, H, W, H, W])
        
        # Apply attention weights to values
        # einsum 'bhwHW,bHWc->bhwc' aggregates values weighted by attention
        h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
      
        # Final projection
        h = self.proj_out(h)
        
        # Residual connection
        h = x + h

        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class UNetStage(tf.keras.layers.Layer):
    """
    U-Net stage comprising ResNet blocks and optionally, attention blocks and a resampling layer.
    
    When the layer is called, a list of skip connections (4D tensors) that operates like a LIFO
    is passed as an argument:
        - In the down-path: new skip connections are appended to the list
            (outputs of ResNet blocks, attention blocks, resampling layers).
        - In the up-path: skip connections are popped from the list and concatenated 
            to the inputs of the ResNet blocks.

    Layer arguments:
        down_path (boolean): If True (False), the stage is in the down-path (up-path) of the U-Net.
        output_channels (integer): Number of channels the stage outputs.
        resolution (integer): Feature map size.
        attn_resolutions (tuple/list of integers): Resolutions where attention blocks must be inserted.
        resample (boolean): If True, a resampling layer is added. If False, no resampling occurs.
        num_res_blocks (integer): Number of ResNet blocks in the stage.
        dropout_rate (float): Dropout rate for the dropout layers inside the ResNet blocks.

    Layer call() method:
        Inputs:
            x: input feature map, a 4D tensor.
            time_emb: time embeddings, a tensor with shape (B, C).
            skip_connections: a list of 4D tensors.

        Returns:
            Output feature map, a 4D tensor.
            Updated skip connections, a list of 4D tensor.
    """
    
    def __init__(self,
        down_path,
        output_channels,
        resolution,
        attn_resolutions, 
        resample,
        num_resnet_blocks=2,
        resample_with_conv=True,
        dropout_rate=0.,
        name=None,
        **kwargs):

        super().__init__(name=name, **kwargs)

        self.down_path = down_path
        self.output_channels = output_channels
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions
        self.resample = resample
        self.num_resnet_blocks = num_resnet_blocks
        self.resample_with_conv = resample_with_conv
        self.dropout_rate = dropout_rate

        self.resnet_blocks = [
            ResnetBlock(output_channels, dropout_rate=dropout_rate, name=f'{self.name}_resnet{i}')
            for i in range(num_resnet_blocks)
        ]

        self.attn_blocks = [
            AttentionBlock(output_channels, name=f'{self.name}_attn{i}') if resolution in attn_resolutions else None
            for i in range(num_resnet_blocks)
        ]
   
        if resample:
            self.resample_layer = ResamplingLayer(
                down_path,
                output_channels,
                with_conv=resample_with_conv,
                name=f'{self.name}_resample')


    def call(self, x, time_emb, skip_connections, training=None):

        skips = skip_connections.copy()
        
        for i in range(self.num_resnet_blocks):
            if not self.down_path:
                x = tf.concat([x, skips.pop()], axis=-1)

            x = self.resnet_blocks[i](x, time_emb, training=training)

            if self.attn_blocks[i]:
                x = self.attn_blocks[i](x, training=training)

            if self.down_path:
                skips.append(x)

        if self.resample:
                x = self.resample_layer(x)
                if self.down_path:
                    skips.append(x)

        return x, skips
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'down_path': self.down_path,
            'output_channels': self.output_channels,
            'resolution': self.resolution,
            'attn_resolutions': self.attn_resolutions,
            'resample': self.resample,
            'num_resnet_blocks': self.num_resnet_blocks,
            'resample_with_conv': self.resample_with_conv,
            'dropout_rate': self.dropout_rate
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class Bottleneck(tf.keras.layers.Layer):
    """
    Bottleneck section of the U-Net (resides between the down and up paths).

    It includes two ResNet blocks with an attention block between them. The spatial resolution 
    and channel depth are preserved.

    Layer arguments:
        channels (integer): Number of channels, constant throughout the bottleneck.
        dropout_rate (float): Dropout rate for the dropout layers inside the ResNet blocks.

    Layer call() method:
        Inputs:
            x: Feature map, a 4D tensor.
            time_emb: Time embeddings, a tensor with shape (B, C)
        Returns:
            Bottleneck output feature map, a 4D tensor.
    """

    def __init__(self, channels, dropout_rate=0., name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.channels = channels
        self.dropout_rate = dropout_rate

        self.resnet1 = ResnetBlock(channels, dropout_rate=dropout_rate, name='resnet1')
        self.attention = AttentionBlock(channels, name=f'attn')
        self.resnet2 = ResnetBlock(channels, dropout_rate=dropout_rate, name='resnet2')

    def call(self, x, time_emb, training=None):

        x = self.resnet1(x, time_emb, training=training)
        x = self.attention(x, training=training)
        x = self.resnet2(x, time_emb, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class UNet(tf.keras.models.Model):
    """
    U-Net from the original DDPM paper.

    Model arguments:
    ---------------
        image_size (integer):
            Size of the input images.

        image_channels (integer):
            Number of channels of the images.

        base_channels (integer):
            Number of channels the input projection convolution outputs.

        channel_multiplier (tuple/list of integers):
            Specifies the number of channels at each resolution as a multiplying factor 
            of the number of base channels.
            The number of elements of `channel_multiplier` gives the number of resolutions.
            When going from one stage to the next, the resolution is divided by 2 in the down path
            and multiplied by 2 in the up path.
            For example, if `image_size`=32, `base_channels`=128 and `channel_multiplier`=(1, 2, 2, 3),
            the resolutions and numbers of channels are as follows:
                32 x 32  -> 128
                16 x 16  -> 256
                8 x 8    -> 256
                4 x 4    -> 384

        num_resnet_blocks (integer):
            Number of residual blocks in the down stages of the U-Net. In the up stages,
            the number of residual blocks is equal to `num_resnet_blocks` + 1.

        attn_resolutions (tuple/list of integers):
            Resolutions where attention blocks are used.
            For example, if `attn_resolutions`=(16,), attention blocks are inserted 
            at the 16x16 resolution.

        dropout_rate (float):
            Dropout rate for the dropout layers in the ResNet blocks.

    Model call() method:
    -------------------
        Inputs:
            x: Input images, a 4D tensor.
            t: timestep index, an integer.

        Returns:
            Images passed through the U-Net, a 4D tensor.
    """

    def __init__(
        self,
        image_size=None,
        image_channels=None,
        base_channels=None,
        channel_multiplier=None,
        num_resnet_blocks=2,
        resample_with_conv=True,
        attn_resolutions=(16,),
        dropout_rate=0.,
        name=None,
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)

        self.image_size = image_size
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_resnet_blocks = num_resnet_blocks
        self.resample_with_conv = resample_with_conv
        self.attn_resolutions = attn_resolutions
        self.dropout_rate = dropout_rate
        
        self.custom_objects = {
            'NiNLayer': NiNLayer,
            'TimeEmbedding': TimeEmbedding,
            'ResamplingLayer': ResamplingLayer,
            'ResnetBlock': ResnetBlock,
            'AttentionBlock': AttentionBlock,
            'UNetStage': UNetStage,
            'Bottleneck': Bottleneck,
            'UNet': UNet
            }
        
        # Resolution sizes
        self.num_resolutions = len(channel_multiplier)
        resolutions = [image_size // (2**i) for i in range(self.num_resolutions)]

        # Time embeddings layer
        self.time_embeddings = TimeEmbedding(base_channels * 4, name='time_emb')

        # Input conv
        self.conv_in = tf.keras.layers.Conv2D(
            base_channels,
            kernel_size=3,
            strides=1, 
            kernel_initializer=default_init(scale=1.), 
            padding='same',
            name='conv_in'
        )
        
        # Down-path stages
        self.down_stages = [
            UNetStage(
                down_path=True,
                output_channels=base_channels * channel_multiplier[i],
                resolution=resolutions[i],
                attn_resolutions=attn_resolutions,
                resample=(i != self.num_resolutions - 1),
                num_resnet_blocks=num_resnet_blocks,
                resample_with_conv=resample_with_conv,
                dropout_rate=dropout_rate,
                name=f'down_block{i}'
            )
            for i in range(self.num_resolutions)
        ]

        # Bottleneck ResNet and attention blocks
        self.bottleneck = Bottleneck(base_channels * channel_multiplier[-1], dropout_rate=dropout_rate, name='bottleneck')

        # Up-path stages
        self.up_stages = [
            UNetStage(
                down_path=False,
                output_channels=base_channels * channel_multiplier[i],
                resolution=resolutions[i],
                attn_resolutions=attn_resolutions,
                resample=(i != 0),
                num_resnet_blocks=num_resnet_blocks + 1,
                resample_with_conv=resample_with_conv,
                dropout_rate=dropout_rate,
                name=f'up_block{i}'
            )
            for i in reversed(range(self.num_resolutions))
        ]
      
        # Output projection
        self.conv_out = tf.keras.layers.Conv2D(
            image_channels,
            kernel_size=3,
            kernel_initializer=default_init(scale=0.),
            padding='same', 
            name='conv_out'
        )
        self.norm_out = tf.keras.layers.GroupNormalization(name='norm_out')


    def call(self, inputs, training=None):

        x, t = inputs

        # Get timestep embeddings
        time_emb = self.time_embeddings(t)

        # Initialize skip connections stack
        skips = []
        
        # Input conv
        h = self.conv_in(x)
        skips.append(h)

        # Down blocks
        for i in range(self.num_resolutions):
            h, skips = self.down_stages[i](h, time_emb, skips, training=training)

        # Middle
        h = self.bottleneck(h, time_emb, training=training)

        # Up blocks
        for i in range(self.num_resolutions):
            h, skips = self.up_stages[i](h, time_emb, skips, training=training)

        # End
        h = self.norm_out(h, training=training)
        h = tf.nn.silu(h)
        h = self.conv_out(h)

        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'image_channels': self.image_channels,
            'base_channels': self.base_channels,
            'channel_multiplier': self.channel_multiplier,
            'num_resnet_blocks': self.num_resnet_blocks,
            'resample_with_conv': self.resample_with_conv,
            'attn_resolutions': self.attn_resolutions,
            'dropout_rate': self.dropout_rate
        })
        return config
