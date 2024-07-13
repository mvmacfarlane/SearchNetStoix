from typing import Sequence, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.environments.routing.connector.types import Observation


class BlockV1(nn.Module):
    channels: int
    stride: Union[int, Sequence[int]]
    use_projection: bool
    bottleneck: bool

    def setup(self) -> None:

        if self.use_projection:
            self.proj_conv = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                padding="SAME",
                name="shortcut_conv",
            )

        channel_div = 4 if self.bottleneck else 1
        conv_0 = nn.Conv(
            features=self.channels // channel_div,
            kernel_size=(1, 1) if self.bottleneck else (3, 3),
            strides=(1, 1) if self.bottleneck else self.stride,
            use_bias=False,
            padding="SAME",
            name="conv_0",
        )

        conv_1 = nn.Conv(
            features=self.channels // channel_div,
            kernel_size=(3, 3),
            strides=self.stride if self.bottleneck else (1, 1),
            use_bias=False,
            padding="SAME",
            name="conv_1",
        )

        layers = ((conv_0,), (conv_1,))

        if self.bottleneck:
            conv_2 = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                padding="SAME",
                name="conv_2",
            )

        layers = layers + ((conv_2,),)

        self.layers = layers

    def __call__(self, inputs: chex.Array) -> chex.Array:
        out = shortcut = inputs

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)

        for i, (conv_i,) in enumerate(self.layers):
            out = conv_i(out)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)


class ResNetSoko(nn.Module):
    output_channels: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    layer_sizes: Sequence[int]
    max_timesteps: int


    def setup(self) -> None:
        self.blocks = [
            (
                nn.Sequential(
                    [
                        nn.Conv(
                            features=i,
                            kernel_size=(k, k),
                            strides=(s, s),
                            padding="SAME",
                        ),
                        jax.nn.relu,
                    ]
                ),
                BlockV1(
                    channels=i,
                    stride=[s, s],
                    use_projection=True,
                    bottleneck=True,
                    name=f"block_v1_1{j}",
                ),
                BlockV1(
                    channels=i,
                    stride=[s, s],
                    use_projection=True,
                    bottleneck=True,
                    name=f"block_v1_2{j}",
                ),
            )
            for j, (i, k, s) in enumerate(
                zip(self.output_channels, self.kernel_sizes, self.strides)
            )
        ]

        self.critic_hidden_layers = nn.Sequential(
            [layer for units in self.layer_sizes for layer in [nn.Dense(units), jax.nn.relu]]
        )

    def preprocess_input(
        self,
        input_array: chex.Array,
    ) -> chex.Array:

        one_hot_array_fixed = jnp.equal(input_array[..., 0:1], jnp.array([3, 4])).astype(
            jnp.float32
        )

        one_hot_array_variable = jnp.equal(input_array[..., 1:2], jnp.array([1, 2])).astype(
            jnp.float32
        )

        total = jnp.concatenate((one_hot_array_fixed, one_hot_array_variable), axis=-1)

        return total

    def __call__(
        self,
        x_batch: Observation,
    ) -> chex.Array:
        
        x_processed = self.preprocess_input(x_batch)

        for (conv_i, resnet_1, resnet_2) in self.blocks:
            x_processed = conv_i(x_processed)
            x_processed = resnet_1(x_processed)
            x_processed = resnet_2(x_processed)

        trailing_dims = x_processed.shape[-3:]
        x_processed = jnp.reshape(x_processed, (*x_processed.shape[:-3], np.prod(trailing_dims)))

        x_value = self.critic_hidden_layers(x_processed)

        return x_value


class SokoObservationActionInput(nn.Module):
    """Observation and Action Input."""

    @nn.compact
    def __call__(self, observation: Observation, action: chex.Array) -> chex.Array:
        observation = observation.agent_view
        return (observation, action)
