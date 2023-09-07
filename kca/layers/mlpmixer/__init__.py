import keras_core as keras
from keras_core import ops, layers, activations

from typing import Tuple
from kca.utils.types import Float, TensorLike

__all__ = ["MLPMixerBlock"]


class MLPMixerBlock(layers.Layer):
    def __init__(self, dropout_rate: Float, eps: Float, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.spatial_mixer_mlp = self.channel_mixer_mlp = lambda x: ops.zeros_like(x)

    def build(self, input_shape: Tuple):
        spatial_dim = input_shape[-2]
        embedding_dim = input_shape[-1]

        self.spatial_mixer_mlp = keras.Sequential([
            layers.Dense(units=spatial_dim, activation=activations.gelu),
            layers.Dense(units=spatial_dim, activation=activations.gelu)
        ])

        self.channel_mixer_mlp = keras.Sequential([
            layers.Dense(units=spatial_dim, activation=activations.gelu),
            layers.Dense(units=embedding_dim),
            layers.Dropout(rate=self.dropout_rate),
        ])

    def spatial_mixer(self, inputs: TensorLike) -> TensorLike:
        x = self.norm(inputs)
        x = ops.transpose(x, (0, 2, 1))
        x = self.spatial_mixer_mlp(x)
        x = ops.transpose(x, (0, 2, 1))
        return inputs + x

    def channel_mixer(self, inputs: TensorLike) -> TensorLike:
        x = self.norm(inputs)
        x = self.channel_mixer_mlp(x)
        return inputs + x

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.spatial_mixer(inputs)
        y = self.channel_mixer(x)
        return y
