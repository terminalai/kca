# layers/fnet/__init__.py

import keras_core as keras
from keras_core import layers, ops, activations
from kca.layers.residual import Residual
from typing import Tuple
from kca.utils.types import Float, TensorLike

__all__ = ["FNetLayer"]


def rfft2d(inputs: TensorLike) -> TensorLike:
    return ops.fft2((inputs, ops.zeros_like(inputs)))[0]


class FNetLayer(layers.Layer):
    def __init__(self, dropout_rate: Float, eps: Float, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.fft = Residual(rfft2d)
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.ffn = lambda x: x

    def build(self, input_shape: Tuple):
        embedding_dim = input_shape[-1]
        self.ffn = Residual(keras.Sequential([
            layers.Dense(units=embedding_dim, activation=activations.gelu),
            layers.Dropout(rate=self.dropout_rate),
            layers.Dense(units=embedding_dim),
        ]))

    def call(self, inputs: TensorLike) -> TensorLike:
        # Apply fourier transformations.
        x = self.norm(self.fft(inputs))
        # Apply Feedforward network.
        y = self.norm(self.ffn(x))

        return y
