# layers/convmixer/__init__.py

import keras_core as keras
from keras_core import layers, activations
from kca.layers.residual import Residual
from kca.utils.types import Int, TensorLike

__all__ = [
    "DepthwiseConvMixerLayer", "PointwiseConvMixerLayer",
    "ConvMixerLayer", "ConvMixer"
]


class DepthwiseConvMixerLayer(layers.Layer):
    def __init__(self, dim: Int, kernel_size: Int = 9, **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.Conv2D(dim, kernel_size, groups=dim, padding="same")
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.depthwise(inputs)
        x = activations.gelu(x)
        x = self.batch_norm(x)
        return x


class PointwiseConvMixerLayer(layers.Layer):
    def __init__(self, dim: Int, **kwargs):
        super().__init__(**kwargs)
        self.pointwise = layers.Conv2D(dim, kernel_size=1)
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.pointwise(inputs)
        x = activations.gelu(x)
        x = self.batch_norm(x)
        return x


class ConvMixerLayer(layers.Layer):
    def __init__(self, dim: Int, kernel_size: Int = 9, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depthwise = Residual(DepthwiseConvMixerLayer(dim, self.kernel_size))
        self.pointwise = PointwiseConvMixerLayer(dim)

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return x


class ConvMixer(layers.Layer):
    def __init__(self, dim: Int, depth: Int, kernel_size: Int = 9, patch_size: Int = 7, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size)
        self.batch_norm = layers.BatchNormalization()

        self.mixer = keras.Sequential([
            ConvMixerLayer(dim, kernel_size) for i in range(depth)
        ])

        self.gap = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()

    def call(self, inputs: TensorLike) -> TensorLike:
        # Patch Embedding
        x = activations.gelu(self.patch_embedding(inputs))
        x = self.batch_norm(x)

        # ConvMixer
        x = self.mixer(x)

        # GAP
        x = self.flatten(self.gap())

        return x
