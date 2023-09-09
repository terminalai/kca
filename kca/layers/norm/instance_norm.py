from keras_core import layers, ops, initializers

__all__ = ["InstanceNormalization"]


class InstanceNormalization(layers.Layer):
    """
    Instance Normalization Layer
    (https://arxiv.org/abs/1607.08022).
    """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.scale = None
        self.offset = None
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=initializers.random_normal(1.0, 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name="offset",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True)

    def call(self, x):
        mean = ops.mean(x, axis=[1, 2], keepdims=True)
        variance = ops.var(x, axis=[1, 2], keepdims=True)
        inv = ops.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        if self.scale is not None and self.offset is not None:
            return self.scale * normalized + self.offset
        return normalized
