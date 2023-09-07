from keras_core import layers, ops, random
from kca.utils.types import Float

__all__ = ['StochasticDepth']


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob: Float, **kwargs):
        super().__init__(**kwargs)
        self.keep_prob = 1 - drop_prob

    def call(self, x, training=True):
        if training:
            shape = ops.shape(x)
            shape = (shape[0],) + (1,) * (len(shape) - 1)

            random_tensor = self.keep_prob + random.uniform(shape)
            random_tensor = ops.floor(random_tensor)
            return (x / self.keep_prob) * random_tensor
        return x
