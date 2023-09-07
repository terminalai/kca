from keras_core import layers, Sequential, activations
from kca.layers.residual import Residual

from kca.utils.types import Int, Float, TensorLike

__all__ = ["TransformerBlock"]


class MultiHeadAttention(layers.MultiHeadAttention):
    def __init__(self, num_heads: Int, embed_dim: Int, dropout: Float = 0.0):
        super().__init__(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)

    def build(self, shape):
        super().build(shape, shape)

    def call(self, inputs):
        return super().call(inputs, inputs)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: Int, num_heads: Int, ff_dim: Int,
                 att_dropout: Float = 0.1, ff_dropout: Float = 0.1,
                 eps: Float = 1e-6, **kwargs):
        super().__init__(**kwargs)

        self.att = Residual(MultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim, dropout=att_dropout
        ))

        self.attnorm = layers.LayerNormalization(epsilon=eps)

        self.ffn = Residual(Sequential([
            layers.Dense(ff_dim, activation=activations.gelu),
            layers.Dropout(ff_dropout),
            layers.Dense(embed_dim),
        ]))

        self.ffnnorm = layers.LayerNormalization(epsilon=eps)

    def call(self, inputs: TensorLike) -> TensorLike:
        # Multi-Head Attention
        x = self.attnorm(self.att(inputs))

        # FFN
        y = self.ffnnorm(self.ffn(x))

        return y
