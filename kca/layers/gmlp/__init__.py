"""
gMLP (Gated Multi-Layer Perceptron) Implementation

Based on the Implementation Proposed by "Pay Attention to MLPs" by Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le
(NeurIPS 2021).

P.S. aMLP has also been implemented here since it's from the same paper.
"""

import keras_core as keras
from keras_core import ops, layers, activations
from typing import Tuple
from kca.utils.types import Int, Float, TensorLike

__all__ = ["SpatialGatingUnit", "gMLPBlock", "TinyAttention"]


class TinyAttention(layers.Layer):
    def __init__(self, d_out: Int, d_attn: Int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_attn = d_attn
        self.initial_proj = layers.Dense(3 * d_attn)
        self.final_proj = layers.Dense(d_out)

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.initial_proj(inputs)
        q, k, v = ops.split(x, 3, axis=-1)
        w = ops.einsum("bnd,bmd->bnm", q, k)
        a = ops.softmax(w * ops.rsqrt(float(self.d_attn)))
        x = ops.einsum("bnm,bmd->bnd", a, v)
        y = self.final_proj(x)
        return y


class SpatialGatingUnit(layers.Layer):
    def __init__(self, eps: Float = 1e-6, use_attention: bool = False, d_attn: Int = 64, **kwargs):
        super().__init__(**kwargs)
        self.use_attention = use_attention
        self.d_attn = d_attn
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.proj = None
        self.att = None

    def build(self, input_shape: Tuple[Int]):
        self.proj = layers.Dense(input_shape[1], bias_initializer="Ones")
        if self.use_attention:
            self.att = TinyAttention(input_shape[-1] // 2, d_attn=self.d_attn)

    def call(self, inputs: TensorLike) -> TensorLike:
        u, v = ops.split(inputs, 2, axis=-1)

        v = self.norm(v)
        v = ops.transpose(v, (0, 2, 1))
        v = self.proj(v)
        v = ops.transpose(v, (0, 2, 1))

        if self.att is not None:
            v += self.att(inputs)

        return u * v


class gMLPBlock(layers.Layer):
    def __init__(self, ffn_dim: Int, dropout: Float = 0.1, eps: Float = 1e-6,
                 use_attention: bool = False, d_attn: Int = 64, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=eps)

        self.proj_in = keras.Sequential([
            layers.Dense(ffn_dim, activation=activations.gelu),
            layers.Dropout(dropout)
        ])

        self.sgu = SpatialGatingUnit(eps=eps, use_attention=use_attention, d_attn=d_attn)

        self.proj_out = None

    def build(self, input_shape):
        self.proj_out = layers.Dense(input_shape[-1])

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.norm(inputs)
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return inputs + x
