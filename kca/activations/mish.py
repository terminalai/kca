from keras_core import activations
from kca.utils.types import TensorLike


def mish(x: TensorLike) -> TensorLike:
    r"""Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    Computes mish activation:

    $$
    \mathrm{mish}(x) = x \cdot \tanh(\mathrm{softplus}(x)).
    $$

    See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).

    Usage:

    >>> x = tf.constant([1.0, 0.0, 1.0])
    >>> kca.activations.mish(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.865098..., 0.       , 0.865098...], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return x * activations.tanh(activations.softplus(x))
