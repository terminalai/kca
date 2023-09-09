from keras_core import activations
from kca.utils.types import TensorLike


def tanhshrink(x: TensorLike) -> TensorLike:
    r"""Tanh shrink function.

    Applies the element-wise function:

    $$
    \mathrm{tanhshrink}(x) = x - \tanh(x).
    $$

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> kca.activations.tanhshrink(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.23840582,  0.        ,  0.23840582], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return x - activations.tanh(x)
