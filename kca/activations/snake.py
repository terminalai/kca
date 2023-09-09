from keras_core import ops, activations
from kca.utils.types import TensorLike, Number


def snake(x: TensorLike, frequency: Number = 1) -> TensorLike:
    r"""Snake activation to learn periodic functions.

    Computes snake activation:

    $$
    \mathrm{snake}(x) = \mathrm{x} + \frac{1 - \cos(2 \cdot \mathrm{frequency} \cdot x)}{2 \cdot \mathrm{frequency}}.
    $$

    See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.snake(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.29192656,  0.        ,  1.7080734 ], dtype=float32)>

    Args:
        x: A `Tensor`.
        frequency: A scalar, frequency of the periodic part.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    frequency = ops.cast(frequency, x.dtype)

    return x + (1 - ops.cos(2 * frequency * x)) / (2 * frequency)
