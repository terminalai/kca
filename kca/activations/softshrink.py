from keras_core import ops

from kca.utils.types import TensorLike, Number


def softshrink(x: TensorLike, lower: Number = -0.5, upper: Number = 0.5) -> TensorLike:
    r"""Soft shrink function.

    Computes soft shrink function:

    $$
    \mathrm{softshrink}(x) =
    \begin{cases}
        x - \mathrm{lower} & \text{if } x < \mathrm{lower} \\
        x - \mathrm{upper} & \text{if } x > \mathrm{upper} \\
        0                  & \text{otherwise}
    \end{cases}.
    $$

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> kca.activations.softshrink(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.5,  0. ,  0.5], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
        lower: `float`, lower bound for setting values to zeros.
        upper: `float`, upper bound for setting values to zeros.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    if lower > upper:
        raise ValueError(
            "The value of lower is {} and should not be higher than the value variable upper, which is {} .".format(
                lower, upper)
        )
    values_below_lower = ops.where(x < lower, x - lower, 0)
    values_above_upper = ops.where(upper < x, x - upper, 0)
    return values_below_lower + values_above_upper
