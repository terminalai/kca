from keras_core import ops, random
from typing import Optional
from kca.utils.types import TensorLike, Number, Generator


def rrelu(x: TensorLike, lower: Number = 0.125, upper: Number = 0.3333333333333333, training: bool = False,
          seed: Optional[int] = None, rng: Optional[Generator] = None) -> TensorLike:
    r"""Randomized leaky rectified liner unit function.

    Computes rrelu function:

    $$
    \mathrm{rrelu}(x) =
    \begin{cases}
        x & \text{if } x > 0 \\
        a x
    \end{cases},
    $$

    where

    $$
    a \sim \mathcal{U}(\mathrm{lower}, \mathrm{upper})
    $$

    when `training` is `True`; or

    $$
    a = \frac{\mathrm{lower} + \mathrm{upper}}{2}
    $$

    when `training` is `False`.

    See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> kca.activations.rrelu(x, training=False)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.22916667,  0.        ,  1.        ], dtype=float32)>
    >>> kca.activations.rrelu(x, training=True, seed=2020)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.22631127,  0.        ,  1.        ], dtype=float32)>
    >>> generator = tf.random.Generator.from_seed(2021)
    >>> kca.activations.rrelu(x, training=True, rng=generator)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.16031083,  0.        ,  1.        ], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
        lower: `float`, lower bound for random alpha.
        upper: `float`, upper bound for random alpha.
        training: `bool`, indicating whether the `call`
        is meant for training or inference.
        seed: `int`, this sets the operation-level seed.
        rng: A `tf.random.Generator`.
    Returns:
        result: A `Tensor`. Has the same type as `x`.
    """
    lower = ops.cast(lower, x.dtype)
    upper = ops.cast(upper, x.dtype)

    def random_a():
        if rng is not None and seed is not None:
            raise ValueError("Either seed or rng should be specified. Not both at the same time.")

        if rng is not None:
            return rng.uniform(ops.shape(x), minval=lower, maxval=upper, dtype=x.dtype)

        return random.uniform(
            ops.shape(x), minval=lower, maxval=upper, dtype=x.dtype, seed=seed
        )

    a = random_a() if training else ((lower + upper) / 2)

    return ops.where(x >= 0, x, a * x)
