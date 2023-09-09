from keras_core import activations
from kca.utils.types import TensorLike


def lisht(x: TensorLike) -> TensorLike:
    r"""LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function.

    Computes linearly scaled hyperbolic tangent (LiSHT):

    $$
    \mathrm{lisht}(x) = x * \tanh(x).
    $$

    See [LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/abs/1901.05894).

    Usage:

    >>> x = tf.constant([1.0, 0.0, 1.0])
    >>> kca.activations.lisht(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.7615942, 0.       , 0.7615942], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return x * activations.tanh(x)
