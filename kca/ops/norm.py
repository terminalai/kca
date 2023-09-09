from keras_core import ops
from typing import Union, Iterable, Optional
from kca.utils.types import TensorLike, Float, Int


def l2_normalize(
    x: TensorLike, axis: Optional[TensorLike] = None, epsilon: TensorLike = 1e-12
):
    return ops.divide(x, ops.sqrt(ops.maximum(ops.sum(x ** 2, axis=axis), epsilon)))
