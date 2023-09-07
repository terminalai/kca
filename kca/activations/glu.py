# activations/glu.py

from keras_core import ops, activations

from kca.utils.types import TensorLike
from typing import Callable


def _glu_builder(activation=activations.sigmoid) -> Callable:
    def glu_func(x: TensorLike) -> TensorLike:
        x, gates = ops.split(x, 2, axis=-1)
        return x * activation(gates)

    return glu_func


glu = _glu_builder()
reglu = _glu_builder(activations.relu)
geglu = _glu_builder(activations.gelu)
swiglu = _glu_builder(activations.swish)
seglu = _glu_builder(activations.selu)
