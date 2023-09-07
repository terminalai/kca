# utils/types.py

import numpy as np
import tensorflow as tf
import keras_core as keras
from typing import Union, List

Int = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64
]

Float = Union[
    float,
    np.float16,
    np.float32,
    np.float64
]

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    keras.KerasTensor,
    # torch.Tensor
]

LossType = Union[
    keras.losses.Loss,
    str
]

OptimizerType = Union[
    keras.optimizers.Optimizer,
    str
]