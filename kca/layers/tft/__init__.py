from keras_core import layers, ops, activations
from kca.activations.glu import glu
from typing import Tuple, List
from kca.utils.types import Int, Float, TensorLike


class GatedResidualNetwork(layers.Layer):
    def __init__(self, units: Int, dropout_rate: Float, eps: Float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.elu_dense = layers.Dense(units, activation=activations.elu)
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu_input = layers.Dense(2*units)
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.project = layers.Dense(units)

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        x = self.glu_input(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + glu(x)
        x = self.norm(x)
        return x


class VariableSelection(layers.Layer):
    def __init__(self, units: Int, dropout_rate: Float = 0.15):
        super().__init__()
        self.units = units
        self.dropout_rate = dropout_rate

        self.grns = [GatedResidualNetwork(units, dropout_rate)]

        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units, activation=activations.softmax)

    def build(self, input_shape: Tuple):
        num_features = input_shape[0]
        self.softmax = layers.Dense(num_features, activation=activations.softmax)

        # Create a GRN for each feature independently
        self.grns = [
            GatedResidualNetwork(self.units, self.dropout_rate) for _ in range(num_features)
        ]

    def call(self, inputs: List[TensorLike]) -> TensorLike:
        # shape of inputs is [num_features, (B, units)]

        v = ops.concatenate(inputs, axis=-1)  # (B, num_features*units)
        v = self.grn_concat(v)  # (B, units)
        v = ops.expand_dims(self.softmax(v), axis=-1)  # (B, num_features, 1)

        x = [self.grns[idx](inp) for idx, inp in enumerate(inputs)]  # [num_features, (B, units)]

        x = ops.stack(x, axis=1)  # (B, num_features, units)

        # (B, 1, units) -> (B, units)
        outputs = ops.squeeze(ops.matmul(ops.transpose(v, (0, 2, 1)), x), axis=1)

        return outputs
