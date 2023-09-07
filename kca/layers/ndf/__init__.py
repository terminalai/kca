# layers/ndf/__init__.py

import keras_core as keras
from keras_core import layers, ops, activations, initializers
import numpy as np
from typing import Tuple
from kca.utils.types import Int, Float, TensorLike

__all__ = ["NeuralDecisionTree", "NeuralDecisionForest"]


class NeuralDecisionTree(layers.Layer):
    def __init__(self, depth: Int, used_features_rate: Float, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.used_features_rate = used_features_rate

        self.used_features_mask = None

        # Initialize the weights of the classes in leaves.
        self.pi = self.add_weight(
            (self.num_leaves,), initializers.random_normal(), dtype="float32", trainable=True
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(self.num_leaves, activation=activations.sigmoid)

    def build(self, input_shape: Tuple):
        batch_size, num_features = input_shape

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * self.used_features_rate)
        self.used_features_mask = ops.transpose(ops.one_hot(
            np.random.choice(np.arange(num_features), num_used_features, replace=False),
            num_classes=num_features
        ))

    def call(self, features: TensorLike) -> TensorLike:
        batch_size, num_features = ops.shape(features)

        # Apply the feature mask to the input features. [batch_size, num_used_features]
        features = ops.matmul(
            features, self.used_features_mask
        ) if self.used_features_mask is not None else features

        # Compute the routing probabilities.
        decisions = ops.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]

        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = ops.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2

        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = ops.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = ops.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[:, begin_idx:end_idx, :]  # [batch_size, 2 ** level, 2]
            mu *= level_decisions  # [batch_size, 2**level, 2]

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = ops.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = activations.softmax(self.pi)  # [num_leaves, ]
        outputs = ops.matmul(mu, probabilities)  # [batch_size, ]
        return outputs


class NeuralDecisionForest(keras.Layer):
    def __init__(self, num_trees: Int, depth: Int, used_features_rate: Float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.num_trees = num_trees
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        self.ensemble = [NeuralDecisionTree(depth, used_features_rate) for _ in range(num_trees)]

    def call(self, inputs: TensorLike) -> TensorLike:
        batch_size = ops.shape(inputs)[0]

        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        outputs = ops.zeros((batch_size, ))

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)

        # Divide the outputs by the ensemble size to get the average.
        outputs /= self.num_trees

        return outputs
