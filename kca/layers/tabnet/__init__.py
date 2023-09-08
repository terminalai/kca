# layers/tabnet.py

import keras_core as keras
from keras_core import layers, ops, Model

from kca.activations.sparsemax import sparsemax
from kca.activations.glu import glu

from typing import Optional, Iterable
from kca.utils.types import Float, Int, TensorLike

__all__ = ["TabNet"]


class _TransformBlock(layers.Layer):
    def __init__(self, num_features: Int, norm_type: str = "group", momentum: Float = 0.9, virtual_batch_size=None,
                 groups: Int = 2, agg: bool = True, glu: bool = True, **kwargs):

        super().__init__(**kwargs)
        self.transform = layers.Dense(num_features, use_bias=False)

        if norm_type.lower() == "group":
            self.bn = layers.GroupNormalization(axis=-1, groups=groups)
        else:
            self.bn = layers.BatchNormalization(axis=-1, momentum=momentum, virtual_batch_size=virtual_batch_size)

        self.agg = agg
        self.glu = glu

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.transform(inputs)
        x = self.bn(x)
        if self.glu:
            x = glu(x)
        if self.agg:
            x = (x + inputs) * ops.sqrt(0.5)
        return x


class TabNet(Model):
    def __init__(self,
                 feature_columns: Optional[Iterable] = None, num_classes: Int = 2, feature_dim: Int = 64,
                 output_dim: Int = 64, num_features: Optional[Int] = None, num_decision_steps: Int = 5,
                 relaxation_factor: Float = 1.5, sparsity_coefficient: Float = 1e-5, norm_type: str = 'group',
                 batch_momentum: Float = 0.98, virtual_batch_size: Optional[Int] = None, num_groups: Int = 2,
                 epsilon: Float = 1e-5, random_state: Optional[Int] = None, **kwargs):
        super(TabNet, self).__init__(**kwargs)

        if feature_columns is None:
            feature_columns = []
        if num_features is None:
            num_features = len(feature_columns)
        if random_state is not None:
            keras.utils.set_random_seed(random_state)

        self.output_dim = output_dim
        self.num_features = num_features
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.epsilon = epsilon

        self.transform_f1 = _TransformBlock(
            2 * feature_dim, norm_type, batch_momentum, virtual_batch_size, num_groups, agg=False
        )
        self.transform_f2 = _TransformBlock(2 * feature_dim, norm_type, batch_momentum, virtual_batch_size, num_groups)

        self.transform_f3 = [
            _TransformBlock(2 * feature_dim, norm_type, batch_momentum, virtual_batch_size, num_groups)
            for _ in range(self.num_decision_steps)
        ]
        self.transform_f4 = [
            _TransformBlock(2 * feature_dim, norm_type, batch_momentum, virtual_batch_size, num_groups)
            for _ in range(self.num_decision_steps)
        ]

        self.transform_coef = [
            _TransformBlock(num_features, norm_type, batch_momentum, virtual_batch_size, num_groups, agg=False, glu=False)
            for _ in range(self.num_decision_steps-1)
        ]

        self._step_feature_selection_masks = None
        self._step_agg_feature_selection_mask = None

        if num_classes <= 2:
            self.clf = layers.Dense(num_classes, activation="sigmoid", use_bias=False)
        else:
            self.clf = layers.Dense(num_classes, activation="softmax", use_bias=False)

    def call(self, inputs: TensorLike) -> TensorLike:
        B = ops.shape(inputs)[0]

        self._step_feature_selection_masks = []
        self._step_agg_feature_selection_mask = None

        output_agg = ops.zeros([B, self.output_dim])
        masked_features = inputs
        mask_values = ops.zeros([B, self.num_features])
        agg_mask_values = ops.zeros([B, self.num_features])
        complementary_agg_mask_values = ops.ones([B, self.num_features])

        total_entropy = 0
        entropy_loss = 0

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            x = self.transform_f1(masked_features)
            x = self.transform_f2(x)
            x = self.transform_f3[ni](x)
            x = self.transform_f4[ni](x)

            if ni > 0 or self.num_decision_steps == 1:
                decision_out = ops.relu(x[..., :self.output_dim])
                # Decision aggregation
                output_agg += decision_out
                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = ops.sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg /= float(self.num_decision_steps - 1)

                agg_mask_values += mask_values * scale_agg

            if ni + 1 < self.num_decision_steps:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                features_for_coef = x[..., self.output_dim:]
                mask_values = self.transform_coef[ni](features_for_coef)
                mask_values *= complementary_agg_mask_values
                mask_values = sparsemax(mask_values, axis=-1)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_agg_mask_values *= (self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += ops.mean(
                    ops.sum(-mask_values * ops.log(mask_values + self.epsilon), axis=1)
                ) / (ops.cast(self.num_decision_steps - 1, "float32"))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = ops.multiply(mask_values, inputs)

                # Visualization of the feature selection mask at decision step ni
                # tf.summary.image(
                #     "Mask for step" + str(ni),
                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                #     max_outputs=1)
                mask_at_step_i = ops.expand_dims(ops.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.0

        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)

        # Visualization of the aggregated feature importances
        # tf.summary.image(
        #     "Aggregated mask",
        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        #     max_outputs=1)

        agg_mask = ops.expand_dims(ops.expand_dims(agg_mask_values, 0), 3)
        self._step_agg_feature_selection_mask = agg_mask

        out = self.clf(output_agg)

        return out
