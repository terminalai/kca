"""
Functions largely acquired from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions
"""

from keras_core import ops, activations

from typing import Callable
from kca.utils.types import TensorLike, Float

_EPSILON = 1e-7

__all__ = [
    "weighted_cross_entropy_with_logits",
    "binary_weighted_cross_entropy", "binary_balanced_cross_entropy", "multiclass_weighted_cross_entropy"
]


def convert_to_logits(y_pred: TensorLike) -> TensorLike:
    """
    Converting output of sigmoid to logits.

    :param y_pred: Predictions after sigmoid (<BATCH_SIZE>, shape=(None, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1)).
    :return: Logits (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
    """
    # To avoid unwanted behaviour of log operation
    y_pred = ops.clip(y_pred, _EPSILON, 1 - _EPSILON)

    return ops.log(y_pred / (1 - y_pred))


def weighted_cross_entropy_with_logits(labels: TensorLike, logits: TensorLike, pos_weight: Float) -> TensorLike:
    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = 1 + (pos_weight - 1) * labels
    return ops.add(
        (1 - labels) * logits,
        log_weight * (ops.log1p(ops.exp(-ops.abs(logits))) + activations.relu(-logits))
    )


def binary_weighted_cross_entropy(beta: Float,
                                  is_logits: bool = False) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Weighted cross entropy. All positive examples get weighted by the coefficient beta:

        WCE(p, p̂) = −[β*p*log(p̂) + (1−p)*log(1−p̂)]

    To decrease the number of false negatives, set β>1. To decrease the number of false positives, set β<1.

    If last layer of network is a sigmoid function, y_pred needs to be reversed into logits before computing the
    weighted cross entropy. To do this, we're using the same method as implemented in Keras binary_crossentropy:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525

    Used as loss function for binary image segmentation with one-hot encoded masks.

    :param beta: Weight coefficient (float)
    :param is_logits: If y_pred are logits (bool, default=False)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Computes the weighted cross entropy.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        if not is_logits:
            y_pred = convert_to_logits(y_pred)

        wce_loss = weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(wce_loss))
        wce_loss = ops.mean(wce_loss, axis=axis_to_reduce)

        return wce_loss

    return loss


def binary_balanced_cross_entropy(beta: Float,
                                  is_logits: bool = False) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Balanced cross entropy. Similar to weighted cross entropy (see weighted_cross_entropy),
    but both positive and negative examples get weighted:

        BCE(p, p̂) = −[β*p*log(p̂) + (1-β)*(1−p)*log(1−p̂)]

    If last layer of network is a sigmoid function, y_pred needs to be reversed into logits before computing the
    balanced cross entropy. To do this, we're using the same method as implemented in Keras binary_crossentropy:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525

    Used as loss function for binary image segmentation with one-hot encoded masks.

    :param beta: Weight coefficient (float)
    :param is_logits: If y_pred are logits (bool, default=False)
    :return: Balanced cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if beta == 1.:  # To avoid division by zero
        beta -= _EPSILON

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Computes the balanced cross entropy in the following way:

            BCE(p, p̂) = −[(β/(1-β))*p*log(p̂) + (1−p)*log(1−p̂)]*(1-β) = −[β*p*log(p̂) + (1-β)*(1−p)*log(1−p̂)]

        :param y_true: Ground truth (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Balanced cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        if not is_logits:
            y_pred = convert_to_logits(y_pred)

        pos_weight = beta / (1 - beta)
        bce_loss = weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
        bce_loss = bce_loss * (1 - beta)

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(bce_loss))
        bce_loss = ops.mean(bce_loss, axis=axis_to_reduce)

        return bce_loss

    return loss


def multiclass_weighted_cross_entropy(class_weights: TensorLike,
                                      is_logits: bool = False) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Multi-class weighted cross entropy.

        WCE(p, p̂) = −Σp*log(p̂)*class_weights

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    :param class_weights: Weight coefficients (list of floats)
    :param is_logits: If y_pred are logits (bool)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Computes the weighted cross entropy.

        :param y_true: Ground truth (tf.Tensor, shape=(None, None, None, None))
        :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        assert len(class_weights) == y_pred.shape[-1], (f"Number of class_weights ({len(class_weights)}) needs to be "
                                                        f"the same as number of classes ({y_pred.shape[-1]})")

        if is_logits:
            y_pred = ops.softmax(y_pred, axis=-1)

        y_pred = ops.clip(y_pred, _EPSILON, 1-_EPSILON)  # To avoid unwanted behaviour in K.log(y_pred)

        # p * log(p̂) * class_weights
        wce_loss = y_true * ops.log(y_pred) * class_weights

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(wce_loss))
        wce_loss = ops.mean(wce_loss, axis=axis_to_reduce)

        return -wce_loss

    return loss
