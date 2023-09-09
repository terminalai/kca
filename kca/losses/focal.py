"""
Functions largely acquired from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions
"""

from keras_core import ops

from typing import Callable
from kca.utils.types import TensorLike, Float

__all__ = ["binary_focal_loss", "multiclass_focal_loss"]


def binary_focal_loss(beta: Float, gamma: Float = 2.) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Focal loss is derived from balanced cross entropy, where focal loss adds an extra focus on hard examples in the
    dataset:

        FL(p, p̂) = −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

    When γ = 0, we obtain balanced cross entropy.

    Paper: https://arxiv.org/pdf/1708.02002.pdf

    Used as loss function for binary image segmentation with one-hot encoded masks.

    :param beta: Weight coefficient (float)
    :param gamma: Focusing parameter, γ ≥ 0 (float, default=2.)
    :return: Focal loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Computes the focal loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Focal loss (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        f_loss = beta * (1 - y_pred) ** gamma * y_true * ops.log(y_pred)  # β*(1-p̂)ᵞ*p*log(p̂)
        f_loss += (1 - beta) * y_pred ** gamma * (1 - y_true) * ops.log(1 - y_pred)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
        f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(f_loss))
        f_loss = ops.mean(f_loss, axis=axis_to_reduce)

        return f_loss

    return loss


def multiclass_focal_loss(class_weights: TensorLike,
                          gamma: TensorLike) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Focal loss.

        FL(p, p̂) = -∑class_weights*(1-p̂)ᵞ*p*log(p̂)

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :param gamma: Focusing parameters, γ_i ≥ 0 (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Focal loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute focal loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Focal loss (tf.Tensor, shape=(None,))
        """
        f_loss = -ops.multiply(class_weights, ops.power((1-y_pred), gamma) * y_true * ops.log(y_pred))

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(f_loss))
        f_loss = ops.mean(f_loss, axis=axis_to_reduce)

        return f_loss

    return loss
