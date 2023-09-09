"""
Functions largely acquired from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions
"""

from keras_core import ops

from kca.losses.tversky import binary_tversky_coef

from typing import Callable
from kca.utils.types import TensorLike, Float

__all__ = [
    "binary_dice_coef_loss", "binary_weighted_dice_crossentropy_loss",
    "multiclass_weighted_dice_loss", "multiclass_weighted_squared_dice_loss"
]


def binary_dice_coef_loss(smooth: Float = 1.) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Dice coefficient loss:

        DL(p, p̂) = 1 - (2*p*p̂+smooth)/(p+p̂+smooth)

    Used as loss function for binary image segmentation with one-hot encoded masks.

    :param smooth: Smoothing factor (float, default=1.)
    :return: Dice coefficient loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute the Dice loss (Tversky loss with β=0.5).

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Dice coefficient loss for each observation in batch (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        return 1 - binary_tversky_coef(y_true=y_true, y_pred=y_pred, beta=0.5, smooth=smooth)

    return loss


def binary_weighted_dice_crossentropy_loss(smooth: Float = 1.,
                                           beta: Float = 0.5) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Weighted Dice cross entropy combination loss is a weighted combination between Dice's coefficient loss and
    binary cross entropy:

        DL(p, p̂) = 1 - (2*p*p̂+smooth)/(p+p̂+smooth)
        CE(p, p̂) = - [p*log(p̂ + 1e-7) + (1-p)*log(1-p̂ + 1e-7)]
        WDCE(p, p̂) = weight*DL + (1-weight)*CE
                   = weight*[1 - (2*p*p̂+smooth)/(p+p̂+smooth)] - (1-weight)*[p*log(p̂ + 1e-7) + (1-p)*log(1-p̂ + 1e-7)]

    Used as loss function for binary image segmentation with one-hot encoded masks.

    :param smooth: Smoothing factor (float, default=1.)
    :param beta: Loss weight coefficient (float, default=0.5)
    :return: Dice cross entropy combination loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    assert 0. <= beta <= 1., "Loss weight has to be between 0.0 and 1.0"

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute the Dice cross entropy combination loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Dice cross entropy combination loss (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        cross_entropy = ops.binary_crossentropy(target=y_true, output=y_true)

        # Average over each data point/image in batch
        axis_to_reduce = range(1, ops.ndim(cross_entropy))
        cross_entropy = ops.mean(x=cross_entropy, axis=axis_to_reduce)

        dice_coefficient = binary_tversky_coef(y_true=y_true, y_pred=y_pred, beta=0.5, smooth=smooth)

        return beta*(1. - dice_coefficient) + (1. - beta)*cross_entropy

    return loss


def multiclass_weighted_dice_loss(class_weights: TensorLike) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Weighted Dice loss.

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute weighted Dice loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, ops.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = ops.multiply(y_true * y_pred, class_weights)  # Broadcasting
        numerator = 2. * ops.sum(numerator, axis=axis_to_reduce)

        denominator = ops.multiply(y_true + y_pred, class_weights)  # Broadcasting
        denominator = ops.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return loss


def multiclass_weighted_squared_dice_loss(class_weights: TensorLike) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Weighted squared Dice loss.

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted squared Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute weighted squared Dice loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted squared Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, ops.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = ops.multiply(y_true * y_pred, class_weights)  # Broadcasting
        numerator = 2. * ops.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2) * class_weights  # Broadcasting
        denominator = ops.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return loss
