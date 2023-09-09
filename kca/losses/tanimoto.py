"""
Functions largely acquired from: https://github.com/maxvfischer/keras-image-segmentation-loss-functions
"""

from keras_core import ops

from typing import Callable
from kca.utils.types import TensorLike

__all__ = ["multiclass_weighted_tanimoto_loss"]


def multiclass_weighted_tanimoto_loss(class_weights: TensorLike) -> Callable[[TensorLike, TensorLike], TensorLike]:
    """
    Weighted Tanimoto loss.

    Defined in the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data",
    under 3.2.4. Generalization to multiclass imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Tanimoto loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """
        Compute weighted Tanimoto loss.

        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Tanimoto loss (tf.Tensor, shape=(None, ))
        """
        axis_to_reduce = range(1, ops.ndim(y_pred))  # All axis but first (batch)
        numerator = ops.multiply(y_true * y_pred, class_weights)
        numerator = ops.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2 - y_true * y_pred) * class_weights
        denominator = ops.sum(denominator, axis=axis_to_reduce)
        return 1 - numerator / denominator

    return loss
