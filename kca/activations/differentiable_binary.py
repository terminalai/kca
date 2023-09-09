from keras_core import ops

from kca.utils.types import Int, Float, TensorLike

__all__ = ["differentiable_binary"]


def differentiable_binary(x: TensorLike, gamma: Int = 20, T: Float = 0.5) -> TensorLike:
    """Approximately binarize the input with a differentiable function.

    Contour Loss for Instance Segmentation via k-step Distance
    Transformation Image, Guo et al., 2021

    Parameters
    ----------
    x : tf.Tensor
        The input to binarize.
    gamma : int
        The slope.
    T : float
        Threshold value.

    Returns
    -------
    tf.Tensor

    Examples
    --------
    FIXME: Add docs.
    """
    return 1. / (1 + ops.exp(-gamma * (x - T)))
