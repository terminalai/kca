from keras_core import ops, losses
from kca.utils.types import TensorLike


def npairs_loss(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from the same labels and each pairs in the
    minibatch have different labels. The loss takes each row of the pair-wise similarity matrix, `y_pred`, as logits
    and the remapped multi-class labels, `y_true`, as labels.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b` with shape `[batch_size, hidden_size]`
    can be computed as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = ops.matmul(a, b.transpose((1, 0)))
    >>> y_pred
    <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
    array([[23., 15., 17.],
       [51., 33., 35.],
       [79., 51., 53.]], dtype=float16)>

    <... Note: constants a & b have been used purely for example purposes and have no significant value ...>

    See: https://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of multi-class labels.
      y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of similarity matrix between embedding matrices.

    Returns:
      npairs_loss: float scalar.
    """
    # Expand to [batch_size, 1]
    y_true = ops.expand_dims(y_true, -1)
    y_true = ops.cast(ops.equal(y_true, ops.transpose(y_true)), y_pred.dtype)
    y_true /= ops.sum(y_true, 1, keepdims=True)

    loss = losses.categorical_crossentropy(y_true, y_pred)

    return ops.mean(loss)
