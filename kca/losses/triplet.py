from keras_core import ops
from typing import Union, Callable
from kca.utils.types import TensorLike
from .metric_learning import pairwise_distance, angular_distance


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = ops.min(data, dim, keepdims=True)
    masked_maximums = ops.max(ops.multiply(data - axis_minimums, mask), dim, keepdims=True) + axis_minimums
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = ops.max(data, dim, keepdims=True)
    masked_minimums = ops.min(ops.multiply(data - axis_maximums, mask), dim, keepdims=True) + axis_maximums
    return masked_minimums


def triplet_semihard_loss(y_true: TensorLike, y_pred: TensorLike, margin: TensorLike = 1.0,
                          distance_metric: Union[str, Callable] = "L2") -> TensorLike:
    """Computes the triplet loss with semi-hard negative mining.

    Usage:

    >>> y_true = ops.convert_to_tensor([0, 0])
    >>> y_pred = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> kca.losses.triplet_semihard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=2.4142137>

    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: ops.matmul(x, x.transpose((1, 0)))
    >>> kca.losses.triplet_semihard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.

        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.

    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """
    labels = ops.convert_to_tensor(y_true)
    precise_embeddings = ops.convert_to_tensor(y_pred)

    # Reshape label tensor to [batch_size, 1].
    lshape = ops.shape(labels)
    labels = ops.reshape(labels, (lshape[0], 1))

    # Build pairwise squared distance matrix

    if distance_metric == "L2":
        pdist_matrix = pairwise_distance(precise_embeddings, squared=False)

    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance(precise_embeddings, squared=True)

    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = ops.equal(labels, ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = ops.logical_not(adjacency)

    batch_size = ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = ops.tile(pdist_matrix, [batch_size, 1])
    mask = ops.logical_and(
        ops.tile(adjacency_not, [batch_size, 1]),
        ops.greater(
            pdist_matrix_tile, ops.reshape(ops.transpose(pdist_matrix), [-1, 1])
        ),
    )
    mask_final = ops.reshape(
        ops.greater(
            ops.sum(
                ops.cast(mask, dtype="float32"), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = ops.transpose(mask_final)

    adjacency_not = ops.cast(adjacency_not, dtype="float32")
    mask = ops.cast(mask, dtype="float32")

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = ops.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = ops.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = ops.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = ops.cast(adjacency, dtype="float32") - ops.diag(ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = ops.sum(mask_positives)

    triplet_loss = ops.true_divide(
        ops.sum(
            ops.maximum(ops.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    return triplet_loss


def triplet_hard_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: TensorLike = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> TensorLike:
    r"""Computes the triplet loss with hard negative and hard positive mining.

    Usage:

    >>> y_true = ops.convert_to_tensor([0, 0])
    >>> y_pred = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> kca.losses.triplet_hard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: ops.matmul(x, x, transpose_b=True)
    >>> kca.losses.triplet_hard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.0>

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      soft: Boolean, if set, use the soft margin version.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.

        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.

    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """
    labels = ops.convert_to_tensor(y_true)
    precise_embeddings = ops.convert_to_tensor(y_pred)

    # Reshape label tensor to [batch_size, 1].
    lshape = ops.shape(labels)
    labels = ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = pairwise_distance(precise_embeddings, squared=False)

    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance(precise_embeddings, squared=True)

    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = ops.equal(labels, ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = ops.logical_not(adjacency)

    adjacency_not = ops.cast(adjacency_not, dtype="float32")
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = ops.size(labels)

    adjacency = ops.cast(adjacency, dtype="float32")

    mask_positives = ops.cast(adjacency, dtype="float32") - ops.diag(ops.ones([batch_size]))

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = ops.log1p(ops.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = ops.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = ops.mean(triplet_loss)

    return triplet_loss
