from keras_core import ops
from kca.ops.norm import l2_normalize
from kca.utils.types import TensorLike


def pairwise_distance(feature: TensorLike, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
      squared: Boolean, whether to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    pairwise_distances_squared = ops.add(
        ops.sum(ops.square(feature), axis=[1], keepdims=True),
        ops.sum(ops.square(ops.transpose(feature)), axis=[0], keepdims=True),
    ) - 2.0 * ops.matmul(feature, ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = ops.sqrt(
            pairwise_distances_squared
            + ops.cast(error_mask, dtype="float32") * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = ops.multiply(
        pairwise_distances,
        ops.cast(ops.logical_not(error_mask), dtype="float32"),
    )

    num_data = ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = ops.ones_like(pairwise_distances) - ops.diag(ops.ones([num_data]))
    pairwise_distances = ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def angular_distance(feature: TensorLike):
    """Computes the angular distance matrix.

    output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])

    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.

    Returns:
      angular_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # normalize input
    feature = l2_normalize(feature, axis=1)
    feature /= ops.sqrt(ops.maximum(ops.sum(feature**2, axis=1), 1e-12))

    # create adjaceny matrix of cosine similarity
    angular_distances = 1 - ops.matmul(feature, feature.transpose())

    # ensure all distances > 1e-16
    angular_distances = ops.maximum(angular_distances, 0.0)

    return angular_distances
