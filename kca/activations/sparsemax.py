# activations/sparsemax.py

from keras_core import ops
import tensorflow as tf

from kca.utils.types import TensorLike, Int

__all__ = ["sparsemax"]


def sparsemax(logits: TensorLike, axis: Int = -1) -> TensorLike:
    r"""Sparsemax activation function.

    For each batch $i$, and class $j$,
    compute sparsemax activation function:

    $$
    \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
    $$

    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).

    Usage:

    >>> x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> sparsemax(x)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 1.],
           [0., 0., 1.]], dtype=float32)>

    Args:
        logits: A `Tensor`.
        axis: `int`, axis along which the sparsemax operation is applied.
    Returns:
        A `Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """

    # We need its original shape for shape inference.
    shape = ops.shape(logits)
    rank = ops.ndim(logits)
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = ops.ndim(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, ops.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, ops.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return ops.transpose(
        logits,
        ops.concatenate(
            [
                ops.arange(dim_index),
                [last_index],
                ops.arange(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = ops.shape(logits)
    obs = ops.prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = ops.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = ops.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = ops.cumsum(z_sorted, axis=-1)
    k = ops.arange(1, ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = ops.sum(ops.cast(z_check, int), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = ops.maximum(k_z, 1)

    indices = ops.stack([ops.arange(0, obs), ops.reshape(k_z_safe, [-1]) - 1], axis=1)

    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / ops.cast(k_z, logits.dtype)

    # calculate p
    p = ops.maximum(ops.cast(0, logits.dtype), z - ops.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = ops.where(
        ops.expand_dims(
            ops.logical_or(ops.equal(k_z, 0), ops.isnan(z_cumsum[:, -1])),
            axis=-1,
        ),
        ops.full([obs, dims], ops.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = ops.reshape(p_safe, shape_op)
    return p_safe
