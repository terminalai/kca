# losses/ntxent.py

from keras_core import ops, losses

from kca.utils.types import TensorLike, Float, Number
from typing import Callable

__all__ = ["ntxent_loss"]


def ntxent_loss(temperature: Float = 1.0) -> Callable:
    """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in
    [SimCLR](https://arxiv.org/abs/2002.05709).
    Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

    Args:
        temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
    """

    def ntxent(z_i: TensorLike, z_j: TensorLike) -> Number:
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1)
        samples in the batch

        Args:
            z_i : anchor batch of samples
            z_j : positive batch of samples

        Returns:
            float: loss
        """
        b = ops.shape(z_i)[0]

        # compute similarity between the sample's embedding and its corrupted view

        z = ops.concatenate([z_i, z_j], axis=0)

        similarity = losses.cosine_similarity(
            ops.expand_dims(z, 1),
            ops.expand_dims(z, 0)
        )

        sim_ij = ops.diag(similarity, b)
        sim_ji = ops.diag(similarity, -b)
        positives = ops.concatenate([sim_ij, sim_ji], axis=0)

        mask = (~ops.identity(b*2, dtype="bool")).astype(float)
        numerator = ops.exp(positives / temperature)
        denominator = ops.exp(mask * ops.exp(similarity / temperature))

        all_losses = -ops.log(numerator / ops.sum(denominator, axis=1))
        loss = ops.sum(all_losses) / (2 * b)

        return loss

    return ntxent

