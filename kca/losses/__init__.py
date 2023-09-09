# kca/losses/__init__.py

__all__ = [
    "weighted_cross_entropy_with_logits", "binary_weighted_cross_entropy", "binary_balanced_cross_entropy",
    "multiclass_weighted_cross_entropy", "binary_focal_loss", "multiclass_focal_loss", "binary_tversky_coef",
    "binary_tversky_loss", "binary_dice_coef_loss", "binary_weighted_dice_crossentropy_loss",
    "multiclass_weighted_dice_loss", "multiclass_weighted_squared_dice_loss", "multiclass_weighted_tanimoto_loss",
    "ntxent"
]

from .crossentropy import (
    weighted_cross_entropy_with_logits, binary_weighted_cross_entropy,
    binary_balanced_cross_entropy, multiclass_weighted_cross_entropy
)

from .focal import binary_focal_loss, multiclass_focal_loss

from .tversky import binary_tversky_coef, binary_tversky_loss
from .dice import (
    binary_dice_coef_loss, binary_weighted_dice_crossentropy_loss,
    multiclass_weighted_dice_loss, multiclass_weighted_squared_dice_loss
)

from .tanimoto import multiclass_weighted_tanimoto_loss

from .ntxent import ntxent_loss
