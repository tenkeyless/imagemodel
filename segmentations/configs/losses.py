from enum import Enum

from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
)


class Losses(Enum):
    cce = "categorical_crossentropy"
    scce_logit = "sparse_categorical_crossentropy_from_logits"
    bce = "binary_crossentropy"

    def get_loss(self):
        if self == Losses.cce:
            return CategoricalCrossentropy()
        elif self == Losses.bce:
            return BinaryCrossentropy()
        elif self == Losses.scce_logit:
            return SparseCategoricalCrossentropy(from_logits=True)
        else:
            return CategoricalCrossentropy()

    @staticmethod
    def get_default() -> str:
        return Losses.cce.value
