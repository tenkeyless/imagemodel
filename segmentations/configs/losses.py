from enum import Enum

from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


class Losses(Enum):
    cce = "categorical_crossentropy"
    bce = "binary_crossentropy"

    def get_loss(self):
        if self == Losses.cce:
            return CategoricalCrossentropy()
        elif self == Losses.bce:
            return BinaryCrossentropy()
        else:
            return CategoricalCrossentropy()

    @staticmethod
    def get_default() -> str:
        return Losses.cce.value
