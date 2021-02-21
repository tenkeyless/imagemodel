from enum import Enum

from tensorflow.keras.losses import BinaryCrossentropy


class Losses(Enum):
    bce = "binary_crossentropy"

    def get_loss(self):
        if self == Losses.bce:
            return BinaryCrossentropy()
        else:
            return BinaryCrossentropy()

    @staticmethod
    def get_default() -> str:
        return Losses.bce.value
