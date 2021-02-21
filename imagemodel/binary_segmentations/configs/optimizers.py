from enum import Enum

from tensorflow.keras.optimizers import Adam


class Optimizers(Enum):
    adam1 = "adam1"
    adam2 = "adam2"

    def get_optimizer(self):
        if self == Optimizers.adam1:
            return Adam(lr=1e-4)
        elif self == Optimizers.adam2:
            return Adam(lr=1e-3)
        else:
            return Adam(lr=1e-4)

    @staticmethod
    def get_default() -> str:
        return Optimizers.adam1.value
