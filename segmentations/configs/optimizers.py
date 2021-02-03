from enum import Enum

from tensorflow.keras.optimizers import Adam


class RefOptimizer(Enum):
    adam1 = "adam1"
    adam2 = "adam2"

    def get_optimizer(self):
        if self == RefOptimizer.adam1:
            return Adam(lr=1e-4)
        elif self == RefOptimizer.adam2:
            return Adam(lr=1e-3)
        else:
            return Adam(lr=1e-4)

    @staticmethod
    def get_default() -> str:
        return RefOptimizer.adam1.value
