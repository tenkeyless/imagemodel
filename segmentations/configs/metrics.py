from enum import Enum

from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy


class Metrics(Enum):
    none = "none"
    accuracy = "accuracy"
    ca = "categorical_accuracy"
    ba = "binary_accuracy"

    def get_metric(self):
        if self == Metrics.accuracy:
            return "accuracy"
        elif self == Metrics.ca:
            return CategoricalAccuracy(name="categorical_accuracy")
        elif self == Metrics.ba:
            return BinaryAccuracy(name="binary_accuracy")
        else:
            return None

    @staticmethod
    def get_default() -> str:
        return Metrics.none.value
