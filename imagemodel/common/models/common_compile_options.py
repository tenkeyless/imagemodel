from typing import List

from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer


class CompileOptions:
    def __init__(self, optimizer: Optimizer, loss_functions: List[Loss], metrics: List[Metric]):
        self.optimizer: Optimizer = optimizer
        self.loss_functions: List[Loss] = loss_functions
        self.metrics: List[Metric] = metrics
