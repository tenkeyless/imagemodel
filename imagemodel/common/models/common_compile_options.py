from typing import List, Optional, Union

from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer


class CompileOptions:
    def __init__(
            self,
            optimizer: Optimizer,
            loss_functions: List[Loss],
            metrics: Union[List[Metric], List[List[Metric]]],
            loss_weights_optional: Optional[List[float]] = None):
        self.optimizer: Optimizer = optimizer
        self.loss_functions: List[Loss] = loss_functions
        self.loss_weights_optional: Optional[List[float]] = loss_weights_optional
        self.metrics: Union[List[Metric], List[List[Metric]]] = metrics
