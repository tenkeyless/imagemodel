from typing import Callable, Generic, TypeVar

import tensorflow as tf

from imagemodel.common.datasets.augmenter import Augmenter
from imagemodel.common.datasets.feeder import Feeder
from imagemodel.common.datasets.preprocessor import Preprocessor
from imagemodel.common.datasets.regularizer import Regularizer

F = TypeVar('F', bound=Feeder)
A = TypeVar('A', bound=Augmenter)
R = TypeVar('R', bound=Regularizer)
P = TypeVar('P', bound=Preprocessor)


class Pipeline(Generic[F, A, R, P]):
    def __init__(
            self,
            feeder: F,
            augmenter_func: Callable[[F], A],
            regularizer_func: Callable[[A], R],
            preprocessor_func: Callable[[R], P],
    ):
        self.feeder: F = feeder
        self.augmenter: A = augmenter_func(self.feeder)
        self.regularizer: R = regularizer_func(self.augmenter)
        self.preprocessor: P = preprocessor_func(self.regularizer)
    
    @property
    def data_description(self):
        return self.feeder.feeder_data_description
    
    def get_input_zipped_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.zip(tuple(self.preprocessor.get_input_dataset()))
    
    def get_zipped_dataset(self) -> tf.data.Dataset:
        return self.preprocessor.get_zipped_dataset()
