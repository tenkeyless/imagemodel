from abc import ABCMeta
from typing import List

import tensorflow as tf


class ManipulatorInputHelper(metaclass=ABCMeta):
    """
    <Interface>

    Methods
    -------
    get_inputs() -> List[tf.data.Dataset]
    """

    def get_inputs(self) -> List[tf.data.Dataset]:
        pass


class PassManipulatorInputHelper(ManipulatorInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets = datasets

    def get_inputs(self) -> List[tf.data.Dataset]:
        return self._datasets


class ManipulatorOutputHelper(metaclass=ABCMeta):
    """
    <Interface>

    Methods
    -------
    get_outputs() -> List[tf.data.Dataset]
    """

    def get_outputs(self) -> List[tf.data.Dataset]:
        pass


class PassManipulatorOutputHelper(ManipulatorOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets = datasets

    def get_outputs(self) -> List[tf.data.Dataset]:
        return self._datasets
