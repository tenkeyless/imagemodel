from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf


class TFSupervisionFeeder(metaclass=ABCMeta):
    """
    <Interface>

    Methods for supervised learning.

    Methods
    -------
    get_inputs() -> List[tf.data.Dataset]
        Dataset for input.
    get_outputs() -> List[tf.data.Dataset]
        Dataset for output.
    get_supervised_dataset() -> tf.data.Dataset
        With default implementation.
        Input, Output dataset for supervised training.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_supervised_dataset")
            and callable(subclass.get_supervised_dataset)
            and hasattr(subclass, "get_inputs")
            and callable(subclass.get_inputs)
            and hasattr(subclass, "get_outputs")
            and callable(subclass.get_outputs)
            or NotImplemented
        )

    @abstractmethod
    def get_inputs(self) -> List[tf.data.Dataset]:
        """
        It returns input dataset.

        Returns
        -------
        List[tf.data.Dataset]
            Input dataset list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_outputs(self) -> List[tf.data.Dataset]:
        """
        It returns output dataset.

        Returns
        -------
        List[tf.data.Dataset]
            Output dataset list.
        """
        raise NotImplementedError

    def get_supervised_dataset(self) -> tf.data.Dataset:
        """
        It returns combined input and output dataset.

        Returns
        -------
        tf.data.Dataset
            It returns tuple of `(Tuple of inputs, Tuple of outputs)`.
        """
        input_dataset = tf.data.Dataset.zip(tuple(self.get_inputs()))
        output_dataset = tf.data.Dataset.zip(tuple(self.get_outputs()))
        return tf.data.Dataset.zip((input_dataset, output_dataset))
