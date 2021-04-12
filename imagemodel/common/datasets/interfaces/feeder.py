from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf


class TFSupervisionPurposeInputHelper(metaclass=ABCMeta):
    """
    <Interface> Methods input data for supervised learning.

    Methods
    -------
    get_inputs() -> List[tf.data.Dataset]
    """

    def get_inputs(self) -> List[tf.data.Dataset]:
        pass


class TFSupervisionPurposeOutputHelper(metaclass=ABCMeta):
    """
    <Interface> Methods output data for supervised learning.

    Methods
    -------
    get_outputs() -> List[tf.data.Dataset]
    """

    def get_outputs(self) -> List[tf.data.Dataset]:
        pass


class TFSupervisionFeeder(metaclass=ABCMeta):
    """
    <Interface> Methods for supervised learning.

    Properties
    ----------
    input_helper: TFSupervisionPurposeInputHelper
        Input helper.
    output_helper: TFSupervisionPurposeOutputHelper
        Output helper.

    Methods
    -------
    get_supervised_dataset() -> tf.data.Dataset
        With default implementation.
        Input, Output dataset for supervised training.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_supervised_dataset")
            and callable(subclass.get_supervised_dataset)
            and hasattr(subclass, "input_helper")
            and callable(subclass.input_helper)
            and hasattr(subclass, "output_helper")
            and callable(subclass.output_helper)
            or NotImplemented
        )

    @property
    @abstractmethod
    def input_helper(self) -> TFSupervisionPurposeInputHelper:
        """
        TFSupervisionPurposeInputHelper: It returns input helper.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_helper(self) -> TFSupervisionPurposeOutputHelper:
        """
        TFSupervisionPurposeOutputHelper: It returns output helper.
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

        input_dataset = tf.data.Dataset.zip(tuple(self.input_helper.get_inputs()))
        output_dataset = tf.data.Dataset.zip(tuple(self.output_helper.get_outputs()))
        return tf.data.Dataset.zip((input_dataset, output_dataset))
