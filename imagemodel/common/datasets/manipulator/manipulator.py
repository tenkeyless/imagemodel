from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf
from imagemodel.common.datasets.manipulator.helper import (
    ManipulatorInputHelper,
    ManipulatorOutputHelper,
    PassManipulatorInputHelper,
    PassManipulatorOutputHelper,
)


class Manipulator(metaclass=ABCMeta):
    """
    <Interface>

    Properties
    ----------
    input_helper: ManipulatorInputHelper
        Input helper.

    Methods
    -------
    get_zipped_dataset() -> tf.data.Dataset

    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_zipped_dataset")
            and callable(subclass.get_zipped_dataset)
            and hasattr(subclass, "input_helper")
            and callable(subclass.input_helper)
            or NotImplemented
        )

    @property
    @abstractmethod
    def input_helper(self) -> ManipulatorInputHelper:
        """
        ManipulatorInputHelper: It returns input helper.
        """
        raise NotImplementedError

    def get_input_dataset(self) -> List[tf.data.Dataset]:
        return self.input_helper.get_inputs()

    def get_zipped_dataset(self) -> tf.data.Dataset:
        """


        Returns
        -------
        tf.data.Dataset
            It returns tuple of `(Tuple of inputs, Tuple of outputs)`.
        """

        input_dataset = tf.data.Dataset.zip(tuple(self.input_helper.get_inputs()))
        return tf.data.Dataset.zip(input_dataset)


class PassManipulator(Manipulator):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets = datasets

    @property
    def input_helper(self) -> ManipulatorInputHelper:
        return PassManipulatorInputHelper(self._datasets)


class SupervisedManipulator(Manipulator, metaclass=ABCMeta):
    """
    <Interface>

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
            hasattr(subclass, "get_zipped_dataset")
            and callable(subclass.get_zipped_dataset)
            and hasattr(subclass, "input_helper")
            and callable(subclass.input_helper)
            and hasattr(subclass, "output_helper")
            and callable(subclass.output_helper)
            or NotImplemented
        )

    @property
    @abstractmethod
    def output_helper(self) -> ManipulatorOutputHelper:
        """
        ManipulatorOutputHelper: It returns output helper.
        """
        raise NotImplementedError

    def get_output_dataset(self) -> List[tf.data.Dataset]:
        return self.output_helper.get_outputs()

    def get_zipped_dataset(self) -> tf.data.Dataset:
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


class PassSupervisedManipulator(SupervisedManipulator):
    """
    [summary]

    Parameters
    ----------
    SupervisedManipulator : [type]
        [description]

    Examples
    --------
    >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.bs_oxford_iiit_pet_feeder import BSOxfordIIITPetTrainingFeeder
    >>> training_feeder = BSOxfordIIITPetTrainingFeeder()
    >>> from imagemodel.common.datasets.manipulator.manipulator import PassSupervisedManipulator
    >>> supervised_training_manipulator = PassSupervisedManipulator(
    ...     training_feeder.input_helper.get_inputs(),
    ...     training_feeder.output_helper.get_outputs()
    ... )
    """

    def __init__(
        self,
        input_datasets: List[tf.data.Dataset],
        output_datasets: List[tf.data.Dataset],
    ):
        self._input_datasets = input_datasets
        self._output_datasets = output_datasets

    @property
    def input_helper(self) -> ManipulatorInputHelper:
        return PassManipulatorInputHelper(self._input_datasets)

    @property
    def output_helper(self) -> ManipulatorOutputHelper:
        return PassManipulatorOutputHelper(self._output_datasets)
