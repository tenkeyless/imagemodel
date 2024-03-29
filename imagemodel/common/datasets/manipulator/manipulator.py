from abc import ABCMeta, abstractmethod
from typing import Generic, List, TypeVar

import tensorflow as tf

from imagemodel.common.datasets.manipulator.helper import (
    ManipulatorInputHelper,
    ManipulatorOutputHelper,
    PassManipulatorInputHelper,
    PassManipulatorOutputHelper,
)

HI = TypeVar('HI', bound=ManipulatorInputHelper)


class Manipulator(Generic[HI], metaclass=ABCMeta):
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
    def input_helper(self) -> HI:
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
    def __init__(self, manipulator: Manipulator):
        self._manipulator: Manipulator = manipulator
    
    @property
    def input_helper(self) -> ManipulatorInputHelper:
        return PassManipulatorInputHelper(self._manipulator.get_input_dataset())


HO = TypeVar('HO', bound=PassManipulatorOutputHelper)


class SupervisedManipulator(Manipulator[HI], Generic[HI, HO], metaclass=ABCMeta):
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
    def output_helper(self) -> HO:
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
    
    def plot_zipped_dataset(self, sample_num: int, target_base_folder: str):
        """
        Plot zipped dataset and save them.
        If `target_base_folder` starts with 'gs://' it will try upload to google storage bucket.
        
        - Do not use inside manipulator. Use to show test image.
        - Do not use too large `sample_num`. (less than 10)
        - Use local `target_base_folder` rather than 'gs://'.
        
        Parameters
        ----------
        sample_num
        target_base_folder

        Returns
        -------

        """
        pass


class PassSupervisedManipulator(SupervisedManipulator):
    """
    [summary]

    Examples
    --------
    >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
    >>> training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
    >>> from imagemodel.common.datasets.manipulator.manipulator import PassSupervisedManipulator
    >>> supervised_training_manipulator = PassSupervisedManipulator(training_feeder)
    """
    
    def __init__(self, manipulator: SupervisedManipulator):
        self._manipulator: SupervisedManipulator = manipulator
    
    @property
    def input_helper(self) -> ManipulatorInputHelper:
        return PassManipulatorInputHelper(self._manipulator.get_input_dataset())
    
    @property
    def output_helper(self) -> ManipulatorOutputHelper:
        return PassManipulatorOutputHelper(self._manipulator.get_output_dataset())
