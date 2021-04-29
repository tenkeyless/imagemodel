from abc import ABCMeta, abstractmethod

import tensorflow as tf


class TFDataDescriptor(metaclass=ABCMeta):
    """
    <Interface>

    Methods for tfds.

    Attributes
    ----------
    original_dataset : tf.data.Dataset
        A dataset of `tf.data.Dataset`.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, "original_dataset")
                and callable(subclass.original_dataset)
                or NotImplemented
        )

    @property
    @abstractmethod
    def original_dataset(self) -> tf.data.Dataset:
        """
        tf.data.Dataset: Original dataset.
        """
        raise NotImplementedError


class BaseTFDataDescriptor(TFDataDescriptor):
    """
    Base implementation of `TFDSDataFeeder`.

    Attributes
    ----------
    original_dataset : tf.data.Dataset
        A dataset of `tf.data.Dataset`.
    """

    @abstractmethod
    def __init__(self, original_dataset: tf.data.Dataset):
        self._original_dataset = original_dataset

    @property
    def original_dataset(self) -> tf.data.Dataset:
        return self._original_dataset
