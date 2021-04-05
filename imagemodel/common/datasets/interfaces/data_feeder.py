from abc import ABCMeta, abstractmethod

import tensorflow as tf
import tensorflow_datasets as tfds


class TFDSDataFeeder(metaclass=ABCMeta):
    """
    <Interface>

    Methods for tfds.

    Attributes
    ----------
    original_dataset : tf.data.Dataset
        A dataset of `tf.data.Dataset`.
    original_info : tfds.core.DatasetInfo
        Dataset info.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "original_dataset")
            and callable(subclass.original_dataset)
            and hasattr(subclass, "original_info")
            and callable(subclass.original_info)
            or NotImplemented
        )

    @property
    @abstractmethod
    def original_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    @property
    @abstractmethod
    def original_info(self) -> tfds.core.DatasetInfo:
        raise NotImplementedError


class BaseTFDSDataFeeder(TFDSDataFeeder):
    """
    Base implementation of `TFDSDataFeeder`.

    Attributes
    ----------
    original_dataset : tf.data.Dataset
        A dataset of `tf.data.Dataset`.
    original_info : tfds.core.DatasetInfo
        Dataset info.
    """

    def __init__(
        self, original_dataset: tf.data.Dataset, original_info: tfds.core.DatasetInfo
    ):
        self._original_dataset = original_dataset
        self._original_info = original_info

    @property
    def original_dataset(self) -> tf.data.Dataset:
        return self._original_dataset

    @property
    def original_info(self) -> tfds.core.DatasetInfo:
        return self._original_info
