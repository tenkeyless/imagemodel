from abc import ABCMeta, abstractmethod
from typing import Tuple

import tensorflow as tf


class Transformer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def in_dataset(self) -> tf.data.Dataset:
        pass
    
    @property
    @abstractmethod
    def out_dataset(self) -> tf.data.Dataset:
        pass
    
    @abstractmethod
    def plot_out_dataset(self, sample_num: int, target_base_folder: str):
        pass


class TransformerT(Transformer, metaclass=ABCMeta):
    @property
    @abstractmethod
    def resize_to(self) -> Tuple[int, int]:
        pass


class TransformerP(Transformer, metaclass=ABCMeta):
    @property
    @abstractmethod
    def resize_to(self) -> Tuple[int, int]:
        pass
