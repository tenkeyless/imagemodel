from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Drafter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def out_dataset(self) -> tf.data.Dataset:
        pass


class DrafterT(Drafter, metaclass=ABCMeta):
    pass


class DrafterP(Drafter, metaclass=ABCMeta):
    pass
