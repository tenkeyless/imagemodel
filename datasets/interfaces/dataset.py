from abc import ABCMeta, abstractmethod
from typing import Optional


class TfdsDatasetInterface(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_training_dataset")
            and callable(subclass.get_training_dataset)
            and hasattr(subclass, "get_validation_dataset")
            and callable(subclass.get_validation_dataset)
            and hasattr(subclass, "get_test_dataset")
            and callable(subclass.get_test_dataset)
            or NotImplemented
        )

    @abstractmethod
    def get_training_dataset(self, batch_size_optional: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def get_training_dataset_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_validation_dataset(self, batch_size_optional: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def get_validation_dataset_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_test_dataset(self, batch_size_optional: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def get_test_dataset_length(self) -> int:
        raise NotImplementedError
