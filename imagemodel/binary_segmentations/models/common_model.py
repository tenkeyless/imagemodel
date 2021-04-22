import abc
from abc import abstractmethod
from typing import Generic, Dict, TypeVar

from tensorflow.python.keras import Model
from typing_extensions import TypedDict

ST = TypeVar('ST', bound=TypedDict)


class CommonModel(metaclass=abc.ABCMeta):
    @abstractmethod
    def setup_model(self) -> Model:
        pass

    @staticmethod
    def layer_name_correction(name: str) -> str:
        return name.replace(" ", "_")


class CommonModelDictGeneratable(Generic[ST], metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def init_with_dict(cls, option_dict: ST):
        pass

    @classmethod
    @abstractmethod
    def init_with_str_dict(cls, option_str_dict: Dict[str, str]):
        pass
