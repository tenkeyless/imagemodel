import abc
from abc import abstractmethod
from typing import Generic, Dict, TypeVar

from typing_extensions import TypedDict

AT = TypeVar('AT', bound=TypedDict)


class ModelArguments(Generic[AT], metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def init_from_str_dict(cls, option_str_dict: Dict[str, str]):
        pass

    @classmethod
    @abstractmethod
    def convert_str_dict(cls, option_string_dict: Dict[str, str]) -> AT:
        pass
