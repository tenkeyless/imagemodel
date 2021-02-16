import inspect
from abc import ABCMeta, abstractmethod
from typing import Dict, Generic, List, TypeVar

from tensorflow.keras.models import Model
from utils.list import sublist

T = TypeVar("T")


class ModelInterface(Generic[T], metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "func")
            and hasattr(subclass, "get_model")
            and callable(subclass.get_model)
            and hasattr(subclass, "convert_str_model_option_dict")
            and callable(subclass.convert_str_model_option_dict)
            and hasattr(subclass, "post_processing")
            and callable(subclass.post_processing)
            and hasattr(subclass, "save_post_processed_result")
            and callable(subclass.save_post_processed_result)
            or NotImplemented
        )

    @property
    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def get_model(self, option_dict: T) -> Model:
        raise NotImplementedError

    @abstractmethod
    def convert_str_model_option_dict(self, option_dict: Dict[str, str]) -> T:
        raise NotImplementedError

    @abstractmethod
    def post_processing(self, predicted_result):
        raise NotImplementedError

    @abstractmethod
    def save_post_processed_result(self, filename: str, result):
        raise NotImplementedError

    def check_dict_option(self, option_dict: Dict[str, str]) -> bool:
        args: List[str] = self.get_model_option_keys()
        return sublist(list(option_dict), args)

    def check_dict_option_key_and_raise(self, option_dict: Dict[str, str]):
        if not self.check_dict_option(option_dict):
            model_option_keys: List[str] = self.get_model_option_keys()
            model_option_key_string: str = "', '".join(
                [str(elem) for elem in model_option_keys]
            )
            raise ValueError(
                "Check `model_option`s. `model_option` should be one of '{}'.".format(
                    model_option_key_string
                )
            )

    def get_model_option_keys(self) -> List[str]:
        return inspect.getfullargspec(self.func()).args

    def get_model_from_str_model_option(self, option_dict: Dict[str, str]) -> Model:
        converted_dict = self.convert_str_model_option_dict(option_dict)
        return self.get_model(converted_dict)
