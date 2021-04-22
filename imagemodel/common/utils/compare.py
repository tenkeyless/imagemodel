from typing import Callable, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def compare_func(func: Callable[[T1], T2], obj1: T1, obj2: T1) -> bool:
    return func(obj1) == func(obj2)
