from typing import Callable, Optional, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def optional_map(value_optional: Optional[T1], func: Callable[[T1], T2]) -> Optional[T2]:
    """
    Map for optional value.

    Parameters
    ----------
    value_optional : Optional[T1]
        Optional value.
    func : Callable[[T1], T2]
        A function to apply to `value_optional`.

    Returns
    -------
    Optional[T2].
        Optional value applied `func` function.

    Examples
    --------
    >>> optional_map(None, lambda el: el+1)
    >>> optional_map(4, lambda el: el+1)
    5
    """
    return func(value_optional) if value_optional is not None else None
