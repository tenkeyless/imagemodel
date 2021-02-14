from typing import List, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def check_all_exists_or_not(list: List[T1]) -> bool:
    """
    Check `None` exists or not in `list`.

    Parameters
    ----------
    list : List[T1]
        A list to check `None` existance.

    Returns
    -------
    bool
        `False` if `list` contains `None`.

    Examples
    --------
    >>> check_all_exists_or_not([1, None])
    False
    >>> check_all_exists_or_not([1, 2])
    True
    """
    return not (any(list) and not all(list))


def check_exists_or_not(*args) -> bool:
    """
    Check `None` exists or not in arguments.

    Returns
    -------
    bool
        `False` if arguments contain `None`.

    Examples
    --------
    >>> check_exists_or_not(1, None)
    False
    >>> check_exists_or_not(1, 2, "a")
    True
    """
    return check_all_exists_or_not(list(args))


def sublist(lst1: List[T1], lst2: List[T1]) -> bool:
    """
    Check `lst1` is sublist of `lst2`.

    Parameters
    ----------
    lst1 : List[T1]
        List 1.
    lst2 : List[T1]
        List 2.

    Returns
    -------
    bool
        `True` if `lst1` is sublist of `lst2`.

    Examples
    --------
    >>> sublist([1,2,3], [1,2,3])
    True
    >>> sublist([1,2,3], [1,2,3,4])
    True
    >>> sublist([1,2,3,5], [1,2,3,4])
    False
    """
    return set(lst1) <= set(lst2)
