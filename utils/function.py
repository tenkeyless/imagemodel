import inspect


def get_default_args(func):
    """
    Return dict for parameter name and default value.

    Parameters
    ----------
    func : Callable
        A function to get parameter name and default value.

    Returns
    -------
    Dict
        Parameter name and default value.

    Examples
    --------
    >>> def test_func(a: int, b: str = "c") -> int:
    ...     return a+1
    >>> get_default_args(test_func)
    {'b': 'c'}

    >>> def test_func2(a: int = 1, b="c") -> int:
    ...     return a+1
    >>> get_default_args(test_func2)
    {'a': 1, 'b': 'c'}
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_annotations(func):
    """
    Return dict for parameter name and type.

    Parameters
    ----------
    func : Callable
        A function to get parameter name and type.
        The type must be specified as a type hint.

    Returns
    -------
    Dict
        Parameter name and type tuple.

    Examples
    --------
    >>> def test_func(a: int, b: str) -> int:
    ...     return a+1
    >>> get_annotations(test_func)
    {'a': <class 'int'>, 'b': <class 'str'>}

    >>> def test_func2(a: int, b) -> int:
    ...     return a+1
    >>> get_annotations(test_func)
    {'a': <class 'int'>, 'b': <class 'inspect._empty'>}
    """
    signature = inspect.signature(func)
    return {k: v.annotation for k, v in signature.parameters.items()}
