import functools


def compose(*functions):
    """
    Compuse functions.

    Returns
    -------
    Function
        Composed function.

    Examples
    --------
    >>> def a(v_a):
    ...     return v_a+1
    ...
    >>> def b(v_b):
    ...     return v_b*2
    ...
    >>> def c(v_c):
    ...     return v_c*3
    ...
    >>> compose(a, b, c)(1)  # run function c -> b -> a  # ((1*3)*2)+1
    7
    >>> compose(a, c, b)(1)  # run function b -> c -> a  # ((1*2)*3)+1
    7
    >>> compose(c, a, b)(1)  # run function b -> a -> c  # ((1*2)+1)*3
    9
    >>> compose(c, a, c)(1)  # run function c -> a -> c  # ((1*3)+1)*3
    12
    >>> compose(b, a, c)(1)  # run function c -> a -> b  # ((1*3)+1)*2
    8
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def compose_left(*functions):
    """
    Compuse functions from left.

    Returns
    -------
    Function
        Composed function.

    Examples
    --------
    >>> def a(v_a):
    ...     return v_a+1
    ...
    >>> def b(v_b):
    ...     return v_b*2
    ...
    >>> def c(v_c):
    ...     return v_c*3
    ...
    >>> compose_left(a, b, c)(1)  # run function a -> b -> c  # ((1+1)*2)*3
    12
    >>> compose_left(a, c, b)(1)  # run function a -> c -> b  # ((1+1)*3)*2
    7
    >>> compose_left(c, a, b)(1)  # run function c -> a -> b  # ((1*3)+1)*2
    8
    >>> compose_left(c, a, c)(1)  # run function c -> a -> c  # ((1*3)+1)*3
    12
    >>> compose_left(b, a, c)(1)  # run function b -> a -> c  # ((1*2)+1)*3
    9
    """
    return functools.reduce(lambda g, f: lambda x: f(g(x)), functions, lambda x: x)
