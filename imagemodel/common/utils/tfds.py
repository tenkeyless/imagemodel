from typing import Optional


def int_range_to_string(
    begin_optional: Optional[int] = None,
    end_optional: Optional[int] = None,
    unit: Optional[str] = None,
    default_begin_value: str = "",
    default_end_value: str = "",
) -> Optional[str]:
    """
    Change range int to string for TFDS.

    Parameters
    ----------
    begin_optional : Optional[int], optional, default=None
        Start value for string range.
    end_optional : Optional[int], optional, default=None
        End value for string range.
    unit : Optional[str], optional, default=None
        Unit for range.
    default_begin_value : str, optional, default=""
        Default value for begin.
    default_end_value : str, optional, default=""
        Default value for end.

    Returns
    -------
    Optional[str]
        Range string for TFDS load.

    Examples
    --------
    >>> int_range_to_string(begin_optional=30, unit="%")
    "30%:"
    >>> int_range_to_string(begin_optional=10, end_optional=50, unit="%")
    "10%:50%"
    >>> self.assertEqual(int_range_to_string(unit="%"), None)
    None
    """
    result: Optional[str] = None
    if begin_optional or end_optional:
        begin_string: str = (
            str(begin_optional) if begin_optional else default_begin_value
        )
        end_string: str = str(end_optional) if end_optional else default_end_value
        if unit:
            begin_string = (
                begin_string + unit
                if begin_string is not default_begin_value
                else begin_string
            )
            end_string = (
                end_string + unit if end_string is not default_end_value else end_string
            )
        result = "{}:{}".format(begin_string, end_string)
    return result


def append_tfds_str_range(
    option_string: str,
    begin_optional: Optional[int] = None,
    end_optional: Optional[int] = None,
) -> str:
    """
    Function for TFDS load range value.

    Parameters
    ----------
    option_string : str
        A prefix for result string.
    begin_optional : Optional[int], optional, default=None
        Begin range with percent.
    end_optional : Optional[int], optional, default=None
        End range with percent.

    Returns
    -------
    str
        Result with percent range.

    Examples
    --------
    >>> append_tfds_str_range(option_string="train")
    "train"
    >>> append_tfds_str_range(option_string="train", begin_optional=30)
    "train[30%:]"
    """
    result: str = option_string
    if begin_optional or end_optional:
        result = result + "[{}]".format(
            int_range_to_string(begin_optional, end_optional, "%")
        )
    return result
