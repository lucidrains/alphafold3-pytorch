import numpy as np

from beartype.typing import Any, Iterable, List


def exists(val: Any) -> bool:
    """Check if a value exists.

    :param val: The value to check.
    :return: `True` if the value exists, otherwise `False`.
    """
    return val is not None


def not_exists(val: Any) -> bool:
    """Check if a value does not exist.

    :param val: The value to check.
    :return: `True` if the value does not exist, otherwise `False`.
    """
    return val is None


def default(v: Any, d: Any) -> Any:
    """Return default value `d` if `v` does not exist (i.e., is `None`).

    :param v: The value to check.
    :param d: The default value to return if `v` does not exist.
    :return: The value `v` if it exists, otherwise the default value `d`.
    """
    return v if exists(v) else d


def first(arr: Iterable[Any]) -> Any:
    """Return the first element of an iterable object such as a list.

    :param arr: An iterable object.
    :return: The first element of the iterable object.
    """
    return arr[0]


def always(value):
    """Always return a value."""

    def inner(*args, **kwargs):
        """Inner function."""
        return value

    return inner


def identity(x, *args, **kwargs):
    """Return the input value."""
    return x


def np_mode(x: np.ndarray) -> Any:
    """Return the mode of a 1D NumPy array."""
    assert x.ndim == 1, f"Input NumPy array must be 1D, not {x.ndim}D."
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]
