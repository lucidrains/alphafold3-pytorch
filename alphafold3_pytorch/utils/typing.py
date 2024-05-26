import os
from functools import wraps

import rootutils
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

# environment

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# function


def null_decorator(fn):
    """A null decorator."""

    @wraps(fn)
    def inner(*args, **kwargs):
        """Run an inner function."""
        return fn(*args, **kwargs)

    return inner


# NOTE: `jaxtyping` is a misnomer, works for PyTorch as well


class TorchTyping:
    """Torch typing."""

    def __init__(self, abstract_dtype):
        """Initialize typing."""
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Get item."""
        return self.abstract_dtype[Tensor, shapes]


Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

# NOTE: use env variable `TYPECHECK` (which is set by `rootutils` above using `.env`) to control whether to use `beartype` + `jaxtyping`

should_typecheck = os.environ.get("TYPECHECK", False)

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else null_decorator

__all__ = [Float, Int, Bool, typecheck]
