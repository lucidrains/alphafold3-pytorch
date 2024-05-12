from functools import wraps
from environs import Env

from torch import Tensor

from beartype import beartype
from jaxtyping import (
    Float,
    Int,
    Bool,
    jaxtyped
)

# environment

env = Env()
env.read_env()

# function

def null_decorator(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)
    return inner

# jaxtyping is a misnomer, works for pytorch

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(Float)
Int   = TorchTyping(Int)
Bool  = TorchTyping(Bool)

# use env variable TYPECHECK to control whether to use beartype + jaxtyping

should_typecheck = env.bool('TYPECHECK', False)

typecheck = jaxtyped(typechecker = beartype) if should_typecheck else null_decorator

__all__ = [
    Float,
    Int,
    Bool,
    typecheck
]
