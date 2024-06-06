from functools import wraps
from environs import Env

from torch import Tensor

from beartype import beartype
from beartype.door import is_bearable

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

def always(value):
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    return t

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

typecheck = jaxtyped(typechecker = beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

__all__ = [
    Float,
    Int,
    Bool,
    typecheck,
    beartype_isinstance
]
