import numpy as np

from beartype import beartype
from beartype.door import is_bearable

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import DisorderedResidue, Residue

from environs import Env
from jaxtyping import (
    Float,
    Int,
    Bool,
    Shaped,
    jaxtyped
)

from torch import Tensor

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

Shaped = TorchTyping(Shaped)
Float  = TorchTyping(Float)
Int    = TorchTyping(Int)
Bool   = TorchTyping(Bool)

# helper type aliases

IntType = int | np.int32 | np.int64
AtomType = Atom | DisorderedAtom
ResidueType = Residue | DisorderedResidue
ChainType = Chain
TokenType = AtomType | ResidueType

# use env variable TYPECHECK to control whether to use beartype + jaxtyping

should_typecheck = env.bool('TYPECHECK', False)

typecheck = jaxtyped(typechecker = beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

__all__ = [
    Shaped,
    Float,
    Int,
    Bool,
    typecheck,
    beartype_isinstance
]
