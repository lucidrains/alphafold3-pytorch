from __future__ import annotations
from functools import partial
import importlib.metadata
import torch
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
from loguru import logger

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

# some more colocated environmental stuff

def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.
    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

# maybe deespeed checkpoint, and always use non reentrant checkpointing

DEEPSPEED_CHECKPOINTING = env.bool('DEEPSPEED_CHECKPOINTING', False)

if DEEPSPEED_CHECKPOINTING:
    assert package_available("deepspeed"), "DeepSpeed must be installed for checkpointing."

    import deepspeed

    checkpoint = deepspeed.checkpointing.checkpoint
else:
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant = False)

# check is github ci

IS_GITHUB_CI = env.bool('IS_GITHUB_CI', False)

# use env variable TYPECHECK to control whether to use beartype + jaxtyping

should_typecheck = env.bool('TYPECHECK', False)
IS_DEBUGGING = env.bool('DEBUG', False)

typecheck = jaxtyped(typechecker = beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

if should_typecheck:
    logger.info("Type checking is enabled.")
else:
    logger.info("Type checking is disabled.")

if IS_DEBUGGING:
    logger.info("Debugging is enabled.")
else:
    logger.info("Debugging is disabled.")

__all__ = [
    Shaped,
    Float,
    Int,
    Bool,
    typecheck,
    should_typecheck,
    beartype_isinstance,
    checkpoint,
    IS_DEBUGGING,
    IS_GITHUB_CI
]
