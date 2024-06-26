from functools import wraps
from dataclasses import dataclass
from typing import Type, Literal, Callable, List, Any

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.tensor_typing import (
    typecheck,
    beartype_isinstance,
    Int, Bool, Float
)

from alphafold3_pytorch.life import (
    HUMAN_AMINO_ACIDS,
    NUCLEOTIDES,
    METALS,
    MISC
)

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

# functions

def exists(v):
    return v is not None

def identity(t):
    return t

def compose(*fns: Callable):
    # for chaining from Alphafold3Input -> MoleculeInput -> AtomInput

    def inner(x, *args, **kwargs):
        for fn in fns:
            x = fn(x, *args, **kwargs)
        return x
    return inner

# atom level, what Alphafold3 accepts

@typecheck
@dataclass
class AtomInput:
    atom_inputs:                Float['m dai']
    molecule_ids:               Int[' n']
    molecule_atom_lens:         Int[' n']
    atompair_inputs:            Float['m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Float[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    token_bonds:                Bool['n n'] | None = None
    atom_ids:                   Int[' m'] | None = None
    atom_parent_ids:            Int[' m'] | None = None
    atompair_ids:               Int['m m'] | Int['nw w (w*2)'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    atom_pos:                   Float['m 3'] | None = None
    molecule_atom_indices:      Int[' n'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int['n n'] | None = None
    plddt_labels:               Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None

@typecheck
@dataclass
class BatchedAtomInput:
    atom_inputs:                Float['b m dai']
    molecule_ids:               Int['b n']
    molecule_atom_lens:         Int['b n']
    atompair_inputs:            Float['b m m dapi'] | Float['b nw w (w*2) dapi']
    additional_molecule_feats:  Float[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'b n {IS_MOLECULE_TYPES}']
    templates:                  Float['b t n n dt']
    msa:                        Float['b s n dm']
    token_bonds:                Bool['b n n'] | None = None
    atom_ids:                   Int['b m'] | None = None
    atom_parent_ids:            Int['b m'] | None = None
    atompair_ids:               Int['b m m'] | Int['b nw w (w*2)'] | None = None
    template_mask:              Bool['b t'] | None = None
    msa_mask:                   Bool['b s'] | None = None
    atom_pos:                   Float['b m 3'] | None = None
    molecule_atom_indices:      Int['b n'] | None = None
    distance_labels:            Int['b n n'] | None = None
    pae_labels:                 Int['b n n'] | None = None
    pde_labels:                 Int['b n n'] | None = None
    plddt_labels:               Int['b n'] | None = None
    resolved_labels:            Int['b n'] | None = None

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens

@typecheck
@dataclass
class MoleculeInput:
    molecules:                  List[Mol]
    molecule_token_pool_lens:   List[List[int]]
    molecule_atom_indices:      List[List[int] | None]
    molecule_ids:               Int[' n']
    additional_molecule_feats:  Float['n 5']
    is_molecule_types:          Bool['n 4']
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    templates:                  Float['t n n dt']
    template_mask:              Bool[' t'] | None = None
    msa:                        Float['s n dm']
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None

@typecheck
def molecule_to_atom_input(molecule_input: MoleculeInput) -> AtomInput:
    raise NotImplementedError

# alphafold3 input - support polypeptides, nucleic acids, metal ions + any number of ligands + misc biomolecules

@typecheck
@dataclass
class Alphafold3Input:
    proteins:                   List[Int[' _']]
    protein_atom_lens:          List[Int[' _']]
    nucleic_acids:              List[Int[' _']]
    nucleic_acid_atom_lens:     List[Int[' _']]
    metal_ions:                 Int[' _']
    misc_molecule_ids:          Int[' _']
    ligands:                    List[Mol | str] # can be given as smiles
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None

@typecheck
def af3_input_to_molecule_input(af3_input: Alphafold3Input) -> AtomInput:
    raise NotImplementedError

# pdb input

@typecheck
@dataclass
class PDBInput:
    filepath: str

@typecheck
def pdb_input_to_alphafold3_input(pdb_input: PDBInput) -> Alphafold3Input:
    raise NotImplementedError

# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    AtomInput: identity,
    MoleculeInput: molecule_to_atom_input,
    Alphafold3Input: compose(
        af3_input_to_molecule_input,
        molecule_to_atom_input
    ),
    PDBInput: compose(
        pdb_input_to_alphafold3_input,
        af3_input_to_molecule_input,
        molecule_to_atom_input
    )
}

# function for extending the config

@typecheck
def register_input_transform(
    input_type: Type,
    fn: Callable[[Any], AtomInput]
):
    assert input_type not in INPUT_TO_ATOM_TRANSFORM, f'{input_type} is already registered'
    INPUT_TO_ATOM_TRANSFORM[input_type] = fn

# functions for transforming to atom inputs

def maybe_transform_to_atom_inputs(inputs: List[Any]) -> List[AtomInput]:
    atom_inputs = []

    for i in inputs:

        maybe_to_atom_fn = INPUT_TO_ATOM_TRANSFORM.get(type(i), None)

        if not exists(maybe_to_atom_fn):
            raise TypeError(f'invalid input type {type(i)} being passed into Trainer that is not converted to AtomInput correctly')

        atom_inputs.append(maybe_to_atom_fn(i))

    return atom_inputs
