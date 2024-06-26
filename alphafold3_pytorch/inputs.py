from functools import wraps
from typing import Type, TypedDict, Literal, Callable, List

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.tensor_typing import (
    typecheck,
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

# simple compose function
# for chaining from Alphafold3Input -> MoleculeInput -> AtomInput

def compose(*fns: Callable):
    def inner(x, *args, **kwargs):
        for fn in fns:
            x = fn(x, *args, **kwargs)
        return x
    return inner

# atom level, what Alphafold3 accepts

@typecheck
class AtomInput(TypedDict):
    atom_inputs:                Float['m dai']
    molecule_ids:               Int['n']
    molecule_atom_lens:         Int['n']
    atompair_inputs:            Float['m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Float[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    token_bonds:                Bool['n n'] | None
    atom_ids:                   Int['m'] | None
    atom_parent_ids:            Int['m'] | None
    atompair_ids:               Int['m m'] | Int['nw w (w*2)'] | None
    template_mask:              Bool['t'] | None
    msa_mask:                   Bool['s'] | None
    atom_pos:                   Float['m 3'] | None
    molecule_atom_indices:      Int['n'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int['n'] | None
    resolved_labels:            Int['n'] | None

@typecheck
class BatchedAtomInput(TypedDict):
    atom_inputs:                Float['b m dai']
    molecule_ids:               Int['b n']
    molecule_atom_lens:         Int['b n']
    atompair_inputs:            Float['b m m dapi'] | Float['b nw w (w*2) dapi']
    additional_molecule_feats:  Float[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'b n {IS_MOLECULE_TYPES}']
    templates:                  Float['b t n n dt']
    msa:                        Float['b s n dm']
    token_bonds:                Bool['b n n'] | None
    atom_ids:                   Int['b m'] | None
    atom_parent_ids:            Int['b m'] | None
    atompair_ids:               Int['b m m'] | Int['b nw w (w*2)'] | None
    template_mask:              Bool['b t'] | None
    msa_mask:                   Bool['b s'] | None
    atom_pos:                   Float['b m 3'] | None
    molecule_atom_indices:      Int['b n'] | None
    distance_labels:            Int['b n n'] | None
    pae_labels:                 Int['b n n'] | None
    pde_labels:                 Int['b n'] | None
    resolved_labels:            Int['b n'] | None

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens

@typecheck
class MoleculeInput(TypedDict):
    molecules:                  List[Mol]
    molecule_token_pool_lens:   List[List[int]]
    molecule_atom_indices:      List[List[int] | None]
    molecule_ids:               Int['n']
    additional_molecule_feats:  Float['n 5']
    is_molecule_types:          Bool['n 4']
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None
    templates:                  Float['t n n dt']
    template_mask:              Bool['t'] | None
    msa:                        Float['s n dm']
    msa_mask:                   Bool['s'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int['n'] | None
    resolved_labels:            Int['n'] | None

@typecheck
def molecule_to_atom_input(molecule_input: MoleculeInput) -> AtomInput:
    raise NotImplementedError

# alphafold3 input - support polypeptides, nucleic acids, metal ions + any number of ligands + misc biomolecules

@typecheck
class Alphafold3Input(TypedDict):
    proteins:                   List[Int['_']]
    protein_atom_lens:          List[Int['_']]
    nucleic_acids:              List[Int['_']]
    nucleic_acid_atom_lens:     List[Int['_']]
    metal_ions:                 List[int]
    misc_molecule_ids:          List[int]
    ligands:                    List[Mol | str] # can be given as smiles
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    template_mask:              Bool['t'] | None
    msa_mask:                   Bool['s'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int['n'] | None
    resolved_labels:            Int['n'] | None

@typecheck
def af3_input_to_molecule_input(af3_input: Alphafold3Input) -> AtomInput:
    raise NotImplementedError

# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    MoleculeInput: molecule_to_atom_input,
    Alphafold3Input: compose(af3_input_to_molecule_input, molecule_to_atom_input)
}

# function for extending the config

@typecheck
def register_input_transform(
    input_type: Type,
    fn: Callable[[TypedDict], AtomInput]
):
    INPUT_TO_ATOM_TRANSFORM[input_type] = fn
