from functools import wraps, partial
from dataclasses import dataclass
from typing import Type, Literal, Callable, List, Any

import torch
import einx

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.tensor_typing import (
    typecheck,
    beartype_isinstance,
    Int, Bool, Float
)

from alphafold3_pytorch.life import (
    HUMAN_AMINO_ACIDS,
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    METALS,
    MISC,
    reverse_complement,
    reverse_complement_tensor
)

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

# functions

def exists(v):
    return v is not None

def identity(t):
    return t

def flatten(arr):
    return [el for sub_arr in arr for el in sub_arr]

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
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
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
    additional_molecule_feats:  Int[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'b n {IS_MOLECULE_TYPES}']
    templates:                  Float['b t n n dt'] | None = None
    msa:                        Float['b s n dm'] | None = None
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
    molecule_token_pool_lens:   List[int]
    molecule_atom_indices:      List[int | None]
    molecule_ids:               Int[' n']
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    template_mask:              Bool[' t'] | None = None
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
    proteins:                   List[Int[' _'] | str]
    ss_dna:                     List[Int[' _'] | str]
    ss_rna:                     List[Int[' _'] | str]
    metal_ions:                 Int[' _'] | List[str]
    misc_molecule_ids:          Int[' _'] | List[str]
    ligands:                    List[Mol | str] # can be given as smiles
    ds_dna:                     List[Int[' _'] | str]
    ds_rna:                     List[Int[' _'] | str]
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None

@typecheck
def map_int_or_string_indices_to_mol(
    entries: dict,
    indices: Int[' _'] | List[str] | str,
    mol_keyname = 'rdchem_mol'
) -> List[Mol]:

    if isinstance(indices, str):
        indices = list(indices)

    entries_list = list(entries.values())

    if torch.is_tensor(indices):
        indices = indices.tolist()
        mols = [entries_list[i][mol_keyname] for i in indices]
    else:
        mols = [entries[s][mol_keyname] for s in indices]

    return mols

@typecheck
def alphafold3_input_to_molecule_input(
    alphafold3_input: Alphafold3Input
) -> MoleculeInput:

    ss_dnas = list(alphafold3_input.ss_dna)
    ss_rnas = list(alphafold3_input.ss_rna)

    # any double stranded nucleic acids is added to single stranded lists with its reverse complement
    # rc stands for reverse complement

    for seq in alphafold3_input.ds_dna:
        rc_fn = partial(reverse_complement, nucleic_acid_type = 'dna') if isinstance(seq, str) else reverse_complement_tensor
        rc_seq = rc_fn(seq)
        ss_dnas.extend([seq, rc_seq])

    for seq in alphafold3_input.ds_rna:
        rc_fn = partial(reverse_complement, nucleic_acid_type = 'rna') if isinstance(seq, str) else reverse_complement_tensor
        rc_seq = rc_fn(seq)
        ss_rnas.extend([seq, rc_seq])

    # convert all proteins to a List[Mol] of each peptide

    proteins = alphafold3_input.proteins
    mol_proteins = []

    for protein in proteins:
        mol_peptides = map_int_or_string_indices_to_mol(HUMAN_AMINO_ACIDS, list(protein))
        mol_proteins.append(mol_peptides)

    # convert all single stranded nucleic acids to mol

    mol_ss_dnas = []
    mol_ss_rnas = []

    for seq in ss_dnas:
        mol_seq = map_int_or_string_indices_to_mol(DNA_NUCLEOTIDES, seq)
        mol_ss_dnas.append(mol_seq)

    for seq in ss_rnas:
        mol_seq = map_int_or_string_indices_to_mol(RNA_NUCLEOTIDES, seq)
        mol_ss_rnas.append(mol_seq)

    # convert metal ions to rdchem.Mol

    metal_ions = alphafold3_input.metal_ions
    mol_metal_ions = map_int_or_string_indices_to_mol(METALS, metal_ions)

    # convert ligands to rdchem.Mol

    ligands = list(alphafold3_input.ligands)
    mol_ligands = [(Chem.MolFromSmiles(l) if isinstance(l, str) else l) for l in ligands]

    # create the molecule input

    all_protein_mols = flatten(mol_proteins)
    all_rna_mols = flatten(mol_ss_rnas)
    all_dna_mols = flatten(mol_ss_dnas)

    molecules_without_ligands = [
        *all_protein_mols,
        *all_rna_mols,
        *all_dna_mols,
    ]

    molecule_token_pool_lens_without_ligands = [mol.GetNumAtoms() for mol in molecules_without_ligands]

    # metal ions pool lens

    num_metal_ions = len(mol_metal_ions)
    metal_ions_pool_lens = [1] * num_metal_ions

    # in the paper, they treat each atom of the ligands as a token

    ligands_token_pool_lens = [[1] * mol.GetNumAtoms() for mol in mol_ligands]

    total_ligand_tokens = sum([mol.GetNumAtoms() for mol in mol_ligands])

    # correctly generate the is_molecule_types, which is a boolean tensor of shape [*, 4]
    # is_protein | is_rna | is_dna | is_ligand
    # this is needed for their special diffusion loss

    molecule_type_token_lens = [
        len(all_protein_mols),
        len(all_rna_mols),
        len(all_dna_mols),
        total_ligand_tokens
    ]

    num_tokens = sum(molecule_type_token_lens) + num_metal_ions

    arange = torch.arange(num_tokens)

    is_molecule_types_lens_cumsum = torch.tensor([0, *molecule_type_token_lens]).cumsum(dim = -1)

    gt_equal_mask = einx.greater_equal(
        'n, is_type -> n is_type',
        arange, is_molecule_types_lens_cumsum[:-1]
    )

    lt_mask = einx.less(
        'n, is_type -> n is_type',
        arange, is_molecule_types_lens_cumsum[1:]
    )

    is_molecule_types = gt_equal_mask & lt_mask

    # all molecules, layout is
    # proteins | ss rna | ss dna | ligands | metal ions

    molecules = [
        *molecules_without_ligands,
        *mol_ligands,
        *mol_metal_ions
    ]

    token_pool_lens = [
        *molecule_token_pool_lens_without_ligands,
        *flatten(ligands_token_pool_lens),
        *metal_ions_pool_lens
    ]

    molecule_input = MoleculeInput(
        molecules = molecules,
        molecule_token_pool_lens = token_pool_lens,
        molecule_atom_indices = [0] * num_tokens,
        molecule_ids = torch.zeros(num_tokens).long(),
        additional_molecule_feats = torch.zeros(num_tokens, 5).long(),
        is_molecule_types = is_molecule_types
    )

    return molecule_input

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
        alphafold3_input_to_molecule_input,
        molecule_to_atom_input
    ),
    PDBInput: compose(
        pdb_input_to_alphafold3_input,
        alphafold3_input_to_molecule_input,
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

@typecheck
def maybe_transform_to_atom_input(i: Any) -> AtomInput:
    maybe_to_atom_fn = INPUT_TO_ATOM_TRANSFORM.get(type(i), None)

    if not exists(maybe_to_atom_fn):
        raise TypeError(f'invalid input type {type(i)} being passed into Trainer that is not converted to AtomInput correctly')

    return maybe_to_atom_fn(i)

@typecheck
def maybe_transform_to_atom_inputs(inputs: List[Any]) -> List[AtomInput]:
    return [maybe_transform_to_atom_input(i) for i in inputs]
