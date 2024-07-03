from __future__ import annotations

from functools import wraps, partial
from dataclasses import dataclass, asdict, field
from typing import Type, Literal, Callable, List, Any, Tuple

import torch
from torch import tensor
import torch.nn.functional as F
import einx

from rdkit.Chem import AllChem as Chem
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
    ATOMS,
    ATOM_BONDS,
    MISC,
    mol_from_smile,
    remove_atom_from_mol,
    reverse_complement,
    reverse_complement_tensor
)

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def flatten(arr):
    return [el for sub_arr in arr for el in sub_arr]

def pad_to_len(t, length, value = 0):
    return F.pad(t, (0, max(0, length - t.shape[-1])), value = value)

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
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    token_bonds:                Bool['n n'] | None = None
    atom_ids:                   Int[' m'] | None = None
    atom_parent_ids:            Int[' m'] | None = None
    atompair_ids:               Int['m m'] | Int['nw w (w*2)'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    atom_pos:                   Float['m 3'] | None = None
    output_atompos_indices:     Int[' m'] | None = None
    molecule_atom_indices:      Int[' n'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int['n n'] | None = None
    plddt_labels:               Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None

    def dict(self):
        return asdict(self)

@typecheck
@dataclass
class BatchedAtomInput:
    atom_inputs:                Float['b m dai']
    molecule_ids:               Int['b n']
    molecule_atom_lens:         Int['b n']
    atompair_inputs:            Float['b m m dapi'] | Float['b nw w (w*2) dapi']
    additional_molecule_feats:  Int[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'b n {IS_MOLECULE_TYPES}']
    additional_token_feats:     Float[f'b n dtf'] | None = None
    templates:                  Float['b t n n dt'] | None = None
    msa:                        Float['b s n dm'] | None = None
    token_bonds:                Bool['b n n'] | None = None
    atom_ids:                   Int['b m'] | None = None
    atom_parent_ids:            Int['b m'] | None = None
    atompair_ids:               Int['b m m'] | Int['b nw w (w*2)'] | None = None
    template_mask:              Bool['b t'] | None = None
    msa_mask:                   Bool['b s'] | None = None
    atom_pos:                   Float['b m 3'] | None = None
    output_atompos_indices:     Int['b m'] | None = None
    molecule_atom_indices:      Int['b n'] | None = None
    distance_labels:            Int['b n n'] | None = None
    pae_labels:                 Int['b n n'] | None = None
    pde_labels:                 Int['b n n'] | None = None
    plddt_labels:               Int['b n'] | None = None
    resolved_labels:            Int['b n'] | None = None

    def dict(self):
        return asdict(self)

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens

def default_extract_atom_feats_fn(atom):
    return [
        atom.GetFormalCharge(),
        atom.GetImplicitValence(),
        atom.GetExplicitValence()
    ]

@typecheck
@dataclass
class MoleculeInput:
    molecules:                  List[Mol]
    molecule_token_pool_lens:   List[int]
    molecule_atom_indices:      List[int | None]
    molecule_ids:               Int[' n']
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    token_bonds:                Bool['n n']
    atom_parent_ids:            Int[' m'] | None = None
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    output_atompos_indices:     Int[' m'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None
    add_atom_ids:               bool = False
    add_atompair_ids:           bool = False
    extract_atom_feats_fn:      Callable[[Any], List[float]] = default_extract_atom_feats_fn

@typecheck
def molecule_to_atom_input(
    mol_input: MoleculeInput
) -> AtomInput:

    i = mol_input

    molecules = i.molecules
    atom_lens = i.molecule_token_pool_lens
    extract_atom_feats_fn = i.extract_atom_feats_fn

    # get total number of atoms

    if not exists(atom_lens):
        atom_lens = []

        for mol, is_ligand in zip(molecules, i.is_molecule_types[:, -1]):
            num_atoms = mol.GetNumAtoms()

            if is_ligand:
                atom_lens.extend([1] * num_atoms)
            else:
                atom_lens.append(num_atoms)

    atom_lens = tensor(atom_lens)
    total_atoms = atom_lens.sum().item()

    # molecule_atom_lens

    atoms = []

    for mol in molecules:
        atoms.extend([*mol.GetAtoms()])

    # handle maybe atom embeds

    atom_ids = None

    if i.add_atom_ids:
        atom_index = {symbol: i for i, symbol in enumerate(ATOMS)}

        atom_ids = []

        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            assert atom_symbol in atom_index, f'{atom_symbol} not found in the ATOMS defined in life.py'

            atom_ids.append(atom_index[atom_symbol])

        atom_ids = tensor(atom_ids, dtype = torch.long)

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atom_bond_index = {symbol: (idx + 1) for idx, symbol in enumerate(ATOM_BONDS)}
        other_index = len(ATOM_BONDS) + 1

        atompair_ids = torch.zeros(total_atoms, total_atoms).long()

        offset = 0

        # need the asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna` for is_molecule_types (chainable biomolecules)
        # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `asym_ids`

        asym_ids = i.additional_molecule_feats[..., 2]
        asym_ids = F.pad(asym_ids, (1, 0), value = -1)
        is_first_mol_in_chains = (asym_ids[1:] - asym_ids[:-1]) == 1

        is_chainable_biomolecules = i.is_molecule_types[..., :3].any(dim = -1)

        # for every molecule, build the bonds id matrix and add to `atompair_ids`

        for idx, (mol, is_first_mol_in_chain, is_chainable_biomolecule) in enumerate(zip(molecules, is_first_mol_in_chains, is_chainable_biomolecules)):

            coordinates = []
            updates = []

            num_atoms = mol.GetNumAtoms()
            mol_atompair_ids = torch.zeros(num_atoms, num_atoms).long()

            for bond in mol.GetBonds():
                atom_start_index = bond.GetBeginAtomIdx()
                atom_end_index = bond.GetEndAtomIdx()

                coordinates.extend([
                    [atom_start_index, atom_end_index],
                    [atom_end_index, atom_start_index]
                ])

                bond_type = bond.GetBondType()
                bond_id = atom_bond_index.get(bond_type, other_index) + 1

                updates.extend([bond_id, bond_id])

            coordinates = tensor(coordinates).long()
            updates = tensor(updates).long()

            mol_atompair_ids = einx.set_at('[h w], c [2], c -> [h w]', mol_atompair_ids, coordinates, updates)

            row_col_slice = slice(offset, offset + num_atoms)
            atompair_ids[row_col_slice, row_col_slice] = mol_atompair_ids

            # if is chainable biomolecule
            # and not the first biomolecule in the chain, add a single covalent bond between first atom of incoming biomolecule and the last atom  of the last biomolecule

            if is_chainable_biomolecule and not is_first_mol_in_chain:
                atompair_ids[offset, offset - 1] = 1
                atompair_ids[offset - 1, offset] = 1

            offset += num_atoms        

    # atom_inputs

    atom_inputs = []

    for mol in molecules:
        atoms = mol.GetAtoms()
        atom_feats = []

        for atom in atoms:
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.extend(atom_feats)

    # atompair_inputs

    atompair_inputs = torch.zeros((total_atoms, total_atoms, 1))

    offset = 0

    for mol in molecules:

        all_atom_pos = []

        for idx, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(idx)
            all_atom_pos.append([pos.x, pos.y, pos.z])

        all_atom_pos_tensor = tensor(all_atom_pos)

        dist_matrix = torch.cdist(all_atom_pos_tensor, all_atom_pos_tensor)

        num_atoms = mol.GetNumAtoms()

        row_col_slice = slice(offset, offset + num_atoms)
        atompair_inputs[row_col_slice, row_col_slice, 0] = dist_matrix

        offset += num_atoms

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim = -2)

    atom_input = AtomInput(
        atom_inputs = tensor(atom_inputs, dtype = torch.float),
        atompair_inputs = atompair_inputs,
        molecule_atom_lens = tensor(atom_lens, dtype = torch.long),
        molecule_ids = i.molecule_ids,
        additional_token_feats = i.additional_token_feats,
        additional_molecule_feats = i.additional_molecule_feats,
        is_molecule_types = i.is_molecule_types,
        atom_pos = atom_pos,
        token_bonds = i.token_bonds,
        atom_parent_ids = i.atom_parent_ids,
        atom_ids = atom_ids,
        atompair_ids = atompair_ids
    )

    return atom_input

# alphafold3 input - support polypeptides, nucleic acids, metal ions + any number of ligands + misc biomolecules

imm_list = partial(field, default_factory = list)

@typecheck
@dataclass
class Alphafold3Input:
    proteins:                   List[Int[' _'] | str] = imm_list()
    ss_dna:                     List[Int[' _'] | str] = imm_list()
    ss_rna:                     List[Int[' _'] | str] = imm_list()
    metal_ions:                 Int[' _'] | List[str] = imm_list()
    misc_molecule_ids:          Int[' _'] | List[str] = imm_list()
    ligands:                    List[Mol | str] = imm_list() # can be given as smiles
    ds_dna:                     List[Int[' _'] | str] = imm_list()
    ds_rna:                     List[Int[' _'] | str] = imm_list()
    atom_parent_ids:            Int[' m'] | None = None
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None
    add_atom_ids:               bool = False
    add_atompair_ids:           bool = False
    add_output_atompos_indices: bool = True
    extract_atom_feats_fn:      Callable[[Any], List[float]] = default_extract_atom_feats_fn

@typecheck
def map_int_or_string_indices_to_mol(
    entries: dict,
    indices: Int[' _'] | List[str] | str,
    mol_keyname = 'rdchem_mol',
    chain = False,
    return_entries = False
) -> List[Mol] | Tuple[List[Mol], List[dict]]:

    if isinstance(indices, str):
        indices = list(indices)

    entries_list = list(entries.values())

    # first get all the peptide or nucleotide entries

    if torch.is_tensor(indices):
        indices = indices.tolist()
        entries = [entries_list[i] for i in indices]
    else:
        entries = [entries[s] for s in indices]

    # for all peptides or nucleotide except last, remove hydroxl

    mols = []

    for idx, entry in enumerate(entries):
        is_last = idx == (len(entries) - 1)

        mol = entry[mol_keyname]

        if chain and not is_last:
            # hydroxyl oxygen to be removed should be the last atom
            hydroxyl_idx = mol.GetNumAtoms() - 1
            mol = remove_atom_from_mol(mol, hydroxyl_idx)

        mols.append(mol)

    if not return_entries:
        return mols

    return mols, entries

@typecheck
def maybe_string_to_int(
    entries: dict,
    indices: Int[' _'] | List[str] | str,
    other_index: int = 0
) -> Int[' _']:
    if isinstance(indices, str):
        indices = list(indices)

    if torch.is_tensor(indices):
        return indices

    index = {symbol: i for i, symbol in enumerate(entries.keys())}

    return tensor([index[c] for c in indices]).long()

@typecheck
def alphafold3_input_to_molecule_input(
    alphafold3_input: Alphafold3Input
) -> MoleculeInput:

    i = alphafold3_input

    chainable_biomol_entries: List[List[dict]] = []  # for reordering the atom positions at the end

    ss_rnas = list(i.ss_rna)
    ss_dnas = list(i.ss_dna)

    # any double stranded nucleic acids is added to single stranded lists with its reverse complement
    # rc stands for reverse complement

    for seq in i.ds_rna:
        rc_fn = partial(reverse_complement, nucleic_acid_type = 'rna') if isinstance(seq, str) else reverse_complement_tensor
        rc_seq = rc_fn(seq)
        ss_rnas.extend([seq, rc_seq])

    for seq in i.ds_dna:
        rc_fn = partial(reverse_complement, nucleic_acid_type = 'dna') if isinstance(seq, str) else reverse_complement_tensor
        rc_seq = rc_fn(seq)
        ss_dnas.extend([seq, rc_seq])

    # keep track of molecule_ids - for now it is
    # other(1) | proteins (20) | rna (4) | dna (4)

    protein_offset = 1
    rna_offset = len(HUMAN_AMINO_ACIDS) + protein_offset
    dna_offset = len(RNA_NUCLEOTIDES) + rna_offset

    molecule_ids = []

    # convert all proteins to a List[Mol] of each peptide

    proteins = i.proteins
    mol_proteins = []
    protein_entries = []
    molecule_atom_indices = []

    for protein in proteins:
        mol_peptides, protein_entries = map_int_or_string_indices_to_mol(HUMAN_AMINO_ACIDS, protein, chain = True, return_entries = True)
        mol_proteins.append(mol_peptides)

        molecule_atom_indices.extend([entry['distogram_atom_idx'] for entry in protein_entries])

        protein_ids = maybe_string_to_int(HUMAN_AMINO_ACIDS, protein) + protein_offset
        molecule_ids.append(protein_ids)

        chainable_biomol_entries.append(protein_entries)

    # convert all single stranded nucleic acids to mol

    mol_ss_dnas = []
    mol_ss_rnas = []

    for seq in ss_rnas:
        mol_seq, ss_rna_entries = map_int_or_string_indices_to_mol(RNA_NUCLEOTIDES, seq, chain = True, return_entries = True)
        mol_ss_rnas.append(mol_seq)

        rna_ids = maybe_string_to_int(RNA_NUCLEOTIDES, seq) + rna_offset
        molecule_ids.append(rna_ids)

        chainable_biomol_entries.append(ss_rna_entries)

    for seq in ss_dnas:
        mol_seq, ss_dna_entries = map_int_or_string_indices_to_mol(DNA_NUCLEOTIDES, seq, chain = True, return_entries = True)
        mol_ss_dnas.append(mol_seq)

        dna_ids = maybe_string_to_int(DNA_NUCLEOTIDES, seq) + dna_offset
        molecule_ids.append(dna_ids)

        chainable_biomol_entries.append(ss_dna_entries)

    # convert metal ions to rdchem.Mol

    metal_ions = alphafold3_input.metal_ions
    mol_metal_ions = map_int_or_string_indices_to_mol(METALS, metal_ions)

    # convert ligands to rdchem.Mol

    ligands = list(alphafold3_input.ligands)
    mol_ligands = [(mol_from_smile(ligand) if isinstance(ligand, str) else ligand) for ligand in ligands]

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

    assert num_tokens > 0, f'you have an empty alphafold3 input'

    arange = torch.arange(num_tokens)[:, None]

    molecule_types_lens_cumsum = tensor([0, *molecule_type_token_lens]).cumsum(dim = -1)
    left, right = molecule_types_lens_cumsum[:-1], molecule_types_lens_cumsum[1:]

    is_molecule_types = (arange >= left) & (arange < right)

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

    total_atoms = sum(token_pool_lens)

    # construct the token bonds

    # will be linearly connected for proteins and nucleic acids
    # but for ligands, will have their atomic bond matrix (as ligands are atom resolution)

    token_bonds = torch.zeros(num_tokens, num_tokens).bool()

    offset = 0

    for biomolecule in (*mol_proteins, *mol_ss_rnas, *mol_ss_dnas):
        chain_len = len(biomolecule)
        eye = torch.eye(chain_len)

        row_col_slice = slice(offset, offset + chain_len - 1)
        token_bonds[row_col_slice, row_col_slice] = (eye[1:, :-1] + eye[:-1, 1:]) > 0
        offset += chain_len

    for ligand in mol_ligands:
        coordinates = []
        updates = []

        num_atoms = ligand.GetNumAtoms()
        has_bond = torch.zeros(num_atoms, num_atoms).bool()

        for bond in ligand.GetBonds():
            atom_start_index = bond.GetBeginAtomIdx()
            atom_end_index = bond.GetEndAtomIdx()

            coordinates.extend([
                [atom_start_index, atom_end_index],
                [atom_end_index, atom_start_index]
            ])

            updates.extend([True, True])

        coordinates = tensor(coordinates).long()
        updates = tensor(updates).bool()

        has_bond = einx.set_at('[h w], c [2], c -> [h w]', has_bond, coordinates, updates)

        row_col_slice = slice(offset, offset + num_atoms)
        token_bonds[row_col_slice, row_col_slice] = has_bond

        offset += num_atoms

    # handle molecule ids

    molecule_ids = torch.cat(molecule_ids)
    molecule_ids = pad_to_len(molecule_ids, num_tokens)

    # handle atom_parent_ids
    # this governs in the atom encoder / decoder, which atom attends to which
    # a design choice is taken so metal ions attend to each other, in case there are more than one

    @typecheck
    def get_num_atoms_per_chain(chains: List[List[Mol]]) -> List[int]:
        atoms_per_chain = []

        for chain in chains:
            num_atoms = 0
            for mol in chain:
                num_atoms += mol.GetNumAtoms()
            atoms_per_chain.append(num_atoms)

        return atoms_per_chain

    num_protein_atoms = get_num_atoms_per_chain(mol_proteins)
    num_ss_rna_atoms = get_num_atoms_per_chain(mol_ss_rnas)
    num_ss_dna_atoms = get_num_atoms_per_chain(mol_ss_dnas)
    num_ligand_atoms = [ligand.GetNumAtoms() for ligand in mol_ligands]

    atom_counts = [*num_protein_atoms, *num_ss_rna_atoms, *num_ss_dna_atoms, *num_ligand_atoms, num_metal_ions]

    atom_parent_ids = torch.repeat_interleave(
        torch.arange(len(atom_counts)),
        tensor(atom_counts)
    )

    # constructing the additional_molecule_feats
    # which is in turn used to derive relative positions
    # (todo) offer a way to precompute relative positions at data prep

    # residue_index - reuse molecular_ids here
    # token_index   - just an arange
    # asym_id       - unique id for each chain of a biomolecule type
    # entity_id     - unique id for each biomolecule - multimeric protein, ds dna
    # sym_id        - unique id for each chain within each biomolecule

    num_protein_tokens = [len(protein) for protein in proteins]
    num_ss_rna_tokens = [len(rna) for rna in ss_rnas]
    num_ss_dna_tokens = [len(dna) for dna in ss_dnas]
    num_ligand_tokens = [ligand.GetNumAtoms() for ligand in mol_ligands]

    token_repeats = tensor([*num_protein_tokens, *num_ss_rna_tokens, *num_ss_dna_tokens, *num_ligand_tokens, num_metal_ions])

    asym_ids = torch.repeat_interleave(
        torch.arange(len(token_repeats)),
        token_repeats
    )

    # entity ids

    unrepeated_entity_ids = tensor([
        0,
        *[*range(len(i.ss_rna))],
        *[*range(len(i.ds_rna))],
        *[*range(len(i.ss_dna))],
        *[*range(len(i.ds_dna))],
        *([1] * len(mol_ligands)),
        1
    ]).cumsum(dim = -1)

    entity_id_counts = [
        sum(num_protein_tokens),
        *[len(rna) for rna in i.ss_rna],
        *[len(rna) * 2 for rna in i.ds_rna],
        *[len(dna) for dna in i.ss_dna],
        *[len(dna) * 2 for dna in i.ds_dna],
        *num_ligand_tokens,
        num_metal_ions
    ]

    entity_ids = torch.repeat_interleave(unrepeated_entity_ids, tensor(entity_id_counts))

    # sym_id

    unrepeated_sym_ids = [
        *[*range(len(i.proteins))],
        *[*range(len(i.ss_rna))],
        *[i for rna in i.ds_rna for i in range(2)],
        *[*range(len(i.ss_dna))],
        *[i for dna in i.ds_dna for i in range(2)],
        *([0] * len(mol_ligands)),
        0
    ]

    sym_id_counts = [
        *num_protein_tokens,
        *[len(rna) for rna in i.ss_rna],
        *flatten([((len(rna),) * 2) for rna in i.ds_rna]),
        *[len(dna) for dna in i.ss_dna],
        *flatten([((len(dna),) * 2) for dna in i.ds_dna]),
        *num_ligand_tokens,
        num_metal_ions
    ]

    sym_ids = torch.repeat_interleave(tensor(unrepeated_sym_ids), tensor(sym_id_counts))

    # concat for all of additional_molecule_feats

    additional_molecule_feats = torch.stack((
        molecule_ids,
        torch.arange(num_tokens),
        asym_ids,
        entity_ids,
        sym_ids
    ), dim = -1)

    # molecule atom indices

    molecule_atom_indices = tensor(molecule_atom_indices)
    molecule_atom_indices = pad_to_len(molecule_atom_indices, num_tokens, value = -1)

    # handle atom positions

    atom_pos = i.atom_pos
    output_atompos_indices = None

    if exists(atom_pos):
        if isinstance(atom_pos, list):
            atom_pos = torch.cat(atom_pos, dim = -2)

        assert atom_pos.shape[-2] == total_atoms

        # to automatically reorder the atom positions back to canonical

        if i.add_output_atompos_indices:
            offset = 0
            output_atompos_indices = []

            for chain in chainable_biomol_entries:
                for idx, entry in enumerate(chain):
                    is_last = idx == (len(chain) - 1)

                    mol = entry['rdchem_mol']
                    num_atoms = mol.GetNumAtoms()
                    atom_reorder_indices = entry['atom_reorder_indices']

                    if not is_last:
                        num_atoms -= 1
                        atom_reorder_indices = atom_reorder_indices[:-1]

                    reorder_back_indices = atom_reorder_indices.argsort()
                    output_atompos_indices.append(reorder_back_indices + offset)

                    offset += num_atoms

            output_atompos_indices = torch.cat(output_atompos_indices, dim = -1)
            output_atompos_indices = F.pad(output_atompos_indices, (0, total_atoms - output_atompos_indices.shape[-1]), value = -1)

    # create molecule input

    molecule_input = MoleculeInput(
        molecules = molecules,
        molecule_token_pool_lens = token_pool_lens,
        molecule_atom_indices = molecule_atom_indices,
        molecule_ids = molecule_ids,
        token_bonds = token_bonds,
        additional_molecule_feats = additional_molecule_feats,
        additional_token_feats = default(i.additional_token_feats, torch.zeros(num_tokens, 2)),
        is_molecule_types = is_molecule_types,
        atom_pos = atom_pos,
        output_atompos_indices = output_atompos_indices,
        templates = i.templates,
        msa = i.msa,
        template_mask = i.template_mask,
        msa_mask = i.msa_mask,
        atom_parent_ids = atom_parent_ids,
        add_atom_ids = i.add_atom_ids,
        add_atompair_ids = i.add_atompair_ids,
        extract_atom_feats_fn = i.extract_atom_feats_fn
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
    if input_type in INPUT_TO_ATOM_TRANSFORM:
        print(f'{input_type} is already registered, but overwriting')

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
