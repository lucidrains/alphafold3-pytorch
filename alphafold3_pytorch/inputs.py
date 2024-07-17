from __future__ import annotations

import os
import json
from functools import wraps, partial
from dataclasses import dataclass, asdict, field
from typing import Type, Literal, Callable, List, Any, Tuple

import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import einx

from loguru import logger
from pdbeccdutils.core import ccd_reader

from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol

from alphafold3_pytorch.attention import (
    pad_to_length
)

from alphafold3_pytorch.common.biomolecule import (
    _from_mmcif_object,
    get_residue_constants,
)

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.data.data_pipeline import get_assembly

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

from alphafold3_pytorch.tensor_typing import (
    typecheck,
    beartype_isinstance,
    Int, Bool, Float
)

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

CCD_COMPONENTS_FILEPATH = os.path.join("data", "ccd_data", "components.cif")
CCD_COMPONENTS_SMILES_FILEPATH = os.path.join("data", "ccd_data", "components_smiles.json")

# load all SMILES strings in the PDB Chemical Component Dictionary (CCD)

CCD_COMPONENTS_SMILES = None

if os.path.exists(CCD_COMPONENTS_SMILES_FILEPATH):
    logger.info(f"Loading CCD component SMILES strings from {CCD_COMPONENTS_SMILES_FILEPATH}.")
    with open(CCD_COMPONENTS_SMILES_FILEPATH) as f:
        CCD_COMPONENTS_SMILES = json.load(f)
elif os.path.exists(CCD_COMPONENTS_FILEPATH):
    logger.info(
        f"Loading CCD components from {CCD_COMPONENTS_FILEPATH} to extract all available SMILES strings (~3 minutes, one-time only)."
    )
    CCD_COMPONENTS = ccd_reader.read_pdb_components_file(
        CCD_COMPONENTS_FILEPATH,
        sanitize=False,  # Reduce loading time
    )
    logger.info(
        f"Saving CCD component SMILES strings to {CCD_COMPONENTS_SMILES_FILEPATH} (one-time only)."
    )
    with open(CCD_COMPONENTS_SMILES_FILEPATH, "w") as f:
        CCD_COMPONENTS_SMILES = {
            ccd_code: Chem.MolToSmiles(CCD_COMPONENTS[ccd_code].component.mol)
            for ccd_code in CCD_COMPONENTS
        }
        json.dump(CCD_COMPONENTS_SMILES, f)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def identity(t):
    return t

def flatten(arr):
    return [el for sub_arr in arr for el in sub_arr]

def exclusive_cumsum(t):
    return t.cumsum(dim = -1) - t

def pad_to_len(t, length, value = 0, dim = -1):
    assert dim < 0
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, 0, max(0, length - t.shape[-1])), value = value)

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
    missing_atom_mask:          Bool[' m'] | None = None
    molecule_atom_indices:      Int[' n'] | None = None
    distogram_atom_indices:     Int[' n'] | None = None
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
    missing_atom_mask:          Bool['b m'] | None = None
    molecule_atom_indices:      Int['b n'] | None = None
    distogram_atom_indices:     Int['b n'] | None = None
    distance_labels:            Int['b n n'] | None = None
    pae_labels:                 Int['b n n'] | None = None
    pde_labels:                 Int['b n n'] | None = None
    plddt_labels:               Int['b n'] | None = None
    resolved_labels:            Int['b n'] | None = None

    def dict(self):
        return asdict(self)

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens

def default_extract_atom_feats_fn(atom: Atom):
    return tensor([
        atom.GetFormalCharge(),
        atom.GetImplicitValence(),
        atom.GetExplicitValence()
    ])

def default_extract_atompair_feats_fn(mol: Mol):
    all_atom_pos = []

    for idx, atom in enumerate(mol.GetAtoms()):
        pos = mol.GetConformer().GetAtomPosition(idx)
        all_atom_pos.append([pos.x, pos.y, pos.z])

    all_atom_pos_tensor = tensor(all_atom_pos)

    dist_matrix = torch.cdist(all_atom_pos_tensor, all_atom_pos_tensor)
    return torch.stack((dist_matrix,), dim = -1)

@typecheck
@dataclass
class MoleculeInput:
    molecules:                  List[Mol]
    molecule_token_pool_lens:   List[int]
    molecule_ids:               Int[' n']
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    src_tgt_atom_indices:       Int['n 2']
    token_bonds:                Bool['n n']
    molecule_atom_indices:      List[int | None] | None = None
    distogram_atom_indices:     List[int | None] | None = None
    missing_atom_indices:       List[Int[' _'] | None] | None = None
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
    directed_bonds:             bool = False
    extract_atom_feats_fn:      Callable[[Atom], Float['m dai']] = default_extract_atom_feats_fn
    extract_atompair_feats_fn:  Callable[[Mol], Float['m m dapi']] = default_extract_atompair_feats_fn

@typecheck
def molecule_to_atom_input(
    mol_input: MoleculeInput
) -> AtomInput:

    i = mol_input

    molecules = i.molecules
    atom_lens = i.molecule_token_pool_lens
    extract_atom_feats_fn = i.extract_atom_feats_fn
    extract_atompair_feats_fn = i.extract_atompair_feats_fn

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

    atoms: List[int] = []

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

    # get List[int] of number of atoms per molecule
    # for the offsets when building the atompair feature map / bonds

    all_num_atoms = tensor([mol.GetNumAtoms() for mol in molecules])
    offsets = exclusive_cumsum(all_num_atoms)

    # handle maybe missing atom indices

    missing_atom_mask = None
    missing_atom_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:

        assert len(molecules) == len(i.missing_atom_indices), f'{len(i.missing_atom_indices)} missing atom indices does not match the number of molecules given ({len(molecules)})'

        missing_atom_indices: List[Int[' _']] = [default(indices, torch.empty((0,), dtype = torch.long)) for indices in i.missing_atom_indices]

        missing_atom_mask: List[Bool[' _']] = []

        for num_atoms, mol_missing_atom_indices in zip(all_num_atoms, missing_atom_indices):

            mol_miss_atom_mask = torch.zeros(num_atoms, dtype = torch.bool)

            if mol_missing_atom_indices.numel() > 0:
                mol_miss_atom_mask.scatter_(-1, mol_missing_atom_indices, True)

            missing_atom_mask.append(mol_miss_atom_mask)

        missing_atom_mask = torch.cat(missing_atom_mask)

        missing_atom_indices = pad_sequence(missing_atom_indices, batch_first = True, padding_value = -1)

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atom_bond_index = {symbol: (idx + 1) for idx, symbol in enumerate(ATOM_BONDS)}
        num_atom_bond_types = len(atom_bond_index)

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

        prev_mol = None
        prev_src_tgt_atom_indices = None

        for idx, (mol, is_first_mol_in_chain, is_chainable_biomolecule, src_tgt_atom_indices, offset) in enumerate(zip(molecules, is_first_mol_in_chains, is_chainable_biomolecules, i.src_tgt_atom_indices, offsets)):

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

                # default to symmetric bond type (undirected atom bonds)

                bond_to = bond_from = bond_id

                # if allowing for directed bonds, assume num_atompair_embeds = (2 * num_atom_bond_types) + 1
                # offset other edge by num_atom_bond_types

                if i.directed_bonds:
                    bond_from += num_atom_bond_types

                updates.extend([bond_to, bond_from])

            coordinates = tensor(coordinates).long()
            updates = tensor(updates).long()

            mol_atompair_ids = einx.set_at('[h w], c [2], c -> [h w]', mol_atompair_ids, coordinates, updates)

            row_col_slice = slice(offset, offset + num_atoms)
            atompair_ids[row_col_slice, row_col_slice] = mol_atompair_ids

            # if is chainable biomolecule
            # and not the first biomolecule in the chain, add a single covalent bond between first atom of incoming biomolecule and the last atom of the last biomolecule

            if is_chainable_biomolecule and not is_first_mol_in_chain:

                _, last_atom_index = prev_src_tgt_atom_indices
                first_atom_index, _ = src_tgt_atom_indices

                last_atom_index_from_end = prev_mol.GetNumAtoms() - last_atom_index

                src_atom_offset = offset - last_atom_index_from_end
                tgt_atom_offset = offset + first_atom_index

                atompair_ids[src_atom_offset, tgt_atom_offset] = 1
                atompair_ids[tgt_atom_offset, src_atom_offset] = 1

            prev_mol = mol
            prev_src_tgt_atom_indices = src_tgt_atom_indices

    # atom_inputs

    atom_inputs: List[Float['m dai']] = []

    for mol in molecules:
        atom_feats = []

        for atom in mol.GetAtoms():
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.append(torch.stack(atom_feats, dim = 0))

    atom_inputs_tensor = torch.cat(atom_inputs).float()

    # atompair_inputs

    atompair_feats: List[Float['m m dapi']] = []

    for mol, offset in zip(molecules, offsets):
        atompair_feats.append(extract_atompair_feats_fn(mol))

    assert len(atompair_feats) > 0

    dim_atompair_inputs = first(atompair_feats).shape[-1]

    atompair_inputs = torch.zeros((total_atoms, total_atoms, dim_atompair_inputs))

    for atompair_feat, num_atoms, offset in zip(atompair_feats, all_num_atoms, offsets):
        row_col_slice = slice(offset, offset + num_atoms)
        atompair_inputs[row_col_slice, row_col_slice] = atompair_feat

    # mask out molecule atom indices and distogram atom indices where it is in the missing atom indices list

    molecule_atom_indices = i.molecule_atom_indices
    distogram_atom_indices = i.distogram_atom_indices

    if exists(missing_atom_indices):
        is_missing_molecule_atom = einx.equal('n missing, n -> n missing', missing_atom_indices, molecule_atom_indices).any(dim = -1)
        is_missing_distogram_atom = einx.equal('n missing, n -> n missing', missing_atom_indices, distogram_atom_indices).any(dim = -1)

        molecule_atom_indices = molecule_atom_indices.masked_fill(is_missing_molecule_atom, -1)
        distogram_atom_indices = distogram_atom_indices.masked_fill(is_missing_distogram_atom, -1)

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim = -2)

    # atom input

    atom_input = AtomInput(
        atom_inputs = atom_inputs_tensor,
        atompair_inputs = atompair_inputs,
        molecule_atom_lens = tensor(atom_lens, dtype = torch.long),
        molecule_ids = i.molecule_ids,
        molecule_atom_indices = i.molecule_atom_indices,
        distogram_atom_indices = i.distogram_atom_indices,
        missing_atom_mask = missing_atom_mask,
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
    missing_atom_indices:       List[List[int] | None] = imm_list()
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dm'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    reorder_atom_pos:           bool = True
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    pae_labels:                 Int['n n'] | None = None
    pde_labels:                 Int[' n'] | None = None
    resolved_labels:            Int[' n'] | None = None
    add_atom_ids:               bool = False
    add_atompair_ids:           bool = False
    directed_bonds:             bool = False
    extract_atom_feats_fn:      Callable[[Atom], Float['m dai']] = default_extract_atom_feats_fn
    extract_atompair_feats_fn:  Callable[[Mol], Float['m m dapi']] = default_extract_atompair_feats_fn

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

    # gather Chem.Mol(s)

    mols = [entry[mol_keyname] for entry in entries]

    if not return_entries:
        return mols

    return mols, entries

@typecheck
def maybe_string_to_int(
    entries: dict,
    indices: Int[' _'] | List[str] | str,
) -> Int[' _']:

    unknown_index = len(entries) - 1

    if isinstance(indices, str):
        indices = list(indices)

    if torch.is_tensor(indices):
        return indices

    index = {symbol: i for i, symbol in enumerate(entries.keys())}

    return tensor([index.get(c, unknown_index) for c in indices]).long()

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
    # proteins (21) | rna (5) | dna (5) | gap? (1) - unknown for each biomolecule is the very last, ligand is 20

    rna_offset = len(HUMAN_AMINO_ACIDS)
    dna_offset = len(RNA_NUCLEOTIDES) + rna_offset

    ligand_id = len(HUMAN_AMINO_ACIDS) - 1
    gap_id = len(DNA_NUCLEOTIDES) + dna_offset

    molecule_ids = []

    # convert all proteins to a List[Mol] of each peptide

    proteins = i.proteins
    mol_proteins = []
    protein_entries = []

    distogram_atom_indices = []
    molecule_atom_indices = []
    src_tgt_atom_indices = []

    for protein in proteins:
        mol_peptides, protein_entries = map_int_or_string_indices_to_mol(HUMAN_AMINO_ACIDS, protein, chain = True, return_entries = True)
        mol_proteins.append(mol_peptides)

        distogram_atom_indices.extend([entry['token_center_atom_idx'] for entry in protein_entries])
        molecule_atom_indices.extend([entry['distogram_atom_idx'] for entry in protein_entries])

        src_tgt_atom_indices.extend([[entry['first_atom_idx'], entry['last_atom_idx']] for entry in protein_entries])

        protein_ids = maybe_string_to_int(HUMAN_AMINO_ACIDS, protein)
        molecule_ids.append(protein_ids)

        chainable_biomol_entries.append(protein_entries)

    # convert all single stranded nucleic acids to mol

    mol_ss_dnas = []
    mol_ss_rnas = []

    for seq in ss_rnas:
        mol_seq, ss_rna_entries = map_int_or_string_indices_to_mol(RNA_NUCLEOTIDES, seq, chain = True, return_entries = True)
        mol_ss_rnas.append(mol_seq)

        distogram_atom_indices.extend([entry['token_center_atom_idx'] for entry in ss_rna_entries])
        molecule_atom_indices.extend([entry['distogram_atom_idx'] for entry in ss_rna_entries])

        src_tgt_atom_indices.extend([[entry['first_atom_idx'], entry['last_atom_idx']] for entry in ss_rna_entries])

        rna_ids = maybe_string_to_int(RNA_NUCLEOTIDES, seq) + rna_offset
        molecule_ids.append(rna_ids)

        chainable_biomol_entries.append(ss_rna_entries)

    for seq in ss_dnas:
        mol_seq, ss_dna_entries = map_int_or_string_indices_to_mol(DNA_NUCLEOTIDES, seq, chain = True, return_entries = True)
        mol_ss_dnas.append(mol_seq)

        distogram_atom_indices.extend([entry['token_center_atom_idx'] for entry in ss_dna_entries])
        molecule_atom_indices.extend([entry['distogram_atom_idx'] for entry in ss_dna_entries])

        src_tgt_atom_indices.extend([[entry['first_atom_idx'], entry['last_atom_idx']] for entry in ss_dna_entries])

        dna_ids = maybe_string_to_int(DNA_NUCLEOTIDES, seq) + dna_offset
        molecule_ids.append(dna_ids)

        chainable_biomol_entries.append(ss_dna_entries)

    # convert metal ions to rdchem.Mol

    metal_ions = alphafold3_input.metal_ions
    mol_metal_ions = map_int_or_string_indices_to_mol(METALS, metal_ions)

    molecule_ids.append(tensor([gap_id] * len(mol_metal_ions)))

    # convert ligands to rdchem.Mol

    ligands = list(alphafold3_input.ligands)
    mol_ligands = [(mol_from_smile(ligand) if isinstance(ligand, str) else ligand) for ligand in ligands]

    molecule_ids.append(tensor([ligand_id] * len(mol_ligands)))

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

    molecule_ids = torch.cat(molecule_ids).long()
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
        *[i for _ in i.ds_rna for i in range(2)],
        *[*range(len(i.ss_dna))],
        *[i for _ in i.ds_dna for i in range(2)],
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

    # distogram and token centre atom indices

    distogram_atom_indices = tensor(distogram_atom_indices)
    distogram_atom_indices = pad_to_len(distogram_atom_indices, num_tokens, value = -1)

    molecule_atom_indices = tensor(molecule_atom_indices)
    molecule_atom_indices = pad_to_len(molecule_atom_indices, num_tokens, value = -1)

    src_tgt_atom_indices = tensor(src_tgt_atom_indices)
    src_tgt_atom_indices = pad_to_len(src_tgt_atom_indices, num_tokens, value = -1, dim = -2)

    # atom positions

    atom_pos = i.atom_pos

    # handle missing atom indices

    missing_atom_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        missing_atom_indices = []

        for mol_miss_atom_indices in i.missing_atom_indices:
            mol_miss_atom_indices = default(mol_miss_atom_indices, [])
            mol_miss_atom_indices = tensor(mol_miss_atom_indices, dtype = torch.long)

            missing_atom_indices.append(mol_miss_atom_indices)

        assert len(molecules) == len(missing_atom_indices)

    # create molecule input

    molecule_input = MoleculeInput(
        molecules = molecules,
        molecule_token_pool_lens = token_pool_lens,
        molecule_atom_indices = molecule_atom_indices,
        distogram_atom_indices = distogram_atom_indices,
        molecule_ids = molecule_ids,
        token_bonds = token_bonds,
        additional_molecule_feats = additional_molecule_feats,
        additional_token_feats = default(i.additional_token_feats, torch.zeros(num_tokens, 2)),
        is_molecule_types = is_molecule_types,
        missing_atom_indices = missing_atom_indices,
        src_tgt_atom_indices = src_tgt_atom_indices,
        atom_pos = atom_pos,
        templates = i.templates,
        msa = i.msa,
        template_mask = i.template_mask,
        msa_mask = i.msa_mask,
        atom_parent_ids = atom_parent_ids,
        add_atom_ids = i.add_atom_ids,
        add_atompair_ids = i.add_atompair_ids,
        directed_bonds = i.directed_bonds,
        extract_atom_feats_fn = i.extract_atom_feats_fn,
        extract_atompair_feats_fn = i.extract_atompair_feats_fn
    )

    return molecule_input

# pdb input

@typecheck
@dataclass
class PDBInput:
    filepath: str

@typecheck
def extract_chain_sequences_from_chemical_components(
    chem_comps: List[mmcif_parsing.ChemComp],
) -> Tuple[List[str], List[str], List[str], List[Mol | str]]:
    assert exists(CCD_COMPONENTS_SMILES), (
        f"The PDB Chemical Component Dictionary (CCD) components SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} does not exist. "
        f"Please re-run this script after ensuring the preliminary CCD file {CCD_COMPONENTS_FILEPATH} has been downloaded according to this project's `README.md` file."
        f"After doing so, the SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} will be cached locally and used for subsequent runs."
    )

    current_chain_seq = []
    proteins, ss_dna, ss_rna, ligands = [], [], [], []

    for idx, details in enumerate(chem_comps):
        residue_constants = get_residue_constants(details.type)
        restype = residue_constants.restype_3to1.get(details.id, "X")

        # Protein residues

        if "peptide" in details.type.lower():
            if not current_chain_seq:
                proteins.append(current_chain_seq)
            current_chain_seq.append(restype)
            # Reset current_chain_seq if the next residue is not a protein residue
            if idx + 1 < len(chem_comps) and "peptide" not in chem_comps[idx + 1].type.lower():
                current_chain_seq = []

        # DNA residues

        elif "dna" in details.type.lower():
            if not current_chain_seq:
                ss_dna.append(current_chain_seq)
            current_chain_seq.append(restype)
            # Reset current_chain_seq if the next residue is not a DNA residue
            if idx + 1 < len(chem_comps) and "dna" not in chem_comps[idx + 1].type.lower():
                current_chain_seq = []

        # RNA residues

        elif "rna" in details.type.lower():
            if not current_chain_seq:
                ss_rna.append(current_chain_seq)
            current_chain_seq.append(restype)
            # Reset current_chain_seq if the next residue is not a RNA residue
            if idx + 1 < len(chem_comps) and "rna" not in chem_comps[idx + 1].type.lower():
                current_chain_seq = []

        # Ligand SMILES strings

        else:
            if not current_chain_seq:
                ligands.append(current_chain_seq)
            current_chain_seq.append(CCD_COMPONENTS_SMILES[details.id])
            # Reset current_chain_seq after adding each ligand's SMILES string
            current_chain_seq = []

    # Efficiently build sequence strings

    proteins = ["".join(protein) for protein in proteins]
    ss_dna = ["".join(dna) for dna in ss_dna]
    ss_rna = ["".join(rna) for rna in ss_rna]
    ligands = ["".join(ligand) for ligand in ligands]

    return proteins, ss_dna, ss_rna, ligands

@typecheck
def pdb_input_to_alphafold3_input(pdb_input: PDBInput) -> Alphafold3Input:
    filepath = pdb_input.filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    assert os.path.exists(filepath), f"PDB input file `{filepath}` does not exist."

    mmcif_object = mmcif_parsing.parse_mmcif_object(
        filepath=filepath,
        file_id=file_id,
    )

    biomol = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in file_id
        else get_assembly(_from_mmcif_object(mmcif_object))
    )

    chem_comp_table = {comp.id: comp for comp in biomol.chem_comp_table}
    chem_comp_details = [chem_comp_table[chemid] for chemid in biomol.chemid]

    proteins, ss_dna, ss_rna, ligands = extract_chain_sequences_from_chemical_components(
        chem_comp_details
    )

    atom_positions = biomol.atom_positions[biomol.atom_mask.astype(bool)]
    alphafold_input = Alphafold3Input(
        proteins=proteins,
        ss_dna=ss_dna,
        ss_rna=ss_rna,
        ligands=ligands,
        atom_pos=torch.from_numpy(atom_positions.astype("float32")),
    )

    # TODO: Add support for AlphaFold 2-style amino/nucleic acid atom parametrization (i.e., 47 possible atom types per residue)

    # TODO: Reference bonds from `biomol` instead of instantiating them within `Alphafold3Input`

    # TODO: Ensure only polymer-ligand (e.g., protein/RNA/DNA-ligand) and ligand-ligand bonds
    # (and bonds less than 2.4 Å) are referenced in `Alphafold3Input` (AF3 Supplement - Table 5, `token_bonds`)

    return alphafold_input

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
