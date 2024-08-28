from __future__ import annotations

import copy
import glob
import json
import os
from pathlib import Path
from functools import partial, wraps
from itertools import groupby
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, Type

import einx
from einops import pack, rearrange

import numpy as np
from numpy.lib.format import open_memmap

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import tensor, repeat_interleave
from torch.nn.utils.rnn import pad_sequence

from loguru import logger
from joblib import Parallel, delayed

from pdbeccdutils.core import ccd_reader

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Geometry import Point3D

from alphafold3_pytorch.common import amino_acid_constants, dna_constants, rna_constants
from alphafold3_pytorch.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
)
from alphafold3_pytorch.data import mmcif_parsing, msa_parsing
from alphafold3_pytorch.data.data_pipeline import (
    FeatureDict,
    get_assembly,
    make_msa_features,
    make_msa_mask,
)
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

from alphafold3_pytorch.life import (
    ATOM_BONDS,
    ATOMS,
    DNA_NUCLEOTIDES,
    HUMAN_AMINO_ACIDS,
    METALS,
    RNA_NUCLEOTIDES,
    mol_from_smile,
    reverse_complement,
    reverse_complement_tensor,
)

from alphafold3_pytorch.tensor_typing import Bool, Float, Int, typecheck
from alphafold3_pytorch.utils.data_utils import (
    PDB_INPUT_RESIDUE_MOLECULE_TYPE,
    extract_mmcif_metadata_field,
    get_pdb_input_residue_molecule_type,
    get_sorted_tuple_indices,
    is_atomized_residue,
    is_polymer,
    make_one_hot,
)
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum
from alphafold3_pytorch.utils.utils import default, exists, first

# silence RDKit's warnings

RDLogger.DisableLog("rdApp.*")

# constants

IS_MOLECULE_TYPES = 5
IS_PROTEIN_INDEX = 0
IS_RNA_INDEX = 1
IS_DNA_INDEX = 2
IS_LIGAND_INDEX = -2
IS_METAL_ION_INDEX = -1
IS_BIOMOLECULE_INDICES = slice(0, 3)

IS_PROTEIN, IS_RNA, IS_DNA, IS_LIGAND, IS_METAL_ION = tuple(
    (IS_MOLECULE_TYPES + i if i < 0 else i)
    for i in [
        IS_PROTEIN_INDEX,
        IS_RNA_INDEX,
        IS_DNA_INDEX,
        IS_LIGAND_INDEX,
        IS_METAL_ION_INDEX,
    ]
)

MOLECULE_GAP_ID = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES)
MOLECULE_METAL_ION_ID = MOLECULE_GAP_ID + 1
NUM_MOLECULE_IDS = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 2

NUM_MSA_ONE_HOT = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 1

DEFAULT_NUM_MOLECULE_MODS = 4  # `mod_protein`, `mod_rna`, `mod_dna`, and `mod_unk`
ADDITIONAL_MOLECULE_FEATS = 5

CCD_COMPONENTS_FILEPATH = os.path.join('data', 'ccd_data', 'components.cif')
CCD_COMPONENTS_SMILES_FILEPATH = os.path.join('data', 'ccd_data', 'components_smiles.json')

# load all SMILES strings in the PDB Chemical Component Dictionary (CCD)

CCD_COMPONENTS_SMILES = None

if os.path.exists(CCD_COMPONENTS_SMILES_FILEPATH):
    print(f'Loading CCD component SMILES strings from {CCD_COMPONENTS_SMILES_FILEPATH}.')
    with open(CCD_COMPONENTS_SMILES_FILEPATH) as f:
        CCD_COMPONENTS_SMILES = json.load(f)
elif os.path.exists(CCD_COMPONENTS_FILEPATH):
    print(
        f'Loading CCD components from {CCD_COMPONENTS_FILEPATH} to extract all available SMILES strings (~3 minutes, one-time only).'
    )
    CCD_COMPONENTS = ccd_reader.read_pdb_components_file(
        CCD_COMPONENTS_FILEPATH,
        sanitize=False,  # Reduce loading time
    )
    print(
        f'Saving CCD component SMILES strings to {CCD_COMPONENTS_SMILES_FILEPATH} (one-time only).'
    )
    with open(CCD_COMPONENTS_SMILES_FILEPATH, 'w') as f:
        CCD_COMPONENTS_SMILES = {
            ccd_code: Chem.MolToSmiles(CCD_COMPONENTS[ccd_code].component.mol_no_h)
            for ccd_code in CCD_COMPONENTS
        }
        json.dump(CCD_COMPONENTS_SMILES, f)

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

def flatten(arr):
    return [el for sub_arr in arr for el in sub_arr]

def without_keys(d: dict, exclude: set):
    return {k: v for k, v in d.items() if k not in exclude}

def pad_to_len(t, length, value = 0, dim = -1):
    assert dim < 0
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, 0, max(0, length - t.shape[dim])), value = value)

def offset_only_positive(t, offset):
    is_positive = t >= 0
    t_offsetted = t + offset
    return torch.where(is_positive, t_offsetted, t)

@typecheck
def remove_consecutive_duplicate(
    t: Int['n ...'],
    remove_to_value = -1
) -> Int['n ...']:

    is_duplicate = t[1:] == t[:-1]

    if is_duplicate.ndim == 2:
        is_duplicate = is_duplicate.all(dim = -1)

    is_duplicate = F.pad(is_duplicate, (1, 0), value = False)
    return einx.where('n, n ..., -> n ... ', ~is_duplicate, t, remove_to_value)

def compose(*fns: Callable):
    # for chaining from Alphafold3Input -> MoleculeInput -> AtomInput

    def inner(x, *args, **kwargs):
        for fn in fns:
            x = fn(x, *args, **kwargs)
        return x
    return inner

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

# validation functions

def hard_validate_atom_indices_ascending(
    indices: Int['b n'] | Int['b n 3'],
    error_msg_field: str = 'indices'
):
    # will do a hard validate
    # asserting if any of the indices that are not -1 (missing) are identical or descending
    # this will cover 'distogram_atom_indices', 'molecule_atom_indices', and 'atom_indices_for_frame'

    if indices.ndim == 2:
        indices = rearrange(indices, '... -> ... 1')

    for batch_index, sample_indices in enumerate(indices):

        all_present = (sample_indices >= 0).all(dim = -1)
        present_indices = sample_indices[all_present]

        # relaxed assumption that if all -1 or only one molecule, it passes the test

        if present_indices.numel() <= 1:
            continue

        difference = einx.subtract('n i, n j -> n (i j)', present_indices[1:], present_indices[:-1])

        assert (difference >= 0).all(), f'detected invalid {error_msg_field} for in a batch: {present_indices}'

# functions for deriving the frames for ligands
# this follows the logic from Alphafold3 Supplementary section 4.3.2

@typecheck
def get_indices_three_closest_atom_pos(
    atom_pos: Float['... n d'],
    mask: Bool['... n'] | None = None
) -> Int['... 3']:

    prec_dims, device = atom_pos.shape[:-2], atom_pos.device
    num_atoms, has_batch = atom_pos.shape[-2], atom_pos.ndim == 3

    if not exists(mask) and num_atoms < 3:
        return atom_pos.new_full((*prec_dims, 3), -1).long()

    if not has_batch:
        atom_pos = rearrange(atom_pos, '... -> 1 ...')

        if exists(mask):
            mask = rearrange(mask, '... -> 1 ...')

    # figure out which set of atoms are less than 3 for masking out later

    if exists(mask):
        insufficient_atom_mask = mask.sum(dim = -1) < 3

    # get distances between all atoms

    atom_dist = torch.cdist(atom_pos, atom_pos)

    # mask out the distance to self

    eye = torch.eye(num_atoms, device = device, dtype = torch.bool)

    mask_value = 1e4
    atom_dist.masked_fill_(eye, mask_value)

    # take care of padding

    if exists(mask):
        pair_mask = einx.logical_and('... i, ... j -> ... i j', mask, mask)
        atom_dist.masked_fill_(~pair_mask, mask_value)

    # will use topk on the negative of the distance

    neg_distance, two_closest_atom_indices = (-atom_dist).topk(2, dim = -1)

    mean_neg_distance = neg_distance.mean(dim = -1)

    best_atom_pair_index = mean_neg_distance.argmax(dim = -1)

    best_two_atom_neighbors = einx.get_at('... [m] c, ... -> ... c', two_closest_atom_indices, best_atom_pair_index)

    # place the chosen atom at the center

    three_atom_indices, _ = pack((
        best_two_atom_neighbors[..., 0],
        best_atom_pair_index,
        best_two_atom_neighbors[..., 1],
    ), 'b *')

    # mask out

    if exists(mask):
        three_atom_indices = einx.where('..., ... three, -> ... three', ~insufficient_atom_mask, three_atom_indices, -1)

    if not has_batch:
        three_atom_indices = rearrange(three_atom_indices, '1 ... -> ...')

    return three_atom_indices

@typecheck
def get_angle_between_edges(
    edge1: Float['... 3'],
    edge2: Float['... 3']
) -> Float['...']:
    cos = torch.dot(l2norm(edge1), l2norm(edge2))
    return torch.acos(cos)

@typecheck
def get_frames_from_atom_pos(
    atom_pos: Float['... n d'],
    mask: Bool['... n'] | None = None,
    filter_colinear_pos: bool = False,
    is_colinear_angle_thres: float = 25. # they use 25 degrees as a way of filtering out invalid frames
) -> Int['... 3']:

    frames = get_indices_three_closest_atom_pos(atom_pos, mask = mask)

    if not filter_colinear_pos:
        return frames

    is_invalid = (frames == -1).any(dim = -1)

    # get the edges and derive angles

    three_atom_pos = einx.get_at('... [m] c, ... three -> ... three c', atom_pos, frames)

    left_pos, center_pos, right_pos = three_atom_pos.unbind(dim = -2)

    edges1, edges2 = (left_pos - center_pos), (right_pos - center_pos)

    angle = get_angle_between_edges(edges1, edges2)

    degree = torch.rad2deg(angle)

    is_colinear = (
        (degree.abs() < is_colinear_angle_thres) |
        ((180. - degree.abs()).abs() < is_colinear_angle_thres)
    )

    # set any three atoms that are colinear to -1 indices

    three_atom_indices = einx.where('..., ... three, -> ... three', ~(is_colinear | is_invalid), frames, -1)
    return three_atom_indices

# atom level, what Alphafold3 accepts

UNCOLLATABLE_ATOM_INPUT_FIELDS = {'filepath'}

ATOM_INPUT_EXCLUDE_MODEL_FIELDS = {
    'filepath',
    'chains'
}

ATOM_DEFAULT_PAD_VALUES = dict(
    molecule_atom_lens = 0,
    missing_atom_mask = True
)

@typecheck
@dataclass
class AtomInput:
    atom_inputs:                Float['m dai']
    molecule_ids:               Int[' n']
    molecule_atom_lens:         Int[' n']
    atompair_inputs:            Float['m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    is_molecule_mod:            Bool['n num_mods'] | None = None
    additional_msa_feats:       Float['s n dmf'] | None = None
    additional_token_feats:     Float['n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dmi'] | None = None
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
    atom_indices_for_frame:     Int['n 3'] | None = None
    distance_labels:            Int['n n'] | None = None
    resolved_labels:            Int[' m'] | None = None
    resolution:                 Float[''] | None = None
    chains:                     Int[' 2'] | None = None
    filepath:                   str | None = None

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
    is_molecule_mod:            Bool['b n num_mods'] | None = None
    additional_msa_feats:       Float['b s n dmf'] | None = None
    additional_token_feats:     Float['b n dtf'] | None = None
    templates:                  Float['b t n n dt'] | None = None
    msa:                        Float['b s n dmi'] | None = None
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
    atom_indices_for_frame:     Int['b n 3'] | None = None
    distance_labels:            Int['b n n'] | None = None
    resolved_labels:            Int['b m'] | None = None
    resolution:                 Float[' b'] | None = None
    chains:                     Int['b 2'] | None = None
    filepath:                   List[str] | None = None

    def dict(self):
        return asdict(self)

    def model_forward_dict(self):
        return without_keys(self.dict(), ATOM_INPUT_EXCLUDE_MODEL_FIELDS)

# functions for saving an AtomInput to disk or loading from disk to AtomInput

@typecheck
def atom_input_to_file(
    atom_input: AtomInput,
    path: str | Path,
    overwrite: bool = False
) -> Path:

    if isinstance(path, str):
        path = Path(path)

    if not overwrite:
        assert not path.exists()

    path.parents[0].mkdir(exist_ok = True, parents = True)

    torch.save(atom_input.dict(), str(path))
    return path

@typecheck
def file_to_atom_input(path: str | Path) -> AtomInput:
    if isinstance(path, str):
        path = Path(path)

    assert path.is_file()

    atom_input_dict = torch.load(str(path), weights_only = True)
    return AtomInput(**atom_input_dict)

@typecheck
def default_none_fields_atom_input(i: AtomInput) -> AtomInput:

    # if templates given but template mask isn't given, default to all True

    if exists(i.templates) and not exists(i.template_mask):
        i.template_mask = torch.ones(i.templates.shape[0], dtype = torch.bool)

    # if msa given but msa mask isn't given default to all True

    if exists(i.msa) and not exists(i.msa_mask):
        i.msa_mask = torch.ones(i.msa.shape[0], dtype = torch.bool)

    # default missing atom mask should be all False

    if not exists(i.missing_atom_mask):
        i.missing_atom_mask = torch.zeros(i.atom_inputs.shape[0], dtype = torch.bool)

    return i

@typecheck
def pdb_dataset_to_atom_inputs(
    pdb_dataset: PDBDataset,
    *,
    output_atom_folder: str | Path | None = None,
    indices: Iterable | None = None,
    return_atom_dataset = False,
    n_jobs: int = 8,
    parallel_kwargs: dict = dict(),
    overwrite_existing: bool = False,
) -> Path | AtomDataset:

    if not exists(output_atom_folder):
        pdb_folder = Path(pdb_dataset.folder).resolve()
        parent_folder = pdb_folder.parents[0]
        output_atom_folder = parent_folder / f'{pdb_folder.stem}.atom-inputs'

    if isinstance(output_atom_folder, str):
        output_atom_folder = Path(output_atom_folder)

    if not exists(indices):
        indices = torch.randperm(len(pdb_dataset)).tolist()

    to_atom_input_fn = compose(
        pdb_input_to_molecule_input,
        molecule_to_atom_input
    )

    def should_process_pdb_input(index):
        atom_input_path = output_atom_folder / f'{index}.pt'
        return not atom_input_path.exists() or overwrite_existing

    @delayed
    def pdb_input_to_atom_file(index, path):
        pdb_input = pdb_dataset[index]

        atom_input = to_atom_input_fn(pdb_input)

        atom_input_path = path / f'{index}.pt'
        atom_input_to_file(atom_input, atom_input_path)

    Parallel(n_jobs = n_jobs, **parallel_kwargs)(pdb_input_to_atom_file(index, output_atom_folder) for index in filter(should_process_pdb_input, indices))

    if not return_atom_dataset:
        return output_atom_folder

    return AtomDataset(output_atom_folder)

# Atom dataset that returns a AtomInput based on folders of atom inputs stored on disk

class AtomDataset(Dataset):
    def __init__(
        self,
        folder: str | Path
    ):
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.exists() and folder.is_dir(), f'atom dataset not found at {str(folder)}'

        self.folder = folder
        self.files = [*folder.glob('**/*.pt')]

        assert len(self) > 0, f'no valid atom .pt files found at {str(folder)}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> AtomInput:
        file = self.files[idx]
        return file_to_atom_input(file)

# functions for extracting atom and atompair features (atom_inputs, atompair_inputs)

# atom reference position to atompair inputs
# will be used in the `default_extract_atompair_feats_fn` below in MoleculeInput

@typecheck
def atom_ref_pos_to_atompair_inputs(
    atom_ref_pos: Float['m 3'],
    atom_ref_space_uid: Int[' m'] | None = None,
) -> Float['m m 5']:

    # Algorithm 5 - lines 2-6

    # line 2

    pairwise_rel_pos = einx.subtract('i c, j c -> i j c', atom_ref_pos, atom_ref_pos)

    # line 5 - pairwise inverse squared distance

    atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim = -1, p = 2) ** 2) ** -1
    
    # line 3

    if exists(atom_ref_space_uid):
        same_ref_space_mask = einx.equal('i, j -> i j', atom_ref_space_uid, atom_ref_space_uid)
    else:
        same_ref_space_mask = torch.ones_like(atom_inv_square_dist).bool()

    # concat all into atompair_inputs for projection into atompair_feats within Alphafold3

    atompair_inputs, _ = pack((
        pairwise_rel_pos,
        atom_inv_square_dist,
        same_ref_space_mask.float(),
    ), 'i j *')

    # mask out

    atompair_inputs = einx.where(
        'i j, i j dapi, -> i j dapi',
        same_ref_space_mask, atompair_inputs, 0.
    )

    # return

    return atompair_inputs

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

    return atom_ref_pos_to_atompair_inputs(all_atom_pos_tensor) # what they did in the paper, but can be overwritten

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens
# `n` here is the token length, which accounts for molecules that are one token per atom

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
    is_molecule_mod:            Bool['n num_mods'] | Bool[' n'] | None = None
    molecule_atom_indices:      List[int | None] | None = None
    distogram_atom_indices:     List[int | None] | None = None
    atom_indices_for_frame:     Int['n 3'] | None = None
    missing_atom_indices:       List[Int[' _'] | None] | None = None
    missing_token_indices:      List[Int[' _'] | None] | None = None
    atom_parent_ids:            Int[' m'] | None = None
    additional_msa_feats:       Float['s n dmf'] | None = None
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dmi'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    resolved_labels:            Int[' m'] | None = None
    resolution:                 Float[''] | None = None
    chains:                     Tuple[int | None, int | None] | None = (None, None)
    filepath:                   str | None = None
    add_atom_ids:               bool = False
    add_atompair_ids:           bool = False
    directed_bonds:             bool = False
    extract_atom_feats_fn:      Callable[[Atom], Float['m dai']] = default_extract_atom_feats_fn
    extract_atompair_feats_fn:  Callable[[Mol], Float['m m dapi']] = default_extract_atompair_feats_fn

@typecheck
def molecule_to_atom_input(mol_input: MoleculeInput) -> AtomInput:
    i = mol_input

    molecules = i.molecules
    atom_lens = i.molecule_token_pool_lens
    extract_atom_feats_fn = i.extract_atom_feats_fn
    extract_atompair_feats_fn = i.extract_atompair_feats_fn

    # validate total number of atoms

    mol_total_atoms = sum([mol.GetNumAtoms() for mol in molecules])
    assert mol_total_atoms == sum(atom_lens), f'total atoms summed up from molecules passed in on `molecules` ({mol_total_atoms}) does not equal the number of atoms summed up in the field `molecule_token_pool_lens` {sum(atom_lens)}'

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
            assert (
                atom_symbol in atom_index
            ), f"{atom_symbol} not found in the ATOMS defined in life.py"

            atom_ids.append(atom_index[atom_symbol])

        atom_ids = tensor(atom_ids, dtype=torch.long)

    # get List[int] of number of atoms per molecule
    # for the offsets when building the atompair feature map / bonds

    all_num_atoms = tensor([mol.GetNumAtoms() for mol in molecules])
    offsets = exclusive_cumsum(all_num_atoms)

    # handle maybe missing atom indices

    missing_atom_mask = None
    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        assert len(molecules) == len(
            i.missing_atom_indices
        ), f"{len(i.missing_atom_indices)} missing atom indices does not match the number of molecules given ({len(molecules)})"

        missing_atom_indices: List[Int[" _"]] = [
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_atom_indices
        ]
        missing_token_indices: List[Int[" _"]] = [
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_token_indices
        ]

        missing_atom_mask: List[Bool[" _"]] = []

        for num_atoms, mol_missing_atom_indices in zip(all_num_atoms, missing_atom_indices):
            mol_miss_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)

            if mol_missing_atom_indices.numel() > 0:
                mol_miss_atom_mask.scatter_(-1, mol_missing_atom_indices, True)

            missing_atom_mask.append(mol_miss_atom_mask)

        missing_atom_mask = torch.cat(missing_atom_mask)
        missing_token_indices = pad_sequence(
            # NOTE: padding value must be any negative integer besides -1,
            # to not erroneously detect "missing" token center/distogram atoms
            # within ligands
            missing_token_indices,
            batch_first=True,
            padding_value=-2,
        )

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atom_bond_index = {symbol: (idx + 1) for idx, symbol in enumerate(ATOM_BONDS)}
        num_atom_bond_types = len(atom_bond_index)

        other_index = len(ATOM_BONDS) + 1

        atompair_ids = torch.zeros(total_atoms, total_atoms).long()

        # need the asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna | is_ligand | is_metal_ion` for is_molecule_types (chainable biomolecules)
        # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `asym_ids`

        asym_ids = i.additional_molecule_feats[..., 2]
        asym_ids = F.pad(asym_ids, (1, 0), value=-1)
        is_first_mol_in_chains = (asym_ids[1:] - asym_ids[:-1]) == 1

        is_chainable_biomolecules = i.is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)

        # for every molecule, build the bonds id matrix and add to `atompair_ids`

        prev_mol = None
        prev_src_tgt_atom_indices = None

        for (
            mol,
            is_first_mol_in_chain,
            is_chainable_biomolecule,
            src_tgt_atom_indices,
            offset,
        ) in zip(
            molecules,
            is_first_mol_in_chains,
            is_chainable_biomolecules,
            i.src_tgt_atom_indices,
            offsets,
        ):
            coordinates = []
            updates = []

            num_atoms = mol.GetNumAtoms()
            mol_atompair_ids = torch.zeros(num_atoms, num_atoms).long()

            for bond in mol.GetBonds():
                atom_start_index = bond.GetBeginAtomIdx()
                atom_end_index = bond.GetEndAtomIdx()

                coordinates.extend(
                    [[atom_start_index, atom_end_index], [atom_end_index, atom_start_index]]
                )

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

            mol_atompair_ids = einx.set_at(
                "[h w], c [2], c -> [h w]", mol_atompair_ids, coordinates, updates
            )

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

    atom_inputs: List[Float["m dai"]] = []

    for mol in molecules:
        atom_feats = []

        for atom in mol.GetAtoms():
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.append(torch.stack(atom_feats, dim=0))

    atom_inputs_tensor = torch.cat(atom_inputs).float()

    # atompair_inputs

    atompair_feats: List[Float["m m dapi"]] = []

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

    if exists(missing_token_indices) and missing_token_indices.shape[-1]:
        is_missing_molecule_atom = einx.equal(
            "n missing, n -> n missing", missing_token_indices, molecule_atom_indices
        ).any(dim=-1)
        is_missing_distogram_atom = einx.equal(
            "n missing, n -> n missing", missing_token_indices, distogram_atom_indices
        ).any(dim=-1)

        molecule_atom_indices = molecule_atom_indices.masked_fill(is_missing_molecule_atom, -1)
        distogram_atom_indices = distogram_atom_indices.masked_fill(is_missing_distogram_atom, -1)

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim=-2)

    # coerce chain indices into a tensor

    chains = tensor([default(chain, -1) for chain in i.chains]).long()

    # handle is_molecule_mod being one dimensional

    is_molecule_mod = i.is_molecule_mod

    if is_molecule_mod.ndim == 1:
        is_molecule_mod = rearrange(is_molecule_mod, 'n -> n 1')

    # atom input

    atom_input = AtomInput(
        atom_inputs=atom_inputs_tensor,
        atompair_inputs=atompair_inputs,
        molecule_atom_lens=atom_lens.long(),
        molecule_ids=i.molecule_ids,
        molecule_atom_indices=i.molecule_atom_indices,
        distogram_atom_indices=i.distogram_atom_indices,
        atom_indices_for_frame=i.atom_indices_for_frame,
        is_molecule_mod=is_molecule_mod,
        msa=i.msa,
        templates=i.templates,
        msa_mask=i.msa_mask,
        template_mask=i.template_mask,
        missing_atom_mask=missing_atom_mask,
        additional_msa_feats=i.additional_msa_feats,
        additional_token_feats=i.additional_token_feats,
        additional_molecule_feats=i.additional_molecule_feats,
        is_molecule_types=i.is_molecule_types,
        atom_pos=atom_pos,
        token_bonds=i.token_bonds,
        atom_parent_ids=i.atom_parent_ids,
        atom_ids=atom_ids,
        atompair_ids=atompair_ids,
        resolution=i.resolution,
        chains=chains,
        filepath=i.filepath,
    )

    return atom_input

# molecule lengthed molecule input
# molecule input - accepting list of molecules as rdchem.Mol

# `n` here refers to the actual number of molecules, NOT the `n` used within Alphafold3
# the proper token length needs to be correctly computed in the corresponding function for MoleculeLengthMoleculeInput -> AtomInput

@typecheck
@dataclass
class MoleculeLengthMoleculeInput:
    molecules:                  List[Mol]
    molecule_ids:               Int[' n']
    additional_molecule_feats:  Int[f'n {ADDITIONAL_MOLECULE_FEATS-1}']
    is_molecule_types:          Bool[f'n {IS_MOLECULE_TYPES}']
    src_tgt_atom_indices:       Int['n 2']
    token_bonds:                Bool['n n'] | None = None
    one_token_per_atom:         List[bool] | None = None
    is_molecule_mod:            Bool['n num_mods'] | Bool[' n'] | None = None
    molecule_atom_indices:      List[int | None] | None = None
    distogram_atom_indices:     List[int | None] | None = None
    atom_indices_for_frame:     List[Tuple[int, int, int] | None] | None = None
    missing_atom_indices:       List[Int[' _'] | None] | None = None
    missing_token_indices:      List[Int[' _'] | None] | None = None
    atom_parent_ids:            Int[' m'] | None = None
    additional_msa_feats:       Float['s n dmf'] | None = None
    additional_token_feats:     Float['n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dmi'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    resolved_labels:            Int[' m'] | None = None
    chains:                     Tuple[int | None, int | None] | None = (None, None)
    filepath:                   str | None = None
    add_atom_ids:               bool = False
    add_atompair_ids:           bool = False
    directed_bonds:             bool = False
    extract_atom_feats_fn:      Callable[[Atom], Float['m dai']] = default_extract_atom_feats_fn
    extract_atompair_feats_fn:  Callable[[Mol], Float['m m dapi']] = default_extract_atompair_feats_fn

@typecheck
def molecule_lengthed_molecule_input_to_atom_input(mol_input: MoleculeLengthMoleculeInput) -> AtomInput:
    i = mol_input

    molecules = i.molecules
    extract_atom_feats_fn = i.extract_atom_feats_fn
    extract_atompair_feats_fn = i.extract_atompair_feats_fn

    # derive `atom_lens` based on `one_token_per_atom`, for ligands and modified biomolecules

    atoms_per_molecule = tensor([mol.GetNumAtoms() for mol in molecules])
    ones = torch.ones_like(atoms_per_molecule)

    # `is_molecule_mod` can either be
    # 1. Bool['n'], in which case it will only be used for determining `one_token_per_atom`, or
    # 2. Bool['n num_mods'], where it will be passed to Alphafold3 for molecule modification embeds

    is_molecule_mod = i.is_molecule_mod
    is_molecule_any_mod = False

    if exists(is_molecule_mod):
        if i.is_molecule_mod.ndim == 2:
            is_molecule_any_mod = is_molecule_mod.any(dim = -1)
        else:
            is_molecule_any_mod = is_molecule_mod
            is_molecule_mod = None

    # get `one_token_per_atom`, which can be fully customizable

    if exists(i.one_token_per_atom):
        one_token_per_atom = tensor(i.one_token_per_atom)
    else:
        # if which molecule is `one_token_per_atom` is not passed in
        # default to what the paper did, which is ligands and any modified biomolecule
        is_ligand = i.is_molecule_types[..., IS_LIGAND_INDEX]
        one_token_per_atom = is_ligand | is_molecule_any_mod

    assert len(molecules) == len(one_token_per_atom)

    # derive the number of repeats needed to expand molecule lengths to token lengths

    token_repeats = torch.where(one_token_per_atom, atoms_per_molecule, ones)

    # derive atoms per token

    atom_repeat_input = torch.where(one_token_per_atom, ones, atoms_per_molecule)
    atoms_per_token = repeat_interleave(atom_repeat_input, token_repeats)

    total_atoms = atoms_per_molecule.sum().item()

    # derive `is_first_mol_in_chains` and `is_chainable_biomolecules` - needed for constructing `token_bonds

    # need the asym_id (to keep track of each molecule for each chain ascending) as well as `is_protein | is_dna | is_rna | is_ligand | is_metal_ion` for is_molecule_types (chainable biomolecules)
    # will do a single bond from a peptide or nucleotide to the one before. derive a `is_first_mol_in_chain` from `asym_ids`

    asym_ids = i.additional_molecule_feats[..., 2]
    asym_ids = F.pad(asym_ids, (1, 0), value=-1)
    is_first_mol_in_chains = (asym_ids[1:] - asym_ids[:-1]) == 1
    is_chainable_biomolecules = i.is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1)

    # repeat all the molecule lengths to the token lengths, using `one_token_per_atom`

    src_tgt_atom_indices = repeat_interleave(i.src_tgt_atom_indices, token_repeats, dim = 0)
    is_molecule_types = repeat_interleave(i.is_molecule_types, token_repeats, dim = 0)

    additional_molecule_feats = repeat_interleave(i.additional_molecule_feats, token_repeats, dim = 0)

    # insert the 2nd entry into additional molecule feats, which is just an arange over the number of tokens

    additional_molecule_feats, _ = pack((
        additional_molecule_feats[..., :1],
        torch.arange(additional_molecule_feats.shape[0]),
        additional_molecule_feats[..., 1:]
    ), 'n *')

    additional_msa_feats = repeat_interleave(i.additional_msa_feats, token_repeats, dim=1)

    additional_token_feats = repeat_interleave(i.additional_token_feats, token_repeats, dim = 0)
    molecule_ids = repeat_interleave(i.molecule_ids, token_repeats)

    atom_indices_offsets = repeat_interleave(exclusive_cumsum(atoms_per_molecule), token_repeats, dim = 0)

    distogram_atom_indices = repeat_interleave(i.distogram_atom_indices, token_repeats)
    molecule_atom_indices = repeat_interleave(i.molecule_atom_indices, token_repeats)

    msa = maybe(repeat_interleave)(i.msa, token_repeats, dim = -2)
    is_molecule_mod = maybe(repeat_interleave)(i.is_molecule_mod, token_repeats, dim = 0)

    templates = maybe(repeat_interleave)(i.templates, token_repeats, dim = -3)
    templates = maybe(repeat_interleave)(templates, token_repeats, dim = -2)

    # get all atoms

    atoms: List[Atom] = []

    for mol in molecules:
        atoms.extend([*mol.GetAtoms()])

    # construct the token bonds

    # will be linearly connected for proteins and nucleic acids
    # but for ligands, will have their atomic bond matrix (as ligands are atom resolution)

    num_tokens = token_repeats.sum().item()

    token_bonds = torch.zeros(num_tokens, num_tokens).bool()

    offset = 0

    for (
        mol,
        mol_is_chainable_biomolecule,
        mol_is_first_mol_in_chain,
        mol_is_one_token_per_atom
    ) in zip(
        molecules,
        is_chainable_biomolecules,
        is_first_mol_in_chains,
        one_token_per_atom
    ):
        num_atoms = mol.GetNumAtoms()

        if mol_is_chainable_biomolecule and not mol_is_first_mol_in_chain:
            token_bonds[offset, offset - 1] = True
            token_bonds[offset - 1, offset] = True

        if mol_is_one_token_per_atom:
            coordinates = []
            updates = []

            has_bond = torch.zeros(num_atoms, num_atoms).bool()

            for bond in mol.GetBonds():
                atom_start_index = bond.GetBeginAtomIdx()
                atom_end_index = bond.GetEndAtomIdx()

                coordinates.extend(
                    [[atom_start_index, atom_end_index], [atom_end_index, atom_start_index]]
                )

                updates.extend([True, True])

            coordinates = tensor(coordinates).long()
            updates = tensor(updates).bool()

            has_bond = einx.set_at("[h w], c [2], c -> [h w]", has_bond, coordinates, updates)

            row_col_slice = slice(offset, offset + num_atoms)
            token_bonds[row_col_slice, row_col_slice] = has_bond

        offset += (num_atoms if mol_is_one_token_per_atom else 1)

    # handle maybe atom embeds

    atom_ids = None

    if i.add_atom_ids:
        atom_index = {symbol: i for i, symbol in enumerate(ATOMS)}

        atom_ids = []

        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            assert (
                atom_symbol in atom_index
            ), f"{atom_symbol} not found in the ATOMS defined in life.py"

            atom_ids.append(atom_index[atom_symbol])

        atom_ids = tensor(atom_ids, dtype=torch.long)

    # get List[int] of number of atoms per molecule
    # for the offsets when building the atompair feature map / bonds

    all_num_atoms = tensor([mol.GetNumAtoms() for mol in molecules])
    offsets = exclusive_cumsum(all_num_atoms)

    # handle maybe missing atom indices

    missing_atom_mask = None
    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        assert len(molecules) == len(
            i.missing_atom_indices
        ), f"{len(i.missing_atom_indices)} missing atom indices does not match the number of molecules given ({len(molecules)})"

        missing_atom_indices: List[Int[" _"]] = [
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_atom_indices
        ]
        missing_token_indices: List[Int[" _"]] = [
            default(indices, torch.empty((0,), dtype=torch.long))
            for indices in i.missing_token_indices
        ]

        missing_atom_mask: List[Bool[" _"]] = []

        for num_atoms, mol_missing_atom_indices in zip(all_num_atoms, missing_atom_indices):
            mol_miss_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)

            if mol_missing_atom_indices.numel() > 0:
                mol_miss_atom_mask.scatter_(-1, mol_missing_atom_indices, True)

            missing_atom_mask.append(mol_miss_atom_mask)

        missing_atom_mask = torch.cat(missing_atom_mask)
        missing_token_indices = pad_sequence(
            # NOTE: padding value must be any negative integer besides -1,
            # to not erroneously detect "missing" token center/distogram atoms
            # within ligands
            missing_token_indices,
            batch_first=True,
            padding_value=-2,
        )

    # handle `atom_indices_for_frame` for the PAE

    atom_indices_for_frame = i.atom_indices_for_frame

    if exists(atom_indices_for_frame):
        atom_indices_for_frame = [default(indices, (-1, -1, -1)) for indices in i.atom_indices_for_frame]
        atom_indices_for_frame = tensor(atom_indices_for_frame)

    atom_indices_for_frame = repeat_interleave(atom_indices_for_frame, token_repeats, dim = 0)

    # handle maybe atompair embeds

    atompair_ids = None

    if i.add_atompair_ids:
        atom_bond_index = {symbol: (idx + 1) for idx, symbol in enumerate(ATOM_BONDS)}
        num_atom_bond_types = len(atom_bond_index)

        other_index = len(ATOM_BONDS) + 1

        atompair_ids = torch.zeros(total_atoms, total_atoms).long()

        # for every molecule, build the bonds id matrix and add to `atompair_ids`

        prev_mol = None
        prev_src_tgt_atom_indices = None

        for (
            mol,
            is_first_mol_in_chain,
            is_chainable_biomolecule,
            src_tgt_atom_indices,
            offset,
        ) in zip(
            molecules,
            is_first_mol_in_chains,
            is_chainable_biomolecules,
            i.src_tgt_atom_indices,
            offsets,
        ):
            coordinates = []
            updates = []

            num_atoms = mol.GetNumAtoms()
            mol_atompair_ids = torch.zeros(num_atoms, num_atoms).long()

            for bond in mol.GetBonds():
                atom_start_index = bond.GetBeginAtomIdx()
                atom_end_index = bond.GetEndAtomIdx()

                coordinates.extend(
                    [[atom_start_index, atom_end_index], [atom_end_index, atom_start_index]]
                )

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

            mol_atompair_ids = einx.set_at(
                "[h w], c [2], c -> [h w]", mol_atompair_ids, coordinates, updates
            )

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

    atom_inputs: List[Float["m dai"]] = []

    for mol in molecules:
        atom_feats = []

        for atom in mol.GetAtoms():
            atom_feats.append(extract_atom_feats_fn(atom))

        atom_inputs.append(torch.stack(atom_feats, dim=0))

    atom_inputs_tensor = torch.cat(atom_inputs).float()

    # atompair_inputs

    atompair_feats: List[Float["m m dapi"]] = []

    for mol, offset in zip(molecules, offsets):
        atompair_feats.append(extract_atompair_feats_fn(mol))

    assert len(atompair_feats) > 0

    dim_atompair_inputs = first(atompair_feats).shape[-1]

    atompair_inputs = torch.zeros((total_atoms, total_atoms, dim_atompair_inputs))

    for atompair_feat, num_atoms, offset in zip(atompair_feats, all_num_atoms, offsets):
        row_col_slice = slice(offset, offset + num_atoms)
        atompair_inputs[row_col_slice, row_col_slice] = atompair_feat

    # mask out molecule atom indices and distogram atom indices where it is in the missing atom indices list

    if exists(missing_token_indices) and missing_token_indices.shape[-1]:
        missing_token_indices = repeat_interleave(missing_token_indices, token_repeats, dim = 0)

        is_missing_molecule_atom = einx.equal(
            "n missing, n -> n missing", missing_token_indices, molecule_atom_indices
        ).any(dim=-1)

        is_missing_distogram_atom = einx.equal(
            "n missing, n -> n missing", missing_token_indices, distogram_atom_indices
        ).any(dim=-1)

        is_missing_atom_indices_for_frame = einx.equal(
            "n missing, n c -> n c missing", missing_token_indices, atom_indices_for_frame
        ).any(dim=(-1, -2))

        molecule_atom_indices = molecule_atom_indices.masked_fill(is_missing_molecule_atom, -1)
        distogram_atom_indices = distogram_atom_indices.masked_fill(is_missing_distogram_atom, -1)
        atom_indices_for_frame = atom_indices_for_frame.masked_fill(is_missing_atom_indices_for_frame[..., None], -1)

    # offsets for all indices

    distogram_atom_indices = offset_only_positive(distogram_atom_indices, atom_indices_offsets)
    molecule_atom_indices = offset_only_positive(molecule_atom_indices, atom_indices_offsets)
    atom_indices_for_frame = offset_only_positive(atom_indices_for_frame, atom_indices_offsets[..., None])

    # just use a hack to remove any duplicated indices (ligands and modified biomolecules) in a row

    atom_indices_for_frame = remove_consecutive_duplicate(atom_indices_for_frame)

    # handle atom positions

    atom_pos = i.atom_pos

    if exists(atom_pos) and isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos, dim=-2)

    # coerce chain indices into a tensor

    chains = tensor([default(chain, -1) for chain in i.chains]).long()

    # atom input

    atom_input = AtomInput(
        atom_inputs = atom_inputs_tensor,
        atompair_inputs = atompair_inputs,
        molecule_atom_lens = atoms_per_token,
        molecule_ids = molecule_ids,
        molecule_atom_indices = molecule_atom_indices,
        distogram_atom_indices = distogram_atom_indices,
        atom_indices_for_frame = atom_indices_for_frame,
        missing_atom_mask = missing_atom_mask,
        additional_msa_feats=additional_msa_feats,
        additional_token_feats = additional_token_feats,
        additional_molecule_feats = additional_molecule_feats,
        is_molecule_mod = is_molecule_mod,
        is_molecule_types = is_molecule_types,
        msa = msa,
        templates = templates,
        atom_pos = atom_pos,
        token_bonds = token_bonds,
        atom_parent_ids = i.atom_parent_ids,
        atom_ids = atom_ids,
        atompair_ids = atompair_ids,
        chains = chains,
        filepath=i.filepath,
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
    additional_msa_feats:       Float['s n dmf'] | None = None
    additional_token_feats:     Float[f'n dtf'] | None = None
    templates:                  Float['t n n dt'] | None = None
    msa:                        Float['s n dmi'] | None = None
    atom_pos:                   List[Float['_ 3']] | Float['m 3'] | None = None
    reorder_atom_pos:           bool = True
    template_mask:              Bool[' t'] | None = None
    msa_mask:                   Bool[' s'] | None = None
    distance_labels:            Int['n n'] | None = None
    resolved_labels:            Int[' m'] | None = None
    chains:                     Tuple[int | None, int | None] | None = (None, None)
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
    return_entries = False
) -> List[Mol] | Tuple[List[Mol], List[dict]]:

    if isinstance(indices, str):
        indices = list(indices)

    entries_list = list(entries.values())

    # get all the peptide or nucleotide entries

    if torch.is_tensor(indices):
        indices = indices.tolist()
        entries = [entries_list[i] for i in indices]
    else:
        entries = [entries[s] for s in indices]

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
def alphafold3_input_to_molecule_lengthed_molecule_input(alphafold3_input: Alphafold3Input) -> MoleculeLengthMoleculeInput:
    i = alphafold3_input

    chainable_biomol_entries: List[List[dict]] = []  # for reordering the atom positions at the end

    ss_rnas = list(i.ss_rna)
    ss_dnas = list(i.ss_dna)

    # handle atom positions - need atom positions for deriving frame of ligand for PAE

    atom_pos = i.atom_pos

    if isinstance(atom_pos, list):
        atom_pos = torch.cat(atom_pos)

    # any double stranded nucleic acids is added to single stranded lists with its reverse complement
    # rc stands for reverse complement

    for seq in i.ds_rna:
        rc_fn = (
            partial(reverse_complement, nucleic_acid_type="rna")
            if isinstance(seq, str)
            else reverse_complement_tensor
        )
        rc_seq = rc_fn(seq)
        ss_rnas.extend([seq, rc_seq])

    for seq in i.ds_dna:
        rc_fn = (
            partial(reverse_complement, nucleic_acid_type="dna")
            if isinstance(seq, str)
            else reverse_complement_tensor
        )
        rc_seq = rc_fn(seq)
        ss_dnas.extend([seq, rc_seq])

    # keep track of molecule_ids - for now it is
    # proteins (21) | rna (5) | dna (5) | gap? (1) - unknown for each biomolecule is the very last, ligand is 20

    rna_offset = len(HUMAN_AMINO_ACIDS)
    dna_offset = len(RNA_NUCLEOTIDES) + rna_offset
    ligand_id = len(HUMAN_AMINO_ACIDS) - 1

    molecule_ids = []

    # convert all proteins to a List[Mol] of each peptide

    proteins = i.proteins
    mol_proteins = []
    protein_entries = []

    atom_indices_for_frame = []
    distogram_atom_indices = []
    molecule_atom_indices = []
    src_tgt_atom_indices = []

    for protein in proteins:
        mol_peptides, protein_entries = map_int_or_string_indices_to_mol(
            HUMAN_AMINO_ACIDS, protein, return_entries=True
        )
        mol_proteins.append(mol_peptides)

        distogram_atom_indices.extend(
            [entry["token_center_atom_idx"] for entry in protein_entries]
        )
        molecule_atom_indices.extend([entry["distogram_atom_idx"] for entry in protein_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in protein_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in protein_entries]
        )

        protein_ids = maybe_string_to_int(HUMAN_AMINO_ACIDS, protein)
        molecule_ids.append(protein_ids)

        chainable_biomol_entries.append(protein_entries)

    # convert all single stranded nucleic acids to mol

    mol_ss_dnas = []
    mol_ss_rnas = []

    for seq in ss_rnas:
        mol_seq, ss_rna_entries = map_int_or_string_indices_to_mol(
            RNA_NUCLEOTIDES, seq, return_entries=True
        )
        mol_ss_rnas.append(mol_seq)

        distogram_atom_indices.extend([entry["token_center_atom_idx"] for entry in ss_rna_entries])
        molecule_atom_indices.extend([entry["distogram_atom_idx"] for entry in ss_rna_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in ss_rna_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in ss_rna_entries]
        )

        rna_ids = maybe_string_to_int(RNA_NUCLEOTIDES, seq) + rna_offset
        molecule_ids.append(rna_ids)

        chainable_biomol_entries.append(ss_rna_entries)

    for seq in ss_dnas:
        mol_seq, ss_dna_entries = map_int_or_string_indices_to_mol(
            DNA_NUCLEOTIDES, seq, return_entries=True
        )
        mol_ss_dnas.append(mol_seq)

        distogram_atom_indices.extend([entry["token_center_atom_idx"] for entry in ss_dna_entries])
        molecule_atom_indices.extend([entry["distogram_atom_idx"] for entry in ss_dna_entries])

        src_tgt_atom_indices.extend(
            [[entry["first_atom_idx"], entry["last_atom_idx"]] for entry in ss_dna_entries]
        )

        atom_indices_for_frame.extend(
            [entry["three_atom_indices_for_frame"] for entry in ss_dna_entries]
        )

        dna_ids = maybe_string_to_int(DNA_NUCLEOTIDES, seq) + dna_offset
        molecule_ids.append(dna_ids)

        chainable_biomol_entries.append(ss_dna_entries)

    # convert ligands to rdchem.Mol

    ligands = list(alphafold3_input.ligands)

    mol_ligands = [
        (mol_from_smile(ligand) if isinstance(ligand, str) else ligand) for ligand in ligands
    ]

    molecule_ids.append(tensor([ligand_id] * len(mol_ligands)))

    # handle frames for the ligands, which depends on knowing the atom positions (section 4.3.2)

    if exists(atom_pos):
        ligand_atom_pos_offset = 0

        for mol in flatten([*mol_proteins, *mol_ss_rnas, *mol_ss_dnas]):
            ligand_atom_pos_offset += mol.GetNumAtoms()

        for mol_ligand in mol_ligands:
            num_ligand_atoms = mol_ligand.GetNumAtoms()
            ligand_atom_pos = atom_pos[ligand_atom_pos_offset:(ligand_atom_pos_offset + num_ligand_atoms)]

            frames = get_frames_from_atom_pos(ligand_atom_pos, filter_colinear_pos = True)

            atom_indices_for_frame.append(frames.tolist())

            ligand_atom_pos_offset += num_ligand_atoms

    # convert metal ions to rdchem.Mol

    metal_ions = alphafold3_input.metal_ions
    mol_metal_ions = map_int_or_string_indices_to_mol(METALS, metal_ions)

    molecule_ids.append(tensor([MOLECULE_METAL_ION_ID] * len(mol_metal_ions)))

    # create the molecule input

    all_protein_mols = flatten(mol_proteins)
    all_rna_mols = flatten(mol_ss_rnas)
    all_dna_mols = flatten(mol_ss_dnas)

    molecules_without_ligands = [
        *all_protein_mols,
        *all_rna_mols,
        *all_dna_mols,
    ]

    # correctly generate the is_molecule_types, which is a boolean tensor of shape [*, 5]
    # is_protein | is_rna | is_dna | is_ligand | is_metal_ions
    # this is needed for their special diffusion loss

    molecule_type_token_lens = [
        len(all_protein_mols),
        len(all_rna_mols),
        len(all_dna_mols),
        len(mol_ligands),
        len(mol_metal_ions)
    ]

    num_tokens = sum(molecule_type_token_lens)

    assert num_tokens > 0, "you have an empty alphafold3 input"

    arange = torch.arange(num_tokens)[:, None]

    molecule_types_lens_cumsum = tensor([0, *molecule_type_token_lens]).cumsum(dim=-1)
    left, right = molecule_types_lens_cumsum[:-1], molecule_types_lens_cumsum[1:]

    is_molecule_types = (arange >= left) & (arange < right)

    # pad the src-tgt indices

    src_tgt_atom_indices = tensor(src_tgt_atom_indices)

    src_tgt_atom_indices = pad_to_len(src_tgt_atom_indices, num_tokens, dim = -2)

    # all molecules, layout is
    # proteins | ss rna | ss dna | ligands | metal ions

    molecules = [
        *molecules_without_ligands,
        *mol_ligands,
        *mol_metal_ions
    ]

    for mol in molecules:
        Chem.SanitizeMol(mol)

    # handle rest of non-biomolecules for atom_indices_for_frame

    atom_indices_for_frame = [*atom_indices_for_frame, *([None] * (len(molecules) - len(atom_indices_for_frame)))]

    assert len(atom_indices_for_frame) == len(molecules)

    # handle molecule ids

    molecule_ids = torch.cat(molecule_ids).long()

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

    atom_counts = [
        *num_protein_atoms,
        *num_ss_rna_atoms,
        *num_ss_dna_atoms,
        *num_ligand_atoms,
        len(metal_ions),
    ]

    atom_parent_ids = repeat_interleave(torch.arange(len(atom_counts)), tensor(atom_counts))

    # constructing the additional_molecule_feats
    # which is in turn used to derive relative positions
    # (todo) offer a way to precompute relative positions at data prep

    # residue_index - an arange that restarts at 1 for each chain
    # token_index   - just an arange
    # asym_id       - unique id for each chain of a biomolecule
    # entity_id     - unique id for each biomolecule sequence
    # sym_id        - unique id for each chain of the same biomolecule sequence

    num_protein_tokens = [len(protein) for protein in proteins]
    num_ss_rna_tokens = [len(rna) for rna in ss_rnas]
    num_ss_dna_tokens = [len(dna) for dna in ss_dnas]

    ligand_tokens: List[int] = [] if len(mol_ligands) == 0 else [len(mol_ligands)]

    token_repeats = tensor([
        *num_protein_tokens,
        *num_ss_rna_tokens,
        *num_ss_dna_tokens,
        *ligand_tokens,
        len(metal_ions),
    ])

    # residue ids

    residue_index = torch.cat([torch.arange(i) for i in token_repeats])

    # asym ids

    asym_ids = repeat_interleave(torch.arange(len(token_repeats)), token_repeats)

    # entity ids

    unrepeated_entity_sequences = defaultdict(int)
    for entity_sequence in (*proteins, *ss_rnas, *ss_dnas, *ligands, *metal_ions):
        if entity_sequence in unrepeated_entity_sequences:
            continue
        unrepeated_entity_sequences[entity_sequence] = len(unrepeated_entity_sequences)

    unrepeated_entity_ids = [
        unrepeated_entity_sequences[entity_sequence]
        for entity_sequence in (*proteins, *ss_rnas, *ss_dnas, *ligands, *metal_ions)
    ]

    entity_id_counts = [
        *num_protein_tokens,
        *[len(rna) for rna in i.ss_rna],
        *[len(rna) for rna in i.ds_rna for _ in range(2)],
        *[len(dna) for dna in i.ss_dna],
        *[len(dna) for dna in i.ds_dna for _ in range(2)],
        *ligand_tokens,
        *[1 for _ in metal_ions],
    ]

    entity_ids = repeat_interleave(tensor(unrepeated_entity_ids), tensor(entity_id_counts))

    # sym ids

    unrepeated_sym_ids = []
    unrepeated_sym_sequences = defaultdict(int)
    for entity_sequence in (*proteins, *ss_rnas, *ss_dnas, *ligands, *metal_ions):
        unrepeated_sym_ids.append(unrepeated_sym_sequences[entity_sequence])
        if entity_sequence in unrepeated_sym_sequences:
            unrepeated_sym_sequences[entity_sequence] += 1

    sym_ids = repeat_interleave(tensor(unrepeated_sym_ids), tensor(entity_id_counts))

    # concat for all of additional_molecule_feats

    additional_molecule_feats = torch.stack((
        residue_index,
        asym_ids,
        entity_ids,
        sym_ids
    ), dim=-1)

    # distogram and token centre atom indices

    distogram_atom_indices = tensor(distogram_atom_indices)
    distogram_atom_indices = pad_to_len(distogram_atom_indices, num_tokens, value=-1)

    molecule_atom_indices = tensor(molecule_atom_indices)
    molecule_atom_indices = pad_to_len(molecule_atom_indices, num_tokens, value=-1)

    # handle missing atom indices

    missing_atom_indices = None
    missing_token_indices = None

    if exists(i.missing_atom_indices) and len(i.missing_atom_indices) > 0:
        missing_atom_indices = []
        missing_token_indices = []

        for mol_index, (mol_miss_atom_indices, mol) in enumerate(
            zip(i.missing_atom_indices, molecules)
        ):
            mol_miss_atom_indices = default(mol_miss_atom_indices, [])
            mol_miss_atom_indices = tensor(mol_miss_atom_indices, dtype=torch.long)

            missing_atom_indices.append(mol_miss_atom_indices)
            missing_token_indices.append(mol_miss_atom_indices)

        assert len(molecules) == len(missing_atom_indices)
        assert len(missing_token_indices) == num_tokens

    # handle MSAs

    num_msas = len(i.msa) if exists(i.msa) else 1

    # create molecule input

    molecule_input = MoleculeLengthMoleculeInput(
        molecules=molecules,
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        molecule_ids=molecule_ids,
        additional_molecule_feats=additional_molecule_feats,
        additional_msa_feats=default(i.additional_msa_feats, torch.zeros(num_msas, num_tokens, 2)),
        additional_token_feats=default(i.additional_token_feats, torch.zeros(num_tokens, 33)),
        is_molecule_types=is_molecule_types,
        missing_atom_indices=missing_atom_indices,
        missing_token_indices=missing_token_indices,
        src_tgt_atom_indices=src_tgt_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        atom_pos=atom_pos,
        templates=i.templates,
        msa=i.msa,
        template_mask=i.template_mask,
        msa_mask=i.msa_mask,
        atom_parent_ids=atom_parent_ids,
        chains=i.chains,
        add_atom_ids=i.add_atom_ids,
        add_atompair_ids=i.add_atompair_ids,
        directed_bonds=i.directed_bonds,
        extract_atom_feats_fn=i.extract_atom_feats_fn,
        extract_atompair_feats_fn=i.extract_atompair_feats_fn,
    )

    return molecule_input

# pdb input

@typecheck
@dataclass
class PDBInput:
    """Dataclass for PDB inputs."""

    mmcif_filepath: str | None = None
    biomol: Biomolecule | None = None
    chains: Tuple[str | None, str | None] | None = (None, None)
    cropping_config: Dict[str, float | int] | None = None
    msa_dir: str | None = None
    templates_dir: str | None = None
    add_atom_ids: bool = False
    add_atompair_ids: bool = False
    directed_bonds: bool = False
    training: bool = False
    resolution: float | None = None
    max_msas_per_chain: int | None = None
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore

    def __post_init__(self):
        """Run post-init checks."""

        if exists(self.mmcif_filepath):
            if not os.path.exists(self.mmcif_filepath):
                raise FileNotFoundError(f"mmCIF file not found: {self.mmcif_filepath}.")
            if not self.mmcif_filepath.endswith(".cif"):
                raise ValueError(
                    f"mmCIF file `{self.mmcif_filepath}` must have a `.cif` file extension."
                )
        elif not exists(self.biomol):
            raise ValueError(
                "Either an mmCIF file or a `Biomolecule` object must be provided."
            )

        if exists(self.cropping_config):
            assert self.cropping_config.keys() == {
                "contiguous_weight",
                "spatial_weight",
                "spatial_interface_weight",
                "n_res",
            }, (
                f"Invalid cropping config keys: {self.cropping_config.keys()}. "
                "Please ensure that the cropping config has the correct keys."
            )
            assert (
                sum(
                    [
                        self.cropping_config["contiguous_weight"],
                        self.cropping_config["spatial_weight"],
                        self.cropping_config["spatial_interface_weight"],
                    ]
                )
                == 1.0
            ), (
                f"Invalid cropping config weights: ({self.cropping_config['contiguous_weight']}, {self.cropping_config['spatial_weight']}, {self.cropping_config['spatial_interface_weight']}). "
                "Please ensure that the cropping config weights sum to 1.0."
            )
            assert self.cropping_config["n_res"] > 0, (
                f"Invalid number of residues for cropping: {self.cropping_config['n_res']}. "
                "Please ensure that the number of residues for cropping is greater than 0."
            )

        if exists(self.msa_dir) and not os.path.exists(self.msa_dir):
            raise FileNotFoundError(
                f"Provided MSA directory not found: {self.msa_dir}."
            )

        if exists(self.templates_dir) and not os.path.exists(self.templates_dir):
            raise FileNotFoundError(
                f"Provided templates directory not found: {self.templates_dir}."
            )


@typecheck
def extract_chain_sequences_from_biomolecule_chemical_components(
    biomol: Biomolecule,
    chem_comps: List[mmcif_parsing.ChemComp],
) -> Tuple[List[str], List[PDB_INPUT_RESIDUE_MOLECULE_TYPE]]:
    """Extract paired chain sequences and chemical types from `Biomolecule` chemical components."""
    chain_index = biomol.chain_index
    residue_index = biomol.residue_index

    assert len(chem_comps) == len(chain_index), (
        f"The number of chemical components ({len(chem_comps)}), chain indices ({len(chain_index)}), and residue indices do not match. "
        "Please ensure that chain and residue indices are correctly assigned to each chemical component."
    )

    chain_seqs = []
    current_chain_seq = []

    chain_res_idx_seen = set()
    for idx, (comp_details, chain_idx, res_idx) in enumerate(
        zip(chem_comps, chain_index, residue_index)
    ):
        # only consider the first atom of each (e.g., ligand) residue

        chain_res_idx = f"{chain_idx}:{res_idx}"
        if chain_res_idx in chain_res_idx_seen:
            current_chain_seq = []
            continue

        is_polymer_residue = is_polymer(comp_details.type)
        residue_constants = get_residue_constants(comp_details.type)
        restype = residue_constants.restype_3to1.get(comp_details.id, "X")
        is_modified_polymer_residue = is_polymer_residue and restype == "X"

        # map chemical types to protein, DNA, RNA, or ligand,
        # treating modified polymer residues as ligands

        res_chem_type = get_pdb_input_residue_molecule_type(
            comp_details.type,
            is_modified_polymer_residue=is_modified_polymer_residue,
        )

        # aggregate the residue sequences of each chain

        if not current_chain_seq:
            chain_seqs.append(current_chain_seq)
        mapped_restype = comp_details.id if is_atomized_residue(res_chem_type) else restype
        current_chain_seq.append((mapped_restype, res_chem_type))

        # reset current_chain_seq if the next residue is either not part of the current chain or is a different (unmodified) molecule type

        unmod_res_chem_type = get_pdb_input_residue_molecule_type(
            comp_details.type,
            is_modified_polymer_residue=False,
        )

        chain_ending = idx + 1 < len(chain_index) and chain_index[idx] != chain_index[idx + 1]
        chem_type_ending = idx + 1 < len(chem_comps) and unmod_res_chem_type != (
            get_pdb_input_residue_molecule_type(
                chem_comps[idx + 1].type,
                is_modified_polymer_residue=False,
            )
        )
        if chain_ending or chem_type_ending:
            current_chain_seq = []

        # keep track of the chain-residue ID pairs seen so far

        chain_res_idx_seen.add(chain_res_idx)

    # efficiently build sequence strings

    mapped_chain_seqs = []
    mapped_chain_chem_types = []
    for chain_seq in chain_seqs:
        # NOTE: from here on, all residue chemical types are guaranteed to be identical within a chain
        chain_chem_type = chain_seq[-1][-1]
        if is_atomized_residue(chain_chem_type):
            for seq, chem_type in chain_seq:
                # NOTE: there originally may have been e.g., multiple ligands in the same chain,
                # so we need to aggregate their CCD codes in order
                mapped_chain_seqs.append(seq)
                mapped_chain_chem_types.append(chem_type)
        else:
            mapped_chain_seqs.append("".join([res[0] for res in chain_seq]))
            mapped_chain_chem_types.append(chain_chem_type)

    assert len(mapped_chain_seqs) == len(mapped_chain_chem_types), (
        f"The number of mapped chain sequences ({len(mapped_chain_seqs)}) does not match the number of mapped chain chemical types ({len(mapped_chain_chem_types)}). "
        "Please ensure that the chain sequences and chemical types are correctly aggregated."
    )
    return mapped_chain_seqs, mapped_chain_chem_types


@typecheck
def add_atom_positions_to_mol(
    mol: Mol,
    atom_positions: np.ndarray,
    missing_atom_indices: Set[int],
) -> Mol:
    """Add atom positions to an RDKit molecule's first conformer while accounting for missing
    atoms."""
    assert len(missing_atom_indices) <= mol.GetNumAtoms(), (
        f"The number of missing atom positions ({len(missing_atom_indices)}) and atoms in the RDKit molecule ({mol.GetNumAtoms()}) are not reconcilable. "
        "Please ensure that these input features are all correctly paired."
    )

    # set missing atom positions to (0, 0, 0) while preserving the order of the remaining atoms

    missing_atom_counter = 0
    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx in missing_atom_indices:
            conf.SetAtomPosition(atom_idx, (0.0, 0.0, 0.0))
            missing_atom_counter += 1
        else:
            conf.SetAtomPosition(atom_idx, atom_positions[atom_idx - missing_atom_counter])

    Chem.SanitizeMol(mol)

    # set a property to indicate the atom positions that are missing

    mol.SetProp("missing_atom_indices", ",".join(map(str, sorted(missing_atom_indices))))

    return mol


def create_mol_from_atom_positions_and_types(
    name: str,
    atom_positions: np.ndarray,
    element_types: List[str],
    missing_atom_indices: Set[int],
    neutral_stable_mol_hypothesis: bool = True,
    verbose: bool = False,
) -> Mol:
    """Create an RDKit molecule from a NumPy array of atom positions and a list of their element
    types.

    :param name: The name of the molecule.
    :param atom_positions: A NumPy array of shape (num_atoms, 3) containing the 3D coordinates of
        each atom.
    :param element_types: A list of element symbols for each atom in the molecule.
    :param missing_atom_indices: A set of atom indices that are missing from the atom_positions
        array.
    :param neutral_stable_mol_hypothesis: Whether to convert radical electrons into explicit
        hydrogens based on the `PDB neutral stable molecule` hypothesis.
    :param verbose: Whether to log warnings when bond determination fails.
    :return: An RDKit molecule with the specified atom positions and element types.
    """
    if len(atom_positions) != len(element_types):
        raise ValueError("The length of atom_elements and xyz_coordinates must be the same.")

    # populate an empty editable molecule

    mol = Chem.RWMol()
    mol.SetProp("_Name", name)

    for element_type in element_types:
        atom = Chem.Atom(element_type)
        mol.AddAtom(atom)

    # set 3D coordinates

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(atom_positions):
        conf.SetAtomPosition(i, Point3D(x, y, z))

    # add the conformer to the molecule

    mol.AddConformer(conf)

    # block the RDKit logger

    blocker = rdBase.BlockLogs()

    # finalize molecule by inferring bonds

    try:
        with StringIO() as buf:
            with redirect_stderr(buf):
                # redirect RDKit's stderr to a buffer to suppress warnings
                rdDetermineBonds.DetermineBonds(mol, allowChargedFragments=False)
    except Exception as e:
        if verbose:
            logger.warning(
                f"Failed to determine bonds for the input molecule {name} due to: {e}. Skipping bond determination."
            )

    # clean up the molecule

    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(mol, catchErrors=True)

    # based on the `PDB neutral stable molecule` hypothesis
    # (see https://github.com/rdkit/rdkit/issues/2683#issuecomment-2273998084),
    # convert radical electrons into explicit hydrogens

    if neutral_stable_mol_hypothesis:
        for a in mol.GetAtoms():
            if a.GetNumRadicalElectrons():
                a.SetNumExplicitHs(a.GetNumRadicalElectrons())
                a.SetNumRadicalElectrons(0)
            Chem.SanitizeMol(mol, catchErrors=True)

    # unblock the RDKit logger

    del blocker

    # set a property to indicate the atom positions that are missing

    mol.SetProp("missing_atom_indices", ",".join(map(str, sorted(missing_atom_indices))))

    return mol


@typecheck
def extract_template_molecules_from_biomolecule_chains(
    biomol: Biomolecule,
    chain_seqs: List[str],
    chain_chem_types: List[PDB_INPUT_RESIDUE_MOLECULE_TYPE],
    mol_keyname: str = "rdchem_mol",
    verbose: bool = False,
) -> Tuple[List[Mol], List[PDB_INPUT_RESIDUE_MOLECULE_TYPE]]:
    """Extract RDKit template molecules and their types for the residues of each `Biomolecule`
    chain.

    NOTE: Missing atom indices are marked as a comma-separated property string for each RDKit molecule
    and can be retrieved via `mol.GetProp('missing_atom_indices')`.
    """
    chain_index = biomol.chain_index
    residue_index = biomol.residue_index
    residue_types = biomol.restype
    atom_positions = biomol.atom_positions
    atom_mask = biomol.atom_mask.astype(bool)

    assert len(chain_seqs) == len(chain_chem_types), (
        f"The number of chain sequences ({len(chain_seqs)}) and chain chemical types ({len(chain_chem_types)}) do not match. "
        "Please ensure that these input features are all correctly paired."
    )
    assert len(chain_index) == len(residue_index) == len(residue_types) == len(atom_positions), (
        f"The number of chain indices ({len(chain_index)}), residue indices ({len(residue_index)}), residue types ({len(residue_types)}), and atom positions ({len(atom_positions)}) do not match. "
        "Please ensure that these input features are correctly paired."
    )
    assert atom_positions.shape[:-1] == atom_mask.shape, (
        f"The number of atom positions ({atom_positions.shape[:-1]}) and atom masks ({atom_mask.shape}) do not match. "
        "Please ensure that these input features are correctly paired."
    )
    assert exists(CCD_COMPONENTS_SMILES), (
        f"The PDB Chemical Component Dictionary (CCD) components SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} does not exist. "
        f"Please re-run this script after ensuring the preliminary CCD file {CCD_COMPONENTS_FILEPATH} has been downloaded according to this project's `README.md` file."
        f"After doing so, the SMILES file {CCD_COMPONENTS_SMILES_FILEPATH} will be cached locally and used for subsequent runs."
    )

    res_index = 0
    molecules = []
    molecule_types = []
    for seq, chem_type in zip(chain_seqs, chain_chem_types):
        # map chemical types to protein, DNA, RNA, or ligand sequences

        if chem_type == "protein":
            seq_mapping = HUMAN_AMINO_ACIDS
        elif chem_type == "rna":
            seq_mapping = RNA_NUCLEOTIDES
        elif chem_type == "dna":
            seq_mapping = DNA_NUCLEOTIDES
        elif is_atomized_residue(chem_type):
            seq_mapping = CCD_COMPONENTS_SMILES
        else:
            raise ValueError(f"Unrecognized chain chemical type: {chem_type}")

        # map residue types to atom positions

        res_constants = get_residue_constants(
            chem_type.replace("protein", "peptide").replace("mod_", "")
        )
        atom_mapping = res_constants.restype_atom47_to_compact_atom

        mol_seq = []
        for res in seq:
            # Ligand and modified polymer (i.e., atomized) residues
            if is_atomized_residue(chem_type):
                if seq not in seq_mapping:
                    raise ValueError(
                        f"Could not locate the PDB CCD's SMILES string for atomized residue: {seq}"
                    )

                # construct template molecule for post-mapping bond orders

                smile = seq_mapping[seq]
                template_mol = mol_from_smile(smile)

                # find all atom positions and masks for the current atomized residue

                res_residue_index = residue_index[res_index]
                res_chain_index = chain_index[res_index]
                res_ligand_atom_mask = (residue_index == res_residue_index) & (
                    chain_index == res_chain_index
                )
                res_atom_positions = atom_positions[res_ligand_atom_mask]
                res_atom_mask = atom_mask[res_ligand_atom_mask]

                # manually construct an RDKit molecule from the atomized residue's atom positions and types

                res_atom_type_indices = np.where(res_atom_positions.any(axis=-1))[1]
                res_atom_elements = [
                    # NOTE: here, we treat the first character of each atom type as its element symbol
                    res_constants.element_types[idx].replace("ATM", "*")
                    for idx in res_atom_type_indices
                ]
                mol = create_mol_from_atom_positions_and_types(
                    # NOTE: for now, we construct molecules without referencing canonical
                    # SMILES strings, which means there are no missing molecule atoms by design
                    name=seq,
                    atom_positions=res_atom_positions[res_atom_mask],
                    element_types=res_atom_elements,
                    missing_atom_indices=set(),
                    verbose=verbose,
                )
                try:
                    mol = AllChem.AssignBondOrdersFromTemplate(template_mol, mol)
                except Exception as e:
                    if verbose:
                        logger.warning(
                            f"Failed to assign bond orders from the template atomized molecule for residue {seq} due to: {e}. "
                            "Skipping bond order assignment."
                        )
                res_index += mol.GetNumAtoms()

            # (Unmodified) polymer residues
            else:
                mol = copy.deepcopy(seq_mapping[res][mol_keyname])

                res_type = residue_types[res_index]
                res_atom_mask = atom_mask[res_index]

                # start by finding all possible atoms that may be present in the residue

                res_unique_atom_mapping_indices = np.unique(
                    atom_mapping[res_type - res_constants.min_restype_num],
                    return_index=True,
                )[1]
                res_unique_atom_mapping = np.array(
                    [
                        atom_mapping[res_type - res_constants.min_restype_num][idx]
                        for idx in sorted(res_unique_atom_mapping_indices)
                    ]
                )

                # then find the subset of atoms that are actually present in the residue,
                # and gather the corresponding indices needed to remap these atoms
                # from `atom47` atom type indexing to compact atom type indexing
                # (e.g., mapping from `atom47` coordinates to `atom14` coordinates
                # uniquely for each type of amino acid residue)

                res_atom_mapping = atom_mapping[res_type - res_constants.min_restype_num][
                    res_atom_mask
                ]
                res_atom_mapping_set = set(res_atom_mapping)

                # ensure any missing atoms are accounted for in the atom positions during index remapping

                missing_atom_indices = {
                    idx
                    for idx in range(len(res_unique_atom_mapping))
                    if res_unique_atom_mapping[idx] not in res_atom_mapping_set
                }

                contiguous_res_atom_mapping = {
                    # NOTE: `np.unique` already sorts the unique values
                    value: index
                    for index, value in enumerate(np.unique(res_atom_mapping))
                }
                contiguous_res_atom_mapping = np.vectorize(contiguous_res_atom_mapping.get)(
                    res_atom_mapping
                )

                res_atom_positions = atom_positions[res_index][res_atom_mask][
                    contiguous_res_atom_mapping
                ]
                mol = add_atom_positions_to_mol(
                    mol,
                    res_atom_positions.reshape(-1, 3),
                    missing_atom_indices,
                )
                mol.SetProp("_Name", res)
                res_index += 1

            mol_seq.append(mol)
            molecule_types.append(chem_type)
            if is_atomized_residue(chem_type):
                break

        molecules.extend(mol_seq)

    assert res_index == len(atom_positions), (
        f"The number of residues matched to atom positions ({res_index}) does not match the number of atom positions ({len(atom_positions)}) available. "
        "Please ensure that these input features were correctly paired."
    )
    assert len(molecules) == len(molecule_types), (
        f"The number of molecules ({len(molecules)}) does not match the number of molecule types ({len(molecule_types)}). "
        "Please ensure that these lists were correctly paired."
    )
    return molecules, molecule_types


@typecheck
def get_token_index_from_composite_atom_id(
    biomol: Biomolecule,
    chain_id: str,
    res_id: int,
    atom_name: str,
    atom_index: int,
    is_polymer_residue: bool,
) -> np.int64:
    """Get the token index (indices) of an atom (residue) in a biomolecule from its chain ID,
    residue ID, and atom name."""
    chain_mask = biomol.chain_id == chain_id
    res_mask = biomol.residue_index == res_id
    atom_mask = biomol.atom_name == atom_name

    if is_polymer_residue:
        return np.where(chain_mask & res_mask)[0][atom_index]
    else:
        return np.where(chain_mask & res_mask & atom_mask)[0][atom_index]


@typecheck
def find_mismatched_symmetry(
    asym_ids: np.ndarray,
    entity_ids: np.ndarray,
    sym_ids: np.ndarray,
    chemid: np.ndarray,
) -> bool:
    """Find mismatched symmetry in a biomolecule's asymmetry, entity, symmetry, and token chemical
    IDs.

    This function compares the chemical IDs of (related) regions with the same entity ID but
    different symmetry IDs. If the chemical IDs of these regions' matching asymmetric chain ID
    regions are not equal, then their symmetry is "mismatched".

    :param asym_ids: An array of asymmetric unit (i.e., chain) IDs for each token in the
        biomolecule.
    :param entity_ids: An array of entity IDs for each token in the biomolecule.
    :param sym_ids: An array of symmetry IDs for each token in the biomolecule.
    :param chemid: An array of chemical IDs for each token in the biomolecule.
    :return: A boolean indicating whether the symmetry IDs are mismatched.
    """
    assert len(asym_ids) == len(entity_ids) == len(sym_ids) == len(chemid), (
        f"The number of asymmetric unit IDs ({len(asym_ids)}), entity IDs ({len(entity_ids)}), symmetry IDs ({len(sym_ids)}), and chemical IDs ({len(chemid)}) do not match. "
        "Please ensure that these input features are correctly paired."
    )

    # Create a combined array of tuples (asym_id, entity_id, sym_id, index)
    combined = np.array(list(zip(asym_ids, entity_ids, sym_ids, range(len(entity_ids)))))

    # Group by entity_id
    grouped_by_entity = defaultdict(list)
    for entity, group in groupby(combined, key=lambda x: x[1]):
        grouped_by_entity[entity].extend(list(group))

    # Compare regions with the same entity_id but different sym_id
    for entity, group in grouped_by_entity.items():
        # Group by sym_id within each entity_id group
        grouped_by_sym = defaultdict(list)
        for _, _, sym, idx in group:
            grouped_by_sym[sym].append(idx)

        # Compare chemid sequences for the asym_id regions of different sym_id groups within the same entity_id group
        sym_ids_keys = list(grouped_by_sym.keys())
        for i in range(len(sym_ids_keys)):
            for j in range(i + 1, len(sym_ids_keys)):
                indices1 = grouped_by_sym[sym_ids_keys[i]]
                indices2 = grouped_by_sym[sym_ids_keys[j]]
                indices1_asym_ids = np.unique(asym_ids[indices1])
                indices2_asym_ids = np.unique(asym_ids[indices2])
                chemid_seq1 = chemid[np.isin(asym_ids, indices1_asym_ids)]
                chemid_seq2 = chemid[np.isin(asym_ids, indices2_asym_ids)]
                if len(chemid_seq1) != len(chemid_seq2) or not np.array_equal(
                    chemid_seq1, chemid_seq2
                ):
                    return True

    return False


@typecheck
def load_msa_from_msa_dir(
    msa_dir: str | None,
    file_id: str,
    chain_id_to_residue: Dict[str, Dict[str, List[int]]],
    max_msas_per_chain: int | None = None,
    randomly_truncate: bool = True,
    raise_missing_exception: bool = False,
    verbose: bool = False,
) -> FeatureDict:
    """Load MSA from a directory containing MSA files."""
    if (not exists(msa_dir) or not os.path.exists(msa_dir)) and raise_missing_exception:
        raise FileNotFoundError(f"{msa_dir} does not exist.")
    elif not exists(msa_dir) or not os.path.exists(msa_dir):
        if verbose:
            logger.warning(f"{msa_dir} does not exist. Skipping MSA loading by returning `Nones`.")
        return {}

    msas = {}
    for chain_id in chain_id_to_residue:
        msa_fpaths = glob.glob(os.path.join(msa_dir, f"{file_id}{chain_id}_*.a3m"))

        if not msa_fpaths:
            msas[chain_id] = None
            continue

        # NOTE: A single chain-specific MSA file contains alignments for all polymer residues in the chain,
        # but the chain's ligands are not included in the MSA file and therefore must be manually inserted
        # into the MSAs as unknown amino acid residues.
        assert len(msa_fpaths) == 1, (
            f"{len(msa_fpaths)} MSA files found for chain {chain_id} of file {file_id}. "
            "Please ensure that one MSA file is present for each chain."
        )
        msa_fpath = msa_fpaths[0]
        msa_type = os.path.splitext(os.path.basename(msa_fpath))[0].split("_")[-1]

        with open(msa_fpath, "r") as f:
            msa = f.read()
            msa = msa_parsing.parse_a3m(msa, msa_type)
            msa = (
                (
                    msa.random_truncate(max_msas_per_chain)
                    if randomly_truncate
                    else msa.truncate(max_msas_per_chain)
                )
                if exists(max_msas_per_chain)
                else msa
            )
            msas[chain_id] = msa

    features = make_msa_features(msas, chain_id_to_residue)
    features = make_msa_mask(features)

    return features


@typecheck
def load_templates_from_templates_dir(
    templates_dir: str | None,
    file_id: str,
    raise_missing_exception: bool = False,
    verbose: bool = False,
) -> FeatureDict:
    """Load templates from a directory containing template PDB mmCIF files."""
    if (
        not exists(templates_dir) or not os.path.exists(templates_dir)
    ) and raise_missing_exception:
        raise FileNotFoundError(f"{templates_dir} does not exist.")
    elif not exists(templates_dir) or not os.path.exists(templates_dir):
        if verbose:
            logger.warning(
                f"{templates_dir} does not exist. Skipping template loading by returning `Nones`."
            )
        return {}

    # template_fpath = os.path.join(templates_dir, f"{file_id}.pdb")
    # with open(template_fpath, "r") as f:
    #     template = f.read()

    # template = template_parsing.parse_pdb(template)
    # features = make_template_features([template])
    # features = make_template_mask(features)

    # return features

    return {}


@typecheck
def pdb_input_to_molecule_input(
    pdb_input: PDBInput,
    biomol: Biomolecule | None = None,
    verbose: bool = False,
) -> MoleculeInput:
    """Convert a PDBInput to a MoleculeInput."""
    i = pdb_input

    filepath = i.mmcif_filepath
    file_id = os.path.splitext(os.path.basename(filepath))[0] if exists(filepath) else None
    resolution = i.resolution

    # acquire a `Biomolecule` object for the given `PDBInput`

    if not exists(biomol) and exists(i.biomol):
        biomol = i.biomol
    else:
        # construct a `Biomolecule` object from the input PDB mmCIF file

        assert os.path.exists(filepath), f"PDB input file `{filepath}` does not exist."

        mmcif_object = mmcif_parsing.parse_mmcif_object(
            filepath=filepath,
            file_id=file_id,
        )
        mmcif_resolution = extract_mmcif_metadata_field(mmcif_object, "resolution")
        biomol = (
            _from_mmcif_object(mmcif_object)
            if "assembly" in file_id
            else get_assembly(_from_mmcif_object(mmcif_object))
        )

        if not exists(resolution) and exists(mmcif_resolution):
            resolution = mmcif_resolution

    # record PDB resolution value if available

    resolution = tensor(resolution) if exists(resolution) else None

    # map (sampled) chain IDs to indices prior to cropping

    chains = None

    if exists(i.chains):
        chain_id_1, chain_id_2 = i.chains
        chain_id_to_idx = {
            chain_id: chain_idx
            for (chain_id, chain_idx) in zip(biomol.chain_id, biomol.chain_index)
        }
        # NOTE: we have to manually nullify a chain ID value
        # e.g., if an empty string is passed in as a "null" chain ID
        if chain_id_1:
            chain_id_1 = chain_id_to_idx[chain_id_1]
        else:
            chain_id_1 = None
        if chain_id_2:
            chain_id_2 = chain_id_to_idx[chain_id_2]
        else:
            chain_id_2 = None
        chains = (chain_id_1, chain_id_2)

    # construct multiple sequence alignment (MSA) and template features prior to cropping

    # retrieve MSA metadata from the `Biomolecule` object
    biomol_chain_ids = list(dict.fromkeys(biomol.chain_id.tolist()))  # NOTE: we must maintain the order of unique chain IDs

    residue_index = torch.from_numpy(biomol.residue_index) - 1  # NOTE: `Biomolecule.residue_index` is 1-based originally
    num_tokens = len(biomol.atom_mask)

    chain_id_to_residue = {
        chain_id: {
            "chemtype": biomol.chemtype[biomol.chain_id == chain_id].tolist(),
            "residue_index": residue_index[biomol.chain_id == chain_id].tolist(),
        }
        for chain_id in biomol_chain_ids
    }

    msa_features = load_msa_from_msa_dir(
        # NOTE: if MSAs are not locally available, `Nones` will be used
        i.msa_dir, file_id, chain_id_to_residue, max_msas_per_chain=i.max_msas_per_chain
    )

    msa = msa_features.get("msa")
    msa_col_mask = msa_features.get("msa_mask")
    msa_row_mask = msa_features.get("msa_row_mask")

    # collect additional MSA and token features
    # 0: has_deletion (msa)
    # 1: deletion_value (msa)
    # 2: profile (token)
    # 3: deletion_mean (token)

    additional_msa_feats = None
    additional_token_feats = None

    num_msas = len(msa) if exists(msa) else 1

    if exists(msa):
        assert (
            msa.shape[-1] == num_tokens
        ), f"The number of tokens in the MSA ({msa.shape[-1]}) does not match the number of tokens in the biomolecule ({num_tokens}). "

        has_deletion = torch.clip(msa_features["deletion_matrix"], 0.0, 1.0)
        deletion_value = torch.atan(msa_features["deletion_matrix"] / 3.0) * (2.0 / torch.pi)

        additional_msa_feats = torch.stack(
            [
                has_deletion,
                deletion_value,
            ],
            dim=-1,
        )

        # NOTE: assumes each aligned sequence has the same mask values
        profile_msa_mask = torch.repeat_interleave(msa_col_mask[None, ...], len(msa), dim=0)
        msa_sum = (profile_msa_mask[:, :, None] * make_one_hot(msa, NUM_MSA_ONE_HOT)).sum(0)
        mask_counts = 1e-6 + profile_msa_mask.sum(0)

        profile = msa_sum / mask_counts[:, None]
        deletion_mean = torch.atan(msa_features["deletion_matrix"].mean(0) / 3.0) * (
            2.0 / torch.pi
        )

        additional_token_feats = torch.cat(
            [
                profile,
                deletion_mean[:, None],
            ],
            dim=-1,
        )

        # convert the MSA into a one-hot representation
        msa = make_one_hot(msa, NUM_MSA_ONE_HOT)
        msa_row_mask = msa_row_mask.bool()

    # TODO: retrieve templates for each chain
    # NOTE: if they are not locally available, `Nones` will be used
    template_features = load_templates_from_templates_dir(i.templates_dir, file_id)

    templates = template_features.get("templates")
    template_mask = template_features.get("template_mask")

    # crop the `Biomolecule` object during training only

    if i.training:
        assert exists(
            i.cropping_config
        ), "A cropping configuration must be provided during training."
        try:
            assert exists(i.chains), "Chain IDs must be provided for cropping during training."
            chain_id_1, chain_id_2 = i.chains

            biomol, chain_ids_and_lengths, crop_masks = biomol.crop(
                contiguous_weight=i.cropping_config["contiguous_weight"],
                spatial_weight=i.cropping_config["spatial_weight"],
                spatial_interface_weight=i.cropping_config["spatial_interface_weight"],
                n_res=i.cropping_config["n_res"],
                chain_1=chain_id_1 if chain_id_1 else None,
                chain_2=chain_id_2 if chain_id_2 else None,
            )

            # retrieve cropped residue and token metadata
            residue_index = torch.from_numpy(biomol.residue_index) - 1  # NOTE: `Biomolecule.residue_index` is 1-based originally
            num_tokens = len(biomol.atom_mask)

            # update MSA and template features after cropping
            chain_id_sorted_indices = get_sorted_tuple_indices(chain_ids_and_lengths, biomol_chain_ids)
            sorted_crop_mask = np.concatenate([crop_masks[idx] for idx in chain_id_sorted_indices])

            # crop MSA features
            if exists(msa):
                msa = msa[:, sorted_crop_mask]

                additional_token_feats = additional_token_feats[sorted_crop_mask]
                additional_msa_feats = additional_msa_feats[:, sorted_crop_mask]

            # TODO: crop template features

        except Exception as e:
            raise ValueError(f"Failed to crop the biomolecule for input {file_id} due to: {e}")

    # retrieve features directly available within the `Biomolecule` object

    # create unique chain-residue index pairs to identify the first atom of each residue
    chain_residue_index = np.array(list(zip(biomol.chain_index, biomol.residue_index)))
    _, unique_chain_residue_indices = np.unique(chain_residue_index, axis=0, return_index=True)

    # retrieve molecule_ids from the `Biomolecule` object, where here it is the mapping of 33 possible residue types
    # `proteins (20) | unknown protein (1) | rna (4) | unknown RNA (1) | dna (4) | unknown DNA (1) | gap (1) | metal ion (1)`,
    # where ligands are mapped to the unknown protein category (i.e., residue index 20)
    # NOTE: below, we will install values for our new (dedicated) type for metal ions
    molecule_ids = torch.from_numpy(biomol.restype)

    # retrieve is_molecule_types from the `Biomolecule` object, which is a boolean tensor of shape [*, 5]
    # is_protein | is_rna | is_dna | is_ligand | is_metal_ion
    # this is needed for their special diffusion loss
    # NOTE: below, we will install values for our new (dedicated) one-hot class for metal ions
    n_one_hot = IS_MOLECULE_TYPES
    is_molecule_types = F.one_hot(torch.from_numpy(biomol.chemtype), num_classes=n_one_hot).bool()

    # manually derive remaining features using the `Biomolecule` object

    # extract chain sequences and chemical types from the `Biomolecule` object
    chem_comp_table = {comp.id: comp for comp in biomol.chem_comp_table}
    chem_comp_details = [chem_comp_table[chemid] for chemid in biomol.chemid]
    chain_seqs, chain_chem_types = extract_chain_sequences_from_biomolecule_chemical_components(
        biomol,
        chem_comp_details,
    )

    # retrieve RDKit template molecules for the residues of each chain,
    # and insert the input atom coordinates into the template molecules
    molecules, molecule_types = extract_template_molecules_from_biomolecule_chains(
        biomol,
        chain_seqs,
        chain_chem_types,
        verbose=verbose,
    )

    # collect pooling lengths and atom-wise molecule types for each molecule,
    # along with a token-wise one-hot tensor indicating whether each molecule is modified
    # and, if so, which type of modification it has (e.g., peptide vs. RNA modification)
    molecule_idx = 0
    token_pool_lens = []
    molecule_atom_types = []
    is_molecule_mod = []
    for mol, mol_type in zip(molecules, molecule_types):
        num_atoms = mol.GetNumAtoms()
        is_mol_mod_type = [False for _ in range(DEFAULT_NUM_MOLECULE_MODS)]
        if is_atomized_residue(mol_type):
            # NOTE: in the paper, they treat each atom of the ligand and modified polymer residues as a token
            token_pool_lens.extend([1] * num_atoms)
            molecule_atom_types.extend([mol_type] * num_atoms)

            molecule_type_row_idx = slice(molecule_idx, molecule_idx + num_atoms)

            # NOTE: we reset all type annotations e.g., since ions are initially considered ligands
            is_molecule_types[molecule_type_row_idx] = False

            if mol_type == "ligand" and num_atoms == 1:
                # NOTE: we manually set the molecule ID of ions to a dedicated category
                molecule_ids[molecule_idx] = MOLECULE_METAL_ION_ID
                is_mol_type_index = IS_METAL_ION_INDEX
            elif mol_type == "ligand":
                is_mol_type_index = IS_LIGAND_INDEX
            elif mol_type == "mod_protein":
                is_mol_type_index = IS_PROTEIN_INDEX
                is_mol_mod_type_index = 0
            elif mol_type == "mod_rna":
                is_mol_type_index = IS_RNA_INDEX
                is_mol_mod_type_index = 1
            elif mol_type == "mod_dna":
                is_mol_type_index = IS_DNA_INDEX
                is_mol_mod_type_index = 2
            else:
                raise ValueError(f"Unrecognized molecule type: {mol_type}")

            is_molecule_types[molecule_type_row_idx, is_mol_type_index] = True

            if "mod" in mol_type:
                is_mol_mod_type[is_mol_mod_type_index] = True
            is_molecule_mod.extend([is_mol_mod_type] * num_atoms)

            molecule_idx += num_atoms
        else:
            token_pool_lens.append(num_atoms)
            molecule_atom_types.append(mol_type)
            is_molecule_mod.append(is_mol_mod_type)
            molecule_idx += 1

    # collect frame, token center, distogram, and source-target atom indices for each token
    atom_indices_for_frame = []
    molecule_atom_indices = []
    distogram_atom_indices = []
    src_tgt_atom_indices = []

    current_atom_index = 0
    current_res_index = -1

    for mol_type, chemid, res_index in zip(
        molecule_atom_types,
        biomol.chemid,
        biomol.residue_index,
    ):
        residue_constants = get_residue_constants(
            mol_type.replace("protein", "peptide").replace("mod_", "")
        )

        if mol_type == "protein":
            entry = HUMAN_AMINO_ACIDS[residue_constants.restype_3to1.get(chemid, "X")]
        elif mol_type == "rna":
            entry = RNA_NUCLEOTIDES[residue_constants.restype_3to1.get(chemid, "X")]
        elif mol_type == "dna":
            entry = DNA_NUCLEOTIDES[residue_constants.restype_3to1.get(chemid, "X")]

        if is_atomized_residue(mol_type):
            # collect indices for each ligand and modified polymer residue token (i.e., atom)
            if current_res_index == res_index:
                current_atom_index += 1
            else:
                current_atom_index = 0
                current_res_index = res_index

            atom_indices_for_frame.append(None)
            molecule_atom_indices.append(current_atom_index)
            distogram_atom_indices.append(current_atom_index)
            # NOTE: ligand and modified polymer residue tokens do not have source-target atom indices
        else:
            # collect indices for each polymer residue token
            atom_indices_for_frame.append(entry["three_atom_indices_for_frame"])
            molecule_atom_indices.append(entry["token_center_atom_idx"])
            distogram_atom_indices.append(entry["distogram_atom_idx"])
            src_tgt_atom_indices.append([entry["first_atom_idx"], entry["last_atom_idx"]])

    molecule_atom_indices = tensor(molecule_atom_indices)
    distogram_atom_indices = tensor(distogram_atom_indices)

    # constructing the additional_molecule_feats
    # which is in turn used to derive relative positions

    # residue_index - an arange that restarts at 1 for each chain - reuse biomol.residue_index here
    # token_index   - just an arange
    # asym_id       - unique id for each chain of a biomolecule - reuse chain_index here
    # entity_id     - unique id for each biomolecule sequence
    # sym_id        - unique id for each chain of the same biomolecule sequence

    # entity ids

    unrepeated_entity_sequences = defaultdict(int)
    for entity_sequence in chain_seqs:
        if entity_sequence in unrepeated_entity_sequences:
            continue
        unrepeated_entity_sequences[entity_sequence] = len(unrepeated_entity_sequences)

    entity_idx = 0
    entity_id_counts = []
    unrepeated_entity_ids = []
    for entity_sequence, chain_chem_type in zip(chain_seqs, chain_chem_types):
        entity_mol = molecules[entity_idx]
        entity_len = (
            entity_mol.GetNumAtoms()
            if is_atomized_residue(chain_chem_type)
            else len(entity_sequence)
        )
        entity_idx += 1 if is_atomized_residue(chain_chem_type) else len(entity_sequence)

        entity_id_counts.append(entity_len)
        unrepeated_entity_ids.append(unrepeated_entity_sequences[entity_sequence])

    entity_ids = repeat_interleave(tensor(unrepeated_entity_ids), tensor(entity_id_counts))

    # sym ids

    unrepeated_sym_ids = []
    unrepeated_sym_sequences = defaultdict(int)
    for entity_sequence in chain_seqs:
        unrepeated_sym_ids.append(unrepeated_sym_sequences[entity_sequence])
        if entity_sequence in unrepeated_sym_sequences:
            unrepeated_sym_sequences[entity_sequence] += 1

    sym_ids = repeat_interleave(tensor(unrepeated_sym_ids), tensor(entity_id_counts))

    # concat for all of additional_molecule_feats

    additional_molecule_feats = torch.stack(
        (
            residue_index,
            torch.arange(num_tokens),
            torch.from_numpy(biomol.chain_index),
            entity_ids,
            sym_ids,
        ),
        dim=-1,
    )

    # construct token bonds, which will be linearly connected for proteins
    # and nucleic acids, but for ligands and modified polymer residues
    # will have their atomic bond matrix (as ligands and modified polymer
    # residues are atom resolution)
    polymer_offset = 0
    ligand_offset = 0
    token_bonds = torch.zeros(num_tokens, num_tokens).bool()

    for chain_seq, chain_chem_type in zip(
        chain_seqs,
        chain_chem_types,
    ):
        if is_atomized_residue(chain_chem_type):
            # construct ligand and modified polymer chain token bonds

            coordinates = []
            updates = []

            ligand = molecules[ligand_offset]
            num_atoms = ligand.GetNumAtoms()
            has_bond = torch.zeros(num_atoms, num_atoms).bool()

            for bond in ligand.GetBonds():
                atom_start_index = bond.GetBeginAtomIdx()
                atom_end_index = bond.GetEndAtomIdx()

                coordinates.extend(
                    [
                        [atom_start_index, atom_end_index],
                        [atom_end_index, atom_start_index],
                    ]
                )

                updates.extend([True, True])

            coordinates = tensor(coordinates).long()
            updates = tensor(updates).bool()

            has_bond = einx.set_at("[h w], c [2], c -> [h w]", has_bond, coordinates, updates)

            row_col_slice = slice(polymer_offset, polymer_offset + num_atoms)
            token_bonds[row_col_slice, row_col_slice] = has_bond

            polymer_offset += num_atoms
            ligand_offset += 1
        else:
            # construct polymer chain token bonds

            chain_len = len(chain_seq)
            eye = torch.eye(chain_len)

            row_col_slice = slice(polymer_offset, polymer_offset + chain_len - 1)
            token_bonds[row_col_slice, row_col_slice] = (eye[1:, :-1] + eye[:-1, 1:]) > 0
            polymer_offset += chain_len
            ligand_offset += chain_len

    # ensure mmCIF polymer-ligand (i.e., protein/RNA/DNA-ligand) and ligand-ligand bonds
    # (and bonds less than 2.4 Å) are installed in `MoleculeInput` during training only
    # per the AF3 supplement (Table 5, `token_bonds`)
    bond_atom_indices = defaultdict(int)
    for bond in biomol.bonds:
        # ascertain whether homomeric (e.g., bonded ligand) symmetry is preserved,
        # which determines whether or not we use the mmCIF bond inputs (AF3 Section 5.1)
        if not i.training or find_mismatched_symmetry(
            biomol.chain_index,
            entity_ids.numpy(),
            sym_ids.numpy(),
            biomol.chemid,
        ):
            continue

        # determine bond type

        # NOTE: in this context, modified polymer residues will be treated as ligands
        ptnr1_is_polymer = any(
            bond.ptnr1_auth_comp_id in rc.restype_3to1
            for rc in {amino_acid_constants, rna_constants, dna_constants}
        )
        ptnr2_is_polymer = any(
            bond.ptnr2_auth_comp_id in rc.restype_3to1
            for rc in {amino_acid_constants, rna_constants, dna_constants}
        )
        ptnr1_is_ligand = not ptnr1_is_polymer
        ptnr2_is_ligand = not ptnr2_is_polymer
        is_polymer_ligand_bond = (ptnr1_is_polymer and ptnr2_is_ligand) or (
            ptnr1_is_ligand and ptnr2_is_polymer
        )
        is_ligand_ligand_bond = ptnr1_is_ligand and ptnr2_is_ligand

        # conditionally install bond

        if (
            is_polymer_ligand_bond
            or is_ligand_ligand_bond
            or (mmcif_parsing._is_set(bond.pdbx_dist_value) and float(bond.pdbx_dist_value) < 2.4)
        ):
            ptnr1_atom_id = (
                f"{bond.ptnr1_auth_asym_id}:{bond.ptnr1_auth_seq_id}:{bond.ptnr1_label_atom_id}"
            )
            ptnr2_atom_id = (
                f"{bond.ptnr2_auth_asym_id}:{bond.ptnr2_auth_seq_id}:{bond.ptnr2_label_atom_id}"
            )
            try:
                row_idx = get_token_index_from_composite_atom_id(
                    biomol,
                    bond.ptnr1_auth_asym_id,
                    int(bond.ptnr1_auth_seq_id),
                    bond.ptnr1_label_atom_id,
                    bond_atom_indices[ptnr1_atom_id],
                    ptnr1_is_polymer,
                )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Could not find a matching token index for token1 {ptnr1_atom_id} due to: {e}. "
                        "Skipping installing the current bond associated with this token."
                    )
                continue
            try:
                col_idx = get_token_index_from_composite_atom_id(
                    biomol,
                    bond.ptnr2_auth_asym_id,
                    int(bond.ptnr2_auth_seq_id),
                    bond.ptnr2_label_atom_id,
                    bond_atom_indices[ptnr2_atom_id],
                    ptnr2_is_polymer,
                )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Could not find a matching token index for token2 {ptnr1_atom_id} due to: {e}. "
                        "Skipping installing the current bond associated with this token."
                    )
                continue
            token_bonds[row_idx, col_idx] = True
            token_bonds[col_idx, row_idx] = True
            bond_atom_indices[ptnr1_atom_id] += 1
            bond_atom_indices[ptnr2_atom_id] += 1

    # handle missing atom indices
    missing_atom_indices = None
    missing_token_indices = None
    molecules_missing_atom_indices = [
        [int(idx) for idx in mol.GetProp("missing_atom_indices").split(",") if idx]
        for mol in molecules
    ]

    missing_atom_indices = []
    missing_token_indices = []

    for mol_miss_atom_indices, mol, mol_type in zip(
        molecules_missing_atom_indices, molecules, molecule_types
    ):
        mol_miss_atom_indices = default(mol_miss_atom_indices, [])
        mol_miss_atom_indices = tensor(mol_miss_atom_indices, dtype=torch.long)

        missing_atom_indices.append(mol_miss_atom_indices)
        if is_atomized_residue(mol_type):
            missing_token_indices.extend([mol_miss_atom_indices for _ in range(mol.GetNumAtoms())])
        else:
            missing_token_indices.append(mol_miss_atom_indices)

    assert len(molecules) == len(missing_atom_indices)
    assert len(missing_token_indices) == num_tokens

    mol_total_atoms = sum([mol.GetNumAtoms() for mol in molecules])
    num_missing_atom_indices = sum(
        len(mol_miss_atom_indices) for mol_miss_atom_indices in missing_atom_indices
    )
    num_present_atoms = mol_total_atoms - num_missing_atom_indices
    assert num_present_atoms == int(biomol.atom_mask.sum())

    # handle `atom_indices_for_frame` for the PAE

    atom_indices_for_frame = tensor(
        [default(indices, (-1, -1, -1)) for indices in atom_indices_for_frame]
    )

    # build offsets for all indices

    # derive `atom_lens` based on `one_token_per_atom`, for ligands and modified biomolecules
    atoms_per_molecule = tensor([mol.GetNumAtoms() for mol in molecules])
    ones = torch.ones_like(atoms_per_molecule)

    # `is_molecule_mod` can either be
    # 1. Bool['n'], in which case it will only be used for determining `one_token_per_atom`, or
    # 2. Bool['n num_mods'], where it will be passed to Alphafold3 for molecule modification embeds
    is_molecule_mod = tensor(is_molecule_mod)
    is_molecule_any_mod = False

    if is_molecule_mod.ndim == 2:
        is_molecule_any_mod = is_molecule_mod[unique_chain_residue_indices].any(dim=-1)
    else:
        is_molecule_any_mod = is_molecule_mod[unique_chain_residue_indices]

    # get `one_token_per_atom`
    # default to what the paper did, which is ligands and any modified biomolecule
    is_ligand = is_molecule_types[unique_chain_residue_indices][..., IS_LIGAND_INDEX]
    one_token_per_atom = is_ligand | is_molecule_any_mod

    assert len(molecules) == len(one_token_per_atom)

    # derive the number of repeats needed to expand molecule lengths to token lengths
    token_repeats = torch.where(one_token_per_atom, atoms_per_molecule, ones)

    # craft offsets for all atom indices
    atom_indices_offsets = repeat_interleave(
        exclusive_cumsum(atoms_per_molecule), token_repeats, dim=0
    )

    # offset only positive atom indices
    distogram_atom_indices = offset_only_positive(distogram_atom_indices, atom_indices_offsets)
    molecule_atom_indices = offset_only_positive(molecule_atom_indices, atom_indices_offsets)
    atom_indices_for_frame = offset_only_positive(
        atom_indices_for_frame, atom_indices_offsets[..., None]
    )

    # construct atom positions from template molecules after instantiating their 3D conformers
    atom_pos = torch.from_numpy(
        np.concatenate([mol.GetConformer().GetPositions() for mol in molecules]).astype(np.float32)
    )
    num_atoms = atom_pos.shape[0]

    # create atom_parent_ids using the `Biomolecule` object, which governs in the atom
    # encoder / decoder which atom attends to which, where a design choice is made such
    # that mmCIF author chain indices are directly adopted to group atoms belonging to
    # the same (author-denoted) chain
    atom_parent_ids = tensor(
        [
            biomol.chain_index[unique_chain_residue_indices][res_index]
            for res_index in range(len(molecules))
            for _ in range(molecules[res_index].GetNumAtoms())
        ]
    )

    # craft experimentally resolved labels per the AF2 supplement's Section 1.9.10

    resolved_labels = None

    if exists(resolution):
        is_resolved_label = ((resolution >= 0.1) & (resolution <= 3.0)).item()
        resolved_labels = torch.full((num_atoms,), is_resolved_label, dtype=torch.long)

    # create molecule input

    molecule_input = MoleculeInput(
        molecules=molecules,
        molecule_token_pool_lens=token_pool_lens,
        molecule_ids=molecule_ids,
        additional_molecule_feats=additional_molecule_feats,
        is_molecule_types=is_molecule_types,
        src_tgt_atom_indices=src_tgt_atom_indices,
        token_bonds=token_bonds,
        is_molecule_mod=is_molecule_mod,
        molecule_atom_indices=molecule_atom_indices,
        distogram_atom_indices=distogram_atom_indices,
        atom_indices_for_frame=atom_indices_for_frame,
        missing_atom_indices=missing_atom_indices,
        missing_token_indices=missing_token_indices,
        atom_parent_ids=atom_parent_ids,
        additional_msa_feats=default(additional_msa_feats, torch.zeros(num_msas, num_tokens, 2)),
        additional_token_feats=default(additional_token_feats, torch.zeros(num_tokens, 33)),
        templates=templates,
        msa=msa,
        atom_pos=atom_pos,
        template_mask=template_mask,
        msa_mask=msa_row_mask,
        resolved_labels=resolved_labels,
        resolution=resolution,
        chains=chains,
        filepath=filepath,
        add_atom_ids=i.add_atom_ids,
        add_atompair_ids=i.add_atompair_ids,
        directed_bonds=i.directed_bonds,
        extract_atom_feats_fn=i.extract_atom_feats_fn,
        extract_atompair_feats_fn=i.extract_atompair_feats_fn,
    )

    return molecule_input

# datasets

# PDB dataset that returns a PDBInput based on folder

class PDBDataset(Dataset):
    """A PyTorch Dataset for PDB mmCIF files."""

    @typecheck
    def __init__(
        self,
        folder: str | Path,
        sampler: WeightedPDBSampler | None = None,
        sample_type: Literal["default", "clustered"] = "default",
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        spatial_interface_weight: float = 0.4,
        crop_size: int = 384,
        training: bool | None = None,  # extra training flag placed by Alex on PDBInput
        sample_only_pdb_ids: Set[str] | None = None,
        **pdb_input_kwargs,
    ):
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.exists() and folder.is_dir(), f"{str(folder)} does not exist for PDBDataset"
        self.folder = folder

        self.sampler = sampler
        self.sample_type = sample_type
        self.training = training
        self.sample_only_pdb_ids = sample_only_pdb_ids
        self.pdb_input_kwargs = pdb_input_kwargs

        self.cropping_config = {
            "contiguous_weight": contiguous_weight,
            "spatial_weight": spatial_weight,
            "spatial_interface_weight": spatial_interface_weight,
            "n_res": crop_size,
        }

        # subsample mmCIF files to those that have a valid (post-filtering) association with a chain/interface cluster

        if exists(self.sampler):
            sampler_pdb_ids = set(self.sampler.mappings.get_column("pdb_id").to_list())
            self.files = {
                os.path.splitext(os.path.basename(filepath.name))[0]: filepath
                for filepath in folder.glob(os.path.join("**", "*.cif"))
                if os.path.splitext(os.path.basename(filepath.name))[0] in sampler_pdb_ids
            }
        else:
            self.files = {
                os.path.splitext(os.path.basename(file.name))[0]: file
                for file in folder.glob(os.path.join("**", "*.cif"))
            }

        if exists(sample_only_pdb_ids):
            assert exists(self.sampler), "A sampler must be provided to use `sample_only_pdb_ids`."
            assert all(
                pdb_id in sampler_pdb_ids for pdb_id in sample_only_pdb_ids
            ), "Some PDB IDs in `sample_only_pdb_ids` are not present in the dataset's sampler mappings."

        assert len(self) > 0, f"No valid mmCIFs / PDBs found at {str(folder)}"

    def __len__(self):
        """Return the number of PDB mmCIF files in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int | str) -> PDBInput:
        """Return a PDBInput object for the specified index."""
        sampled_id = None

        if exists(self.sampler):
            sample_fn = (
                self.sampler.cluster_based_sample
                if self.sample_type == "clustered"
                else self.sampler.sample
            )
            (sampled_id,) = sample_fn(1)

            # ensure that the sampled PDB ID is in the specified set of PDB IDs from which to sample

            if exists(self.sample_only_pdb_ids):
                while sampled_id[0] not in self.sample_only_pdb_ids:
                    (sampled_id,) = sample_fn(1)

        pdb_id, chain_id_1, chain_id_2 = None, None, None

        if exists(sampled_id):
            pdb_id, chain_id_1, chain_id_2 = sampled_id
            mmcif_filepath = self.files.get(pdb_id, None)

        elif isinstance(idx, int):
            pdb_id, mmcif_filepath = [*self.files.items()][idx]

        elif isinstance(idx, str):
            pdb_id = idx
            mmcif_filepath = self.files.get(pdb_id, None)

        # get the mmCIF file corresponding to the sampled structure

        if not exists(mmcif_filepath):
            raise FileNotFoundError(f"mmCIF file for PDB ID {pdb_id} not found.")
        if not os.path.exists(mmcif_filepath):
            raise FileNotFoundError(f"mmCIF file {mmcif_filepath} not found.")

        cropping_config = None

        if self.training:
            cropping_config = self.cropping_config

        pdb_input = PDBInput(
            mmcif_filepath=str(mmcif_filepath),
            chains=(chain_id_1, chain_id_2),
            cropping_config=cropping_config,
            training=self.training,
            **self.pdb_input_kwargs,
        )

        return pdb_input

# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    AtomInput: compose(
        default_none_fields_atom_input
    ),
    MoleculeInput: compose(
        molecule_to_atom_input,
        default_none_fields_atom_input
    ),
    Alphafold3Input: compose(
        alphafold3_input_to_molecule_lengthed_molecule_input,
        molecule_lengthed_molecule_input_to_atom_input,
        default_none_fields_atom_input
    ),
    PDBInput: compose(
        pdb_input_to_molecule_input,
        molecule_to_atom_input,
        default_none_fields_atom_input
    ),
}

# function for extending the config

@typecheck
def register_input_transform(
    input_type: Type,
    fn: Callable[[Any], AtomInput]
):
    if input_type in INPUT_TO_ATOM_TRANSFORM:
        logger.warning(f'{input_type} is already registered, but overwriting')

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
