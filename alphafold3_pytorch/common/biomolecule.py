"""A generic `Biomolecule` data structure for parsing macromolecular structures."""

import collections
import dataclasses
import functools
import io
import random
from functools import partial
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Model import Model

from alphafold3_pytorch.common import (
    amino_acid_constants,
    dna_constants,
    ligand_constants,
    mmcif_metadata,
    rna_constants,
)
from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.tensor_typing import IntType, typecheck
from alphafold3_pytorch.utils.data_utils import deep_merge_dicts, is_polymer
from alphafold3_pytorch.utils.utils import exists, np_mode

MMCIF_PREFIXES_TO_DROP_POST_PARSING = [
    "_atom_site.",
    "_atom_type.",
    "_chem_comp.",
    "_entity.",
    "_entity_poly.",
    "_entity_poly_seq.",
    "_pdbx_branch_scheme.",
    "_pdbx_nonpoly_scheme.",
    "_pdbx_poly_seq_scheme.",
    "_pdbx_struct_assembly.",
    "_pdbx_struct_assembly_gen.",
    "_struct_asym.",
    "_struct_conn.",
]
MMCIF_PREFIXES_TO_DROP_POST_AF3 = MMCIF_PREFIXES_TO_DROP_POST_PARSING + [
    "_audit_author.",
    "_citation.",
    "_citation_author.",
]


@dataclasses.dataclass(frozen=True)
class Biomolecule:
    """Biomolecule structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, e.g., the first three are N, CA, CB for
    # amino acid residues.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Name of each residue-representative atom as a string,
    # which matches the number of (pseudo)residues (A.K.A. tokens).
    atom_name: np.ndarray  # [num_res]

    # Amino-acid or nucleotide type for each residue represented as an integer
    # between 0 and 31, where:
    # 20 represents the the unknown amino acid 'X';
    # 25 represents the unknown RNA nucleotide `N`;
    # 30 represents the unknown DNA nucleotide `DN`;
    # and 31 represents the gap token `-`.
    restype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the biomolecule to which this
    # residue belongs.
    chain_index: np.ndarray  # [num_res]

    # A string representation of the chain in the biomolecule to which this
    # residue belongs.
    chain_id: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of the atoms of each residue
    # (in sq. angstroms units), representing the displacement of the
    # residue's atoms from their ground truth mean values.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chemical ID of each amino-acid, nucleotide, or ligand residue represented
    # as a string. This is primarily used to record a ligand residue's name
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    chemid: np.ndarray  # [num_res]

    # Chemical type of each amino-acid, nucleotide, or ligand residue represented
    # as an integer between 0 and 3. This is used to determine whether a residue is
    # a protein (0), RNA (1), DNA (2), or ligand (3) residue.
    chemtype: np.ndarray  # [num_res]

    # Bonds between atoms in the biomolecule.
    bonds: Optional[List[mmcif_parsing.Bond]]  # [num_bonds]

    # Atom name-chain ID-residue ID tuples for each (e.g. ligand) "pseudoresidue" of each residue in each chain.
    # This is used to group "pseudoresidues" (e.g., ligand atoms) by parent residue.
    unique_res_atom_names: Optional[
        List[Tuple[List[List[str]], str, int]]
    ]  # [num_res, num_pseudoresidues, num_atoms]

    # Mapping from (original) author chain ID-residue name-residue ID (CRI) tuples to (new) author CRI tuples.
    author_cri_to_new_cri: Dict[Tuple[str, str, int], Tuple[str, str, int]]  # [num_res]

    # Chemical component details of each residue as a unique `ChemComp` object.
    # This is used to determine the biomolecule's unique chemical IDs, names, types, etc.
    # N.b., this is primarily used to record chemical component metadata
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    chem_comp_table: Set[mmcif_parsing.ChemComp]  # [num_unique_chem_comp]

    # Mapping from entity ID string to chain ID strings.
    entity_to_chain: Dict[str, List[str]]  # [1]

    # Mapping from (internal) mmCIF chain ID string to integer author chain ID.
    mmcif_to_author_chain: Dict[str, int]  # [1]

    # Raw mmCIF metadata dictionary parsed for the biomolecule using Biopython.
    # N.b., this is primarily used to retain mmCIF assembly metadata
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    mmcif_metadata: Dict[str, Any]  # [1]

    def __add__(self, other: "Biomolecule") -> "Biomolecule":
        """Merges two `Biomolecule` instances."""
        return Biomolecule(
            atom_positions=np.concatenate([self.atom_positions, other.atom_positions], axis=0),
            atom_name=np.concatenate([self.atom_name, other.atom_name], axis=0),
            restype=np.concatenate([self.restype, other.restype], axis=0),
            atom_mask=np.concatenate([self.atom_mask, other.atom_mask], axis=0),
            residue_index=np.concatenate([self.residue_index, other.residue_index], axis=0),
            chain_index=np.concatenate([self.chain_index, other.chain_index], axis=0),
            chain_id=np.concatenate([self.chain_id, other.chain_id], axis=0),
            b_factors=np.concatenate([self.b_factors, other.b_factors], axis=0),
            chemid=np.concatenate([self.chemid, other.chemid], axis=0),
            chemtype=np.concatenate([self.chemtype, other.chemtype], axis=0),
            bonds=list(dict.fromkeys(self.bonds + other.bonds)),
            unique_res_atom_names=self.unique_res_atom_names + other.unique_res_atom_names,
            author_cri_to_new_cri={**self.author_cri_to_new_cri, **other.author_cri_to_new_cri},
            chem_comp_table=self.chem_comp_table.union(other.chem_comp_table),
            entity_to_chain=deep_merge_dicts(
                self.entity_to_chain, other.entity_to_chain, value_op="union"
            ),
            mmcif_to_author_chain=deep_merge_dicts(
                self.mmcif_to_author_chain, other.mmcif_to_author_chain, value_op="union"
            ),
            mmcif_metadata={**self.mmcif_metadata, **other.mmcif_metadata},
        )

    def subset_chains(self, subset_chain_ids: List[str]) -> "Biomolecule":
        """Filters a `Biomolecule` instance to only include a subset's chain IDs."""
        chain_id_mapping = {chain_id: n for n, chain_id in enumerate(np.unique(self.chain_id))}
        assert all(
            chain_id in chain_id_mapping for chain_id in subset_chain_ids
        ), "All subset chain IDs must be present in the Biomolecule object."
        subset_chain_index_mapping = {
            chain_id_mapping[chain_id]: chain_id for chain_id in subset_chain_ids
        }
        entity_to_chain = {
            entity_id: [
                chain_id for chain_id in chain_ids if chain_id in subset_chain_index_mapping
            ]
            for entity_id, chain_ids in self.entity_to_chain.items()
            if any(chain_id in subset_chain_index_mapping for chain_id in chain_ids)
        }
        mmcif_to_author_chain = {
            mmcif_chain: author_chain_id
            for mmcif_chain, author_chain_id in self.mmcif_to_author_chain.items()
            if author_chain_id in subset_chain_index_mapping
        }
        chain_mask = np.isin(self.chain_index, list(subset_chain_index_mapping.keys()))
        return Biomolecule(
            atom_positions=self.atom_positions[chain_mask],
            atom_name=self.atom_name[chain_mask],
            restype=self.restype[chain_mask],
            atom_mask=self.atom_mask[chain_mask],
            residue_index=self.residue_index[chain_mask],
            chain_index=self.chain_index[chain_mask],
            chain_id=self.chain_id[chain_mask],
            b_factors=self.b_factors[chain_mask],
            chemid=self.chemid[chain_mask],
            chemtype=self.chemtype[chain_mask],
            bonds=[
                bond
                for bond in self.bonds
                if bond.ptnr1_auth_asym_id in subset_chain_ids
                and bond.ptnr2_auth_asym_id in subset_chain_ids
            ],
            unique_res_atom_names=[
                unique_res_atom_names
                for unique_res_atom_names in self.unique_res_atom_names
                if unique_res_atom_names[1] in subset_chain_ids
            ],
            author_cri_to_new_cri={
                author_cri: new_cri
                for author_cri, new_cri in self.author_cri_to_new_cri.items()
                if new_cri[0] in subset_chain_index_mapping
            },
            chem_comp_table=self.chem_comp_table,
            entity_to_chain=entity_to_chain,
            mmcif_to_author_chain=mmcif_to_author_chain,
            mmcif_metadata=self.mmcif_metadata,
        )

    def repeat(self, coord: np.ndarray) -> "Biomolecule":
        """Repeat a Biomolecule according to a (repeated) coordinate array."""
        return Biomolecule(
            atom_positions=coord.reshape(-1, 47, 3),
            atom_name=np.tile(self.atom_name, (coord.shape[0], 1)).reshape(-1),
            restype=np.tile(self.restype, (coord.shape[0], 1)).reshape(-1),
            atom_mask=np.tile(self.atom_mask, (coord.shape[0], 1, 1)).reshape(-1, 47),
            residue_index=np.tile(self.residue_index, (coord.shape[0], 1)).reshape(-1),
            chain_index=np.tile(self.chain_index, (coord.shape[0], 1)).reshape(-1),
            chain_id=np.tile(self.chain_id, (coord.shape[0], 1)).reshape(-1),
            b_factors=np.tile(self.b_factors, (coord.shape[0], 1, 1)).reshape(-1, 47),
            chemid=np.tile(self.chemid, (coord.shape[0], 1)).reshape(-1),
            chemtype=np.tile(self.chemtype, (coord.shape[0], 1)).reshape(-1),
            bonds=self.bonds,
            unique_res_atom_names=[
                unique_res_atom_names
                for _ in range(coord.shape[0])
                for unique_res_atom_names in self.unique_res_atom_names
            ],
            author_cri_to_new_cri=self.author_cri_to_new_cri,
            chem_comp_table=self.chem_comp_table,
            entity_to_chain=self.entity_to_chain,
            mmcif_to_author_chain=self.mmcif_to_author_chain,
            mmcif_metadata=self.mmcif_metadata,
        )

    def crop_chains_with_masks(
        self, chain_ids_and_lengths: List[Tuple[str, int]], crop_masks: List[np.ndarray]
    ) -> "Biomolecule":
        """
        Crop the chains and metadata within a Biomolecule
        to only include the specified chain residues.
        """
        assert len(chain_ids_and_lengths) == len(
            crop_masks
        ), "The number of chains and crop masks must be equal."
        assert not all(
            crop_mask.all() for crop_mask in crop_masks
        ), "Not all tokens can be cropped out of a Biomolecule."

        # collect metadata for each chain

        unique_chain_ids = np.unique(self.chain_id)
        chains_to_remove = {
            chain_id_and_length[0]
            for chain_id_and_length, crop_mask in zip(chain_ids_and_lengths, crop_masks)
            if not crop_mask.any()
        }
        subset_chain_id_mapping = {
            chain_id: n
            for n, chain_id in enumerate(unique_chain_ids)
            if chain_id not in chains_to_remove
        }
        subset_chain_index_mapping = {
            n: chain_id for chain_id, n in subset_chain_id_mapping.items()
        }
        chain_id_to_index = {
            chain_id_and_length[0]: i
            for i, chain_id_and_length in enumerate(chain_ids_and_lengths)
        }

        # create metadata for cropping

        chain_mask = np.concatenate(
            [crop_masks[chain_id_to_index[c_id]] for c_id in unique_chain_ids]
        )
        chain_residue_index = np.array(
            list(zip(self.chain_index[chain_mask], self.residue_index[chain_mask]))
        )
        # NOTE: We must only consider unique chain-residue index pairs here,
        # as otherwise we might count each ligand heavy atom as a residue in this mapping
        subset_chain_residue_mapping = set(map(tuple, chain_residue_index))

        # manually subset certain Biomolecule metadata

        entity_to_chain = {
            entity_id: [
                chain_id for chain_id in chain_ids if chain_id in subset_chain_index_mapping
            ]
            for entity_id, chain_ids in self.entity_to_chain.items()
            if any(chain_id in subset_chain_index_mapping for chain_id in chain_ids)
        }
        mmcif_to_author_chain = {
            mmcif_chain: author_chain_id
            for mmcif_chain, author_chain_id in self.mmcif_to_author_chain.items()
            if author_chain_id in subset_chain_index_mapping
        }

        # construct a new cropped Biomolecule

        return Biomolecule(
            atom_positions=self.atom_positions[chain_mask],
            atom_name=self.atom_name[chain_mask],
            restype=self.restype[chain_mask],
            atom_mask=self.atom_mask[chain_mask],
            residue_index=self.residue_index[chain_mask],
            chain_index=self.chain_index[chain_mask],
            chain_id=self.chain_id[chain_mask],
            b_factors=self.b_factors[chain_mask],
            chemid=self.chemid[chain_mask],
            chemtype=self.chemtype[chain_mask],
            bonds=[
                bond
                for bond in self.bonds
                if bond.ptnr1_auth_asym_id not in chains_to_remove
                and bond.ptnr2_auth_asym_id not in chains_to_remove
            ],
            unique_res_atom_names=[
                unique_res_atom_names
                for unique_res_atom_names in self.unique_res_atom_names
                if unique_res_atom_names[1] not in chains_to_remove
                and (subset_chain_id_mapping[unique_res_atom_names[1]], unique_res_atom_names[2])
                in subset_chain_residue_mapping
            ],
            author_cri_to_new_cri={
                author_cri: new_cri
                for author_cri, new_cri in self.author_cri_to_new_cri.items()
                if new_cri[0] in subset_chain_index_mapping
            },
            chem_comp_table=self.chem_comp_table,
            entity_to_chain=entity_to_chain,
            mmcif_to_author_chain=mmcif_to_author_chain,
            mmcif_metadata=self.mmcif_metadata,
        )

    def contiguous_crop(self, n_res: int = 384) -> "Biomolecule":
        """
        Crop a Biomolecule to only include contiguous
        polymer residues and/or ligand atoms for each chain.
        """
        chain_ids_and_lengths = list(collections.Counter(self.chain_id).items())
        random.shuffle(chain_ids_and_lengths)
        crop_masks = create_contiguous_crop_masks(chain_ids_and_lengths, n_res)
        return self.crop_chains_with_masks(chain_ids_and_lengths, crop_masks)

    def spatial_crop(
        self,
        interface_cropping: bool,
        n_res: int = 384,
        chain_1: Optional[str] = None,
        chain_2: Optional[str] = None,
        interface_distance_threshold: float = 15.0,
    ) -> "Biomolecule":
        """
        Crop a Biomolecule to only include polymer residues and ligand atoms
        near a (random) reference atom within a sampled chain/interface.
        """

        # curate a list of candidate token center atoms from which to sample a reference atom

        token_num_chains = np.unique(self.chain_id).size
        token_center_atom_indices = np.arange(self.chain_id.size)
        token_center_atom_mask = np.zeros(self.chain_id.size, dtype=bool)
        token_res_atom_position_mask = np.zeros(self.atom_positions.shape[:-1], dtype=bool)
        token_res_rep_atom_indices = np.array(
            [
                get_residue_constants(res_chem_index=chemtype).res_rep_atom_index
                for chemtype in self.chemtype
            ]
        )
        # NOTE: ligand atom position indices vary per ligand residue, so we can't rely on representative atom indices here
        token_res_rep_atom_indices[self.chemtype == 3] = np.where(
            self.atom_mask[self.chemtype == 3]
        )[1]
        token_res_atom_position_mask[np.arange(self.chain_id.size), token_res_rep_atom_indices] = (
            True
        )
        token_center_atom_positions = self.atom_positions[token_res_atom_position_mask]

        # potentially filter candidate token center atoms by chain ID

        if exists(chain_1) and exists(chain_2):
            chain_1_mask = self.chain_id == chain_1
            chain_2_mask = self.chain_id == chain_2
            token_center_atom_mask[chain_1_mask | chain_2_mask] = True
        elif exists(chain_1):
            token_center_atom_mask[self.chain_id == chain_1] = True
        elif exists(chain_2):
            token_center_atom_mask[self.chain_id == chain_2] = True

        # potentially filter candidate token center atoms by interface proximity

        if interface_cropping and token_num_chains > 1:
            token_center_atom_positions_ = token_center_atom_positions[token_center_atom_mask]
            token_center_atom_chains_ = self.chain_id[token_center_atom_mask]

            interface_token_center_atom_distances = np.linalg.norm(
                token_center_atom_positions_[:, None] - token_center_atom_positions_[None, :],
                axis=-1,
            )
            interface_token_center_atom_mask = np.where(
                (
                    (interface_token_center_atom_distances < interface_distance_threshold)
                    & (token_center_atom_chains_[:, None] != token_center_atom_chains_[None, :])
                ).any(axis=1)
            )[0]

            token_center_atom_mask.fill(False)
            token_center_atom_mask[interface_token_center_atom_mask] = True

        token_center_atom_indices = token_center_atom_indices[token_center_atom_mask]
        if token_center_atom_indices.size == 0:
            raise ValueError(
                "No chain atoms found for the specified chain(s) after spatial cropping."
            )

        # sample a reference atom for spatial cropping

        reference_atom_index = random.choice(token_center_atom_indices).item()

        # perform spatial cropping according to reference atom proximity

        chain_ids_and_lengths = list(collections.Counter(self.chain_id).items())
        crop_masks = create_spatial_crop_masks(
            token_center_atom_positions,
            self.chain_index,
            reference_atom_index,
            n_res,
        )
        return self.crop_chains_with_masks(chain_ids_and_lengths, crop_masks)

    def crop(
        self,
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        spatial_interface_weight: float = 0.4,
        n_res: int = 384,
        chain_1: Optional[str] = None,
        chain_2: Optional[str] = None,
    ) -> "Biomolecule":
        """Crop a Biomolecule using a randomly-sampled cropping function."""
        crop_fn_weights = [contiguous_weight, spatial_weight, spatial_interface_weight]
        crop_fns = [
            partial(self.contiguous_crop, n_res=n_res),
            partial(
                self.spatial_crop,
                interface_cropping=False,
                n_res=n_res,
                chain_1=chain_1,
                chain_2=chain_2,
            ),
            partial(
                self.spatial_crop,
                interface_cropping=True,
                n_res=n_res,
                chain_1=chain_1,
                chain_2=chain_2,
            ),
        ]
        crop_fn = random.choices(crop_fns, crop_fn_weights)[0]
        return crop_fn()


@typecheck
def create_contiguous_crop_masks(
    chain_ids_and_lengths: List[Tuple[str, int]], n_res: int
) -> List[np.ndarray]:
    """
    Create contiguous crop masks for each given chain.
    Implements Algorithm 1 from the AlphaFold-Multimer paper.
    """
    m_ks = []
    n_added = 0
    n_remaining = n_res
    for chain_id_and_length in chain_ids_and_lengths:
        n_k = chain_id_and_length[1]
        n_remaining -= n_k
        crop_size_max = min(n_res - n_added, n_k)
        # NOTE: `max(0, n_remaining)` was analytically added to prevent invalid crop sizes.
        crop_size_min = min(n_k, max(0, n_res - (n_added + max(0, n_remaining))))
        crop_size = random.randrange(crop_size_min, crop_size_max + 1)
        n_added += crop_size
        crop_start = random.randrange(0, n_k - crop_size + 1)
        m_k = np.zeros(n_k, dtype=bool)
        keep = np.arange(crop_start, crop_start + crop_size)
        m_k[keep] = True
        m_ks.append(m_k)
    return m_ks


@typecheck
def create_spatial_crop_masks(
    token_center_atom_positions: np.ndarray,
    token_center_atom_chains: np.ndarray,
    reference_token_center_atom_index: int,
    n_res: int,
) -> List[np.ndarray]:
    """
    Create spatial crop masks for each given chain.
    Implements Algorithm 2 from the AlphaFold-Multimer paper.
    """
    # calculate distances with small uniquifying values to break ties

    num_atoms = len(token_center_atom_positions)
    uniquifying_values = np.arange(num_atoms) * 1e-3
    reference_position = token_center_atom_positions[reference_token_center_atom_index]
    distances = np.linalg.norm(token_center_atom_positions - reference_position, axis=-1)
    distances_with_ties_broken = distances + uniquifying_values

    # select the `n_res`th nearest neighbor distance as cutoff

    d_cutoff = np.partition(distances_with_ties_broken, n_res - 1)[n_res - 1]

    # create a mask for atoms within the cutoff distance

    m_k = distances_with_ties_broken <= d_cutoff

    # identify the indices where chains change

    chain_change_indices = np.where(np.diff(token_center_atom_chains) != 0)[0] + 1

    # split the mask into separate arrays for each chain

    m_ks = np.split(m_k, chain_change_indices)

    return m_ks


@typecheck
def get_residue_constants(
    res_chem_type: Optional[str] = None, res_chem_index: Optional[IntType] = None
) -> ModuleType:
    """Returns the corresponding residue constants for a given residue chemical type."""
    assert exists(res_chem_type) or exists(
        res_chem_index
    ), "Either `res_chem_type` or `res_chem_index` must be provided."
    if (exists(res_chem_type) and "peptide" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 0
    ):
        residue_constants = amino_acid_constants
    elif (exists(res_chem_type) and "rna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 1
    ):
        residue_constants = rna_constants
    elif (exists(res_chem_type) and "dna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 2
    ):
        residue_constants = dna_constants
    else:
        residue_constants = ligand_constants
    return residue_constants


@typecheck
def get_ligand_atom_name(atom_name: str, atom_types_set: Set[str]) -> str:
    """Gets the in-vocabulary atom name where possible for a ligand atom."""
    if len(atom_name) == 1:
        return atom_name
    elif len(atom_name) == 2:
        return atom_name if atom_name in atom_types_set else atom_name[0]
    elif len(atom_name) == 3:
        if atom_name in atom_types_set:
            return atom_name
        elif atom_name[:2] in atom_types_set:
            return atom_name[:2]
        elif atom_name[1:] in atom_types_set:
            return atom_name[1:]
        elif atom_name[0] + atom_name[2] in atom_types_set:
            return atom_name[0] + atom_name[2]
        elif atom_name.split("H")[0] in atom_types_set:
            return atom_name.split("H")[0]
        else:
            return atom_name
    else:
        return atom_name


@typecheck
def get_unique_res_atom_names(
    mmcif_object: mmcif_parsing.MmcifObject,
) -> List[Tuple[List[List[str]], str, int]]:
    """Get atom name-chain ID tuples for each (e.g. ligand) "pseudoresidue" of each residue in each chain."""
    unique_res_atom_names = []
    for chain in mmcif_object.structure:
        chain_chem_comp = mmcif_object.chem_comp_details[chain.id]
        for res, res_chem_comp in zip(chain, chain_chem_comp):
            is_polymer_residue = is_polymer(res_chem_comp.type)
            residue_constants = get_residue_constants(res_chem_type=res_chem_comp.type)
            if is_polymer_residue:
                # For polymer residues, append the atom types directly.
                atoms_to_append = [residue_constants.atom_types]
            else:
                # For non-polymer residues, create a nested list of atom names.
                atoms_to_append = [
                    [atom.name for _ in range(residue_constants.atom_type_num)] for atom in res
                ]
            unique_res_atom_names.append((atoms_to_append, chain.id, res.id[1]))
    return unique_res_atom_names


@typecheck
def _from_mmcif_object(
    mmcif_object: mmcif_parsing.MmcifObject,
    chain_ids: Optional[Set[str]] = None,
) -> Biomolecule:
    """Takes a Biopython structure/model mmCIF object and creates a `Biomolecule` instance.

    WARNING: All non-standard residue types will be converted into the corresponding
      unknown residue type for the residue's chemical type (e.g., RNA).
      All non-standard atoms will be ignored.

    WARNING: The residues in each chain will be reindexed to start at 1 and to be
      monotonically increasing, potentially in contrast to the original residue indices
      in the input mmCIF object. This is to enable successful re-parsing of any mmCIF
      objects written to a new mmCIF file using the `to_mmcif()` function, as including
      missing residues (denoted by gaps in residue indices) in an output mmCIF file
      can cause numerous downstream parsing errors.

    :param mmcif_object: The parsed Biopython structure/model mmCIF object.
    :param chain_ids: If chain_ids are specified (e.g. A), then only these chains are parsed.
        Otherwise all chains are parsed.

    :return: A new `Biomolecule` created from the structure/model mmCIF object contents.

    :raise:
      ValueError: If the number of models included in a given structure is not 1.
      ValueError: If insertion code is detected at a residue.
    """
    structure = mmcif_object.structure
    # Resolve alternative locations for atoms/residues by taking the one with the largest occupancy.
    # NOTE: For `DisorderedAtom` objects, selecting the highest-occupancy atom is already the default behavior in Biopython.
    # Reference: https://biopython-tutorial.readthedocs.io/en/latest/notebooks/11%20-%20Going%203D%20-%20The%20PDB%20module.html#Disordered-atoms[disordered-atoms]
    if isinstance(structure, Model):
        model = structure
    else:
        models = list(structure.get_models())
        if len(models) != 1:
            raise ValueError(
                "Only single model mmCIFs are supported. Found" f" {len(models)} models."
            )
        model = models[0]

    atom_positions = []
    atom_names = []
    restype = []
    chemid = []
    chemtype = []
    residue_chem_comp_details = set()
    atom_mask = []
    residue_index = []
    chain_idx = []
    b_factors = []
    author_cri_to_new_cri = {}

    for chain in model:
        if exists(chain_ids) and chain.id not in chain_ids:
            continue
        for res_index, res in enumerate(chain):
            if res.id[2] != " ":
                raise ValueError(
                    f"mmCIF contains an insertion code at chain {chain.id} and"
                    f" original (new) residue index {res.id[1]} ({res_index + 1}). These are not supported."
                )
            res_chem_comp_details = mmcif_object.chem_comp_details[chain.id][res_index]
            assert res.resname == res_chem_comp_details.id, (
                f"Structural residue name {res.resname} does not match the residue ID"
                f" {res_chem_comp_details.id} in the mmCIF chemical component dictionary for {mmcif_object.file_id}."
            )
            is_polymer_residue = is_polymer(res_chem_comp_details.type)
            is_peptide_residue = "peptide" in res_chem_comp_details.type.lower()
            residue_constants = get_residue_constants(res_chem_type=res_chem_comp_details.type)
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            if is_polymer_residue:
                pos = np.zeros((residue_constants.atom_type_num, 3))
                mask = np.zeros((residue_constants.atom_type_num,))
                res_b_factors = np.zeros((residue_constants.atom_type_num,))
                for atom in res:
                    if is_polymer_residue and atom.name not in residue_constants.atom_types_set:
                        continue
                    elif (
                        is_peptide_residue
                        and atom.name.upper() == "SE"
                        and res.get_resname() == "MSE"
                    ):
                        # Put the coords of the selenium atom in the sulphur column.
                        pos[residue_constants.atom_order["SD"]] = atom.coord
                        mask[residue_constants.atom_order["SD"]] = 1.0
                        res_b_factors[residue_constants.atom_order["SD"]] = atom.bfactor
                    else:
                        pos[residue_constants.atom_order[atom.name]] = atom.coord
                        mask[residue_constants.atom_order[atom.name]] = 1.0
                        res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
                if np.sum(mask) < 0.5:
                    # If no known atom positions are reported for a polymer residue then skip it.
                    continue
                if is_peptide_residue:
                    # Fix naming errors in arginine residues where NH2 is incorrectly
                    # assigned to be closer to CD than NH1
                    cd = residue_constants.atom_order["CD"]
                    nh1 = residue_constants.atom_order["NH1"]
                    nh2 = residue_constants.atom_order["NH2"]
                    if (
                        res.get_resname() == "ARG"
                        and all(mask[atom_index] for atom_index in (cd, nh1, nh2))
                        and (
                            np.linalg.norm(pos[nh1] - pos[cd]) > np.linalg.norm(pos[nh2] - pos[cd])
                        )
                    ):
                        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
                        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()
                        res_b_factors[nh1], res_b_factors[nh2] = (
                            res_b_factors[nh2].copy(),
                            res_b_factors[nh1].copy(),
                        )
                # Collect the residue's features.
                restype.append(restype_idx)
                chemid.append(res_chem_comp_details.id)
                chemtype.append(residue_constants.chemtype_num)
                atom_positions.append(pos)
                atom_names.append(
                    residue_constants.atom_types[residue_constants.res_rep_atom_index]
                )
                atom_mask.append(mask)
                residue_index.append(res_index + 1)
                chain_idx.append(chain.id)
                b_factors.append(res_b_factors)
                author_cri_to_new_cri[(chain.id, res.resname, res.id[1])] = (
                    chain.id,
                    res.resname,
                    res_index + 1,
                )
                if res.resname == residue_constants.unk_restype:
                    # If the polymer residue is unknown, then it is of the corresponding unknown polymer residue type.
                    residue_chem_comp_details.add(
                        mmcif_parsing.ChemComp(
                            id=residue_constants.unk_restype,
                            formula="?",
                            formula_weight="0.0",
                            mon_nstd_flag="no",
                            name=residue_constants.unk_chemname,
                            type=residue_constants.unk_chemtype,
                        )
                    )
                else:
                    residue_chem_comp_details.add(res_chem_comp_details)
            else:
                # Represent each ligand atom as a single "pseudoresidue".
                # NOTE: Ligand "pseudoresidues" can later be grouped back
                # into a single ligand residue using indexing operations
                # working jointly on chain_index and residue_index.
                for atom in res:
                    # NOTE: This code assumes water residues have previously been filtered out.
                    pos = np.zeros((residue_constants.atom_type_num, 3))
                    mask = np.zeros((residue_constants.atom_type_num,))
                    res_b_factors = np.zeros((residue_constants.atom_type_num,))
                    atom_name = get_ligand_atom_name(atom.name, residue_constants.atom_types_set)
                    if atom_name not in residue_constants.atom_types_set:
                        atom_name = "ATM"
                    atom_names.append(atom_name)
                    pos[residue_constants.atom_order[atom_name]] = atom.coord
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                    res_b_factors[residue_constants.atom_order[atom_name]] = atom.bfactor
                    restype.append(restype_idx)
                    chemid.append(res_chem_comp_details.id)
                    chemtype.append(residue_constants.chemtype_num)
                    atom_positions.append(pos)
                    atom_mask.append(mask)
                    residue_index.append(res_index + 1)
                    chain_idx.append(chain.id)
                    b_factors.append(res_b_factors)

                author_cri_to_new_cri[(chain.id, res.resname, res.id[1])] = (
                    chain.id,
                    res.resname,
                    res_index + 1,
                )

                if res.resname == residue_constants.unk_restype:
                    # If the ligand residue is unknown, then it is of the unknown ligand residue type.
                    residue_chem_comp_details.add(
                        mmcif_parsing.ChemComp(
                            id=residue_constants.unk_restype,
                            formula="?",
                            formula_weight="0.0",
                            mon_nstd_flag="no",
                            name=residue_constants.unk_chemname,
                            type=residue_constants.unk_chemtype,
                        )
                    )
                else:
                    residue_chem_comp_details.add(res_chem_comp_details)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_idx)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_idx])

    # Construct a mapping from an integer entity ID to integer chain IDs.
    entity_to_chain = {
        entity: [chain_id_mapping[chain] for chain in chains if chain in chain_id_mapping]
        for entity, chains in mmcif_object.entity_to_chain.items()
        if any(chain in chain_id_mapping for chain in chains)
    }

    # Construct a mapping from (internal) mmCIF chain ID string to integer author chain ID.
    mmcif_to_author_chain = {
        mmcif_chain: chain_id_mapping[author_chain]
        for mmcif_chain, author_chain in mmcif_object.mmcif_to_author_chain.items()
        if author_chain in chain_id_mapping
    }

    # For mmCIF I/O, retain a list of unique atom names for each "pseudoresidue" of each residue.
    unique_res_atom_names = get_unique_res_atom_names(mmcif_object)

    return Biomolecule(
        atom_positions=np.array(atom_positions),
        atom_name=np.array(atom_names),
        restype=np.array(restype),
        atom_mask=np.array(atom_mask),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        chain_id=np.array(chain_idx),
        b_factors=np.array(b_factors),
        chemid=np.array(chemid),
        chemtype=np.array(chemtype),
        bonds=mmcif_object.bonds,
        unique_res_atom_names=unique_res_atom_names,
        author_cri_to_new_cri=author_cri_to_new_cri,
        chem_comp_table=residue_chem_comp_details,
        entity_to_chain=entity_to_chain,
        mmcif_to_author_chain=mmcif_to_author_chain,
        mmcif_metadata=mmcif_object.raw_string,
    )


@typecheck
def from_mmcif_string(
    mmcif_str: str, file_id: str, chain_ids: Optional[Set[str]] = None
) -> Biomolecule:
    """Takes a mmCIF string and constructs a `Biomolecule` object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    :param mmcif_str: The contents of the mmCIF file.
    :param file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    :param chain_ids: If chain_ids are specified (e.g. A), then only these chains are parsed.
        Otherwise all chains are parsed.

    :return: A new `Biomolecule` parsed from the mmCIF contents.

    :raise:
        ValueError: If the mmCIF file is not valid.
    """
    parsing_result = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_str)

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with beforehand (e.g., at the alignment stage).
    if parsing_result.mmcif_object is None:
        raise list(parsing_result.errors.values())[0]

    return _from_mmcif_object(parsing_result.mmcif_object, chain_ids=chain_ids)


@typecheck
def atom_id_to_type(atom_id: str) -> str:
    """Convert atom ID to atom type, works only for standard residues.

    :param atom_id: Atom ID to be converted.
    :return: String corresponding to atom type.

    :raise:
        ValueError: If `atom_id` is empty or does not contain any alphabetic characters.
    """
    assert len(atom_id) > 0, f"Atom ID must have at least one character, but received: {atom_id}."
    for char in atom_id:
        if char.isalpha():
            return char
    raise ValueError(
        f"Atom ID must contain at least one alphabetic character, but received: {atom_id}."
    )


@typecheck
def remove_metadata_fields_by_prefixes(
    metadata_dict: Dict[str, List[Any]], field_prefixes: List[str]
) -> Dict[str, List[Any]]:
    """Remove metadata fields from the metadata dictionary.

    :param metadata_dict: The metadata default dictionary from which to remove metadata fields.
    :param field_prefixes: A list of prefixes to remove from the metadata default dictionary.

    :return: A metadata dictionary with the specified metadata fields removed.
    """
    return {
        key: value
        for key, value in metadata_dict.items()
        if not any(key.startswith(prefix) for prefix in field_prefixes)
    }


@typecheck
def to_mmcif(
    biomol: Biomolecule,
    file_id: str,
    gapless_poly_seq: bool = True,
    insert_alphafold_mmcif_metadata: bool = True,
    unique_res_atom_names: Optional[List[Tuple[List[List[str]], str, int]]] = None,
) -> str:
    """Converts a `Biomolecule` instance to an mmCIF string.

    WARNING 1: When gapless_poly_seq is True, the _pdbx_poly_seq_scheme is filled
      with unknown residues for any missing residue indices in the range
      from min(1, min(residue_index)) to max(residue_index).
      E.g. for a biomolecule object with positions for (majority) protein residues
      2 (MET), 3 (LYS), 6 (GLY), this method would set the _pdbx_poly_seq_scheme to:
      1 UNK
      2 MET
      3 LYS
      4 UNK
      5 UNK
      6 GLY
      This is done to preserve the residue numbering.

    WARNING 2: Converting ground truth mmCIF file to Biomolecule and then back to
      mmCIF using this method will convert all non-standard residue types to the
      corresponding unknown residue type of the residue's chemical type (e.g., RNA).
      If you need this behaviour, you need to store more mmCIF metadata in the
      Biomolecule object (e.g. all fields except for the _atom_site loop).

    WARNING 3: Converting ground truth mmCIF file to Biomolecule and then back to
      mmCIF using this method will not retain the original chain indices and will
      instead install author chain IDs.

    WARNING 4: In case of multiple identical chains, they are assigned different
      `_atom_site.label_entity_id` values.

    :param biomol: A biomolecule to convert to mmCIF string.
    :param file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    :param gapless_poly_seq: If True, the polymer output will contain gapless residue indices.
    :param insert_alphafold_mmcif_metadata: If True, insert metadata fields
        referencing AlphaFold in the output mmCIF file.
    :param unique_res_atom_names: A dictionary mapping each author chain ID to a list of lists
        of lists of atom names for each "pseudoresidue" of each unique residue. If None, the
        atom names are assumed to be the same for all residues of the same chemical type (e.g., RNA).
        This is used to group "pseudoresidues" (e.g., ligand atoms) by parent residue.

    :return: A valid mmCIF string.

    :raise:
      ValueError: If amino-acid or nucleotide residue types array contains entries with
      too many biomolecule types.
    """
    atom_positions = biomol.atom_positions
    restype = biomol.restype
    atom_mask = biomol.atom_mask
    residue_index = biomol.residue_index.astype(np.int32)
    chain_index = biomol.chain_index.astype(np.int32)
    b_factors = biomol.b_factors
    chemid = biomol.chemid
    chemtype = biomol.chemtype
    bonds = biomol.bonds
    author_cri_to_new_cri = biomol.author_cri_to_new_cri
    entity_id_to_chain_ids = biomol.entity_to_chain
    mmcif_to_author_chain_ids = biomol.mmcif_to_author_chain
    orig_mmcif_metadata = biomol.mmcif_metadata

    # Via unique chain-residue indexing, ensure that each ligand residue
    # is represented by only a single atom for all logic except atom site parsing.
    chain_residue_index = np.array(
        [f"{chain_index[i]}_{residue_index[i]}" for i in range(residue_index.shape[0])]
    )
    _, unique_indices = np.unique(chain_residue_index, return_index=True)
    unique_indices = sorted(unique_indices)

    unique_restype = restype[unique_indices]
    unique_residue_index = residue_index[unique_indices]
    unique_chain_index = chain_index[unique_indices]
    unique_chemid = chemid[unique_indices]
    unique_chemtype = chemtype[unique_indices]

    # Construct a mapping from integer chain indices to chain ID strings.
    chain_ids = {}
    for chain_id in np.unique(unique_chain_index):  # np.unique gives sorted output.
        chain_ids[chain_id] = _int_id_to_str_id(chain_id + 1)

    mmcif_dict = collections.defaultdict(list)

    mmcif_dict["data_"] = file_id.upper()
    mmcif_dict["_entry.id"] = file_id.upper()

    label_asym_id_to_entity_id = {}
    unique_polymer_entity_pdbx_strand_ids = set()
    # Entity and chain information.
    for entity_id, entity_chain_ids in entity_id_to_chain_ids.items():
        assert len(entity_chain_ids) > 0, f"Entity {entity_id} must contain at least one chain."
        entity_types = []
        polymer_entity_types = []
        polymer_entity_pdbx_strand_ids = []
        for chain_id in entity_chain_ids:
            # Determine the (majority) chemical type of the chain.
            res_chemindex = np_mode(unique_chemtype[unique_chain_index == chain_id])[0].item()
            residue_constants = get_residue_constants(res_chem_index=res_chemindex)
            # Add all chain information to the _struct_asym table.
            label_asym_id_to_entity_id[chain_ids[chain_id]] = str(entity_id)
            mmcif_dict["_struct_asym.id"].append(chain_ids[chain_id])
            mmcif_dict["_struct_asym.entity_id"].append(str(entity_id))
            # Collect entity information for each chain.
            entity_types.append(residue_constants.POLYMER_CHAIN)
            # Collect only polymer information for each chain.
            if res_chemindex < 3:
                polymer_entity_types.append(residue_constants.BIOMOLECULE_CHAIN)
                polymer_entity_pdbx_strand_ids.append(chain_ids[chain_id])
                unique_polymer_entity_pdbx_strand_ids.add(chain_ids[chain_id])

        # Generate the _entity table, labeling the entity's (majority) type.
        entity_type = np_mode(np.array(entity_types))[0].item()
        mmcif_dict["_entity.id"].append(str(entity_id))
        mmcif_dict["_entity.type"].append(entity_type)
        # Add information about the polymer entities to the _entity_poly table,
        # labeling each polymer entity's (majority) type.
        if polymer_entity_types:
            polymer_entity_type = np_mode(np.array(polymer_entity_types))[0].item()
            mmcif_dict["_entity_poly.entity_id"].append(str(entity_id))
            mmcif_dict["_entity_poly.type"].append(polymer_entity_type)
            mmcif_dict["_entity_poly.pdbx_strand_id"].append(
                ",".join(polymer_entity_pdbx_strand_ids)
            )

    # Bioassembly information.
    # Export latest data for the _pdbx_struct_assembly_gen table.
    pdbx_struct_assembly_oligomeric_count = collections.defaultdict(int)
    pdbx_struct_assembly_gen_assembly_ids = orig_mmcif_metadata.get(
        "_pdbx_struct_assembly_gen.assembly_id", []
    )
    pdbx_struct_assembly_gen_oper_expressions = orig_mmcif_metadata.get(
        "_pdbx_struct_assembly_gen.oper_expression", []
    )
    pdbx_struct_assembly_gen_asym_id_lists = orig_mmcif_metadata.get(
        "_pdbx_struct_assembly_gen.asym_id_list", []
    )
    if any(
        len(item) > 0
        for item in (
            pdbx_struct_assembly_gen_assembly_ids,
            pdbx_struct_assembly_gen_oper_expressions,
            pdbx_struct_assembly_gen_asym_id_lists,
        )
    ):
        # Sanity-check the lengths of the _pdbx_struct_assembly_gen table entries.
        assert (
            len(pdbx_struct_assembly_gen_assembly_ids)
            == len(pdbx_struct_assembly_gen_oper_expressions)
            == len(pdbx_struct_assembly_gen_asym_id_lists)
        ), f"Mismatched lengths ({len(pdbx_struct_assembly_gen_assembly_ids)}, {len(pdbx_struct_assembly_gen_oper_expressions)}, {len(pdbx_struct_assembly_gen_asym_id_lists)}) for _pdbx_struct_assembly_gen table entries."
    for assembly_id, oper_expression, asym_id_list in zip(
        pdbx_struct_assembly_gen_assembly_ids,
        pdbx_struct_assembly_gen_oper_expressions,
        pdbx_struct_assembly_gen_asym_id_lists,
    ):
        author_asym_ids = []
        for asym_id in asym_id_list.split(","):
            if asym_id not in mmcif_to_author_chain_ids:
                continue
            author_asym_id = chain_ids[mmcif_to_author_chain_ids[asym_id]]
            if author_asym_id in unique_polymer_entity_pdbx_strand_ids:
                author_asym_ids.append(author_asym_id)
        if not author_asym_ids:
            continue
        # In original order, remove any duplicate author chains arising from the many-to-one mmCIF-to-author chain mapping.
        author_asym_ids = list(dict.fromkeys(author_asym_ids))
        mmcif_dict["_pdbx_struct_assembly_gen.assembly_id"].append(str(assembly_id))
        mmcif_dict["_pdbx_struct_assembly_gen.oper_expression"].append(str(oper_expression))
        mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"].append(",".join(author_asym_ids))
        pdbx_struct_assembly_oligomeric_count[assembly_id] += sum(
            # Only count polymer entities in the oligomeric count.
            asym_id in unique_polymer_entity_pdbx_strand_ids
            for asym_id in author_asym_ids
        )
    # Export latest data for the _pdbx_struct_assembly table.
    pdbx_struct_assembly_gen_ids = set(mmcif_dict["_pdbx_struct_assembly_gen.assembly_id"])
    pdbx_struct_assembly_ids = orig_mmcif_metadata.get("_pdbx_struct_assembly.id", [])
    pdbx_struct_assembly_details = orig_mmcif_metadata.get("_pdbx_struct_assembly.details", [])
    if any(
        len(item) > 0
        for item in (
            pdbx_struct_assembly_ids,
            pdbx_struct_assembly_details,
        )
    ):
        # Sanity-check the lengths of the _pdbx_struct_assembly table entries.
        assert len(pdbx_struct_assembly_ids) == len(
            pdbx_struct_assembly_details
        ), f"Mismatched lengths ({len(pdbx_struct_assembly_ids)}, {len(pdbx_struct_assembly_details)}) for _pdbx_struct_assembly table entries."
    for assembly_id, assembly_details in zip(
        pdbx_struct_assembly_ids,
        pdbx_struct_assembly_details,
    ):
        if assembly_id not in pdbx_struct_assembly_gen_ids:
            continue
        mmcif_dict["_pdbx_struct_assembly.id"].append(str(assembly_id))
        mmcif_dict["_pdbx_struct_assembly.details"].append(str(assembly_details))
        mmcif_dict["_pdbx_struct_assembly.oligomeric_details"].append(
            "?"
        )  # NOTE: After chain filtering, we cannot assume the oligmeric state.
        mmcif_dict["_pdbx_struct_assembly.oligomeric_count"].append(
            str(pdbx_struct_assembly_oligomeric_count[assembly_id])
        )

    # Populate the _struct_conn table.
    for bond in bonds:
        # Skip bonds between residues that have previously been filtered out.
        ptnr1_key = (
            bond.ptnr1_auth_asym_id,
            bond.ptnr1_auth_comp_id,
            int(bond.ptnr1_auth_seq_id),
        )
        ptnr2_key = (
            bond.ptnr2_auth_asym_id,
            bond.ptnr2_auth_comp_id,
            int(bond.ptnr2_auth_seq_id),
        )
        if ptnr1_key not in author_cri_to_new_cri or ptnr2_key not in author_cri_to_new_cri:
            continue
        # Partner 1
        ptnr1_mapping = author_cri_to_new_cri[ptnr1_key]
        mmcif_dict["_struct_conn.ptnr1_auth_seq_id"].append(
            str(ptnr1_mapping[2])
        )  # Reindex ptnr1 residue ID.
        mmcif_dict["_struct_conn.ptnr1_auth_comp_id"].append(bond.ptnr1_auth_comp_id)
        mmcif_dict["_struct_conn.ptnr1_auth_asym_id"].append(bond.ptnr1_auth_asym_id)
        mmcif_dict["_struct_conn.ptnr1_label_atom_id"].append(bond.ptnr1_label_atom_id)
        mmcif_dict["_struct_conn.pdbx_ptnr1_label_alt_id"].append(bond.pdbx_ptnr1_label_alt_id)
        # Partner 2
        ptnr2_mapping = author_cri_to_new_cri[ptnr2_key]
        mmcif_dict["_struct_conn.ptnr2_auth_seq_id"].append(
            str(ptnr2_mapping[2])
        )  # Reindex ptnr2 residue ID.
        mmcif_dict["_struct_conn.ptnr2_auth_comp_id"].append(bond.ptnr2_auth_comp_id)
        mmcif_dict["_struct_conn.ptnr2_auth_asym_id"].append(bond.ptnr2_auth_asym_id)
        mmcif_dict["_struct_conn.ptnr2_label_atom_id"].append(bond.ptnr2_label_atom_id)
        mmcif_dict["_struct_conn.pdbx_ptnr2_label_alt_id"].append(bond.pdbx_ptnr2_label_alt_id)
        # Connection metadata
        mmcif_dict["_struct_conn.pdbx_leaving_atom_flag"].append(bond.pdbx_leaving_atom_flag)
        mmcif_dict["_struct_conn.pdbx_dist_value"].append(bond.pdbx_dist_value)
        mmcif_dict["_struct_conn.pdbx_role"].append(bond.pdbx_role)
        mmcif_dict["_struct_conn.conn_type_id"].append(bond.conn_type_id)

    # Populate the _chem_comp table.
    for chem_comp in biomol.chem_comp_table:
        mmcif_dict["_chem_comp.id"].append(chem_comp.id)
        mmcif_dict["_chem_comp.formula"].append(chem_comp.formula)
        mmcif_dict["_chem_comp.formula_weight"].append(chem_comp.formula_weight)
        mmcif_dict["_chem_comp.mon_nstd_flag"].append(chem_comp.mon_nstd_flag)
        mmcif_dict["_chem_comp.name"].append(chem_comp.name)
        mmcif_dict["_chem_comp.type"].append(chem_comp.type)
    chem_comp_ids = set(mmcif_dict["_chem_comp.id"])

    # Add the polymer residues to the _pdbx_poly_seq_scheme table.
    for chain_id, (res_ids, chemids, chemindices) in _get_chain_seq(
        unique_restype,
        unique_residue_index,
        unique_chain_index,
        unique_chemid,
        unique_chemtype,
        gapless=gapless_poly_seq,
        non_polymer_only=False,
    ).items():
        for res_id, res_chemid, res_chemindex in zip(res_ids, chemids, chemindices):
            mmcif_dict["_pdbx_poly_seq_scheme.asym_id"].append(chain_ids[chain_id])
            mmcif_dict["_pdbx_poly_seq_scheme.entity_id"].append(
                label_asym_id_to_entity_id[chain_ids[chain_id]]
            )
            mmcif_dict["_pdbx_poly_seq_scheme.seq_id"].append(str(res_id))
            mmcif_dict["_pdbx_poly_seq_scheme.auth_seq_num"].append(str(res_id))
            mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"].append(str(res_id))
            mmcif_dict["_pdbx_poly_seq_scheme.mon_id"].append(res_chemid)
            mmcif_dict["_pdbx_poly_seq_scheme.auth_mon_id"].append(res_chemid)
            mmcif_dict["_pdbx_poly_seq_scheme.pdb_mon_id"].append(res_chemid)
            mmcif_dict["_pdbx_poly_seq_scheme.hetero"].append("n")

            # Add relevant missing polymer residue types to the _chem_comp table.
            residue_constants = get_residue_constants(res_chem_index=res_chemindex)
            if res_chemid == residue_constants.unk_restype and res_chemid not in chem_comp_ids:
                chem_comp_ids.add(residue_constants.unk_restype)
                mmcif_dict["_chem_comp.id"].append(residue_constants.unk_restype)
                mmcif_dict["_chem_comp.formula"].append("?")
                mmcif_dict["_chem_comp.formula_weight"].append("0.0")
                mmcif_dict["_chem_comp.mon_nstd_flag"].append("no")
                mmcif_dict["_chem_comp.name"].append(residue_constants.unk_chemname)
                mmcif_dict["_chem_comp.type"].append(residue_constants.unk_chemtype)

    # Add the non-polymer residues to the _pdbx_nonpoly_scheme table.
    for chain_id, (res_ids, chemids, chemindices) in _get_chain_seq(
        unique_restype,
        unique_residue_index,
        unique_chain_index,
        unique_chemid,
        unique_chemtype,
        gapless=False,
        non_polymer_only=True,
    ).items():
        for res_id, res_chemid, res_chemindex in zip(res_ids, chemids, chemindices):
            mmcif_dict["_pdbx_nonpoly_scheme.asym_id"].append(chain_ids[chain_id])
            mmcif_dict["_pdbx_nonpoly_scheme.entity_id"].append(
                label_asym_id_to_entity_id[chain_ids[chain_id]]
            )
            mmcif_dict["_pdbx_nonpoly_scheme.auth_seq_num"].append(str(res_id))
            mmcif_dict["_pdbx_nonpoly_scheme.pdb_seq_num"].append(str(res_id))
            mmcif_dict["_pdbx_nonpoly_scheme.auth_mon_id"].append(res_chemid)
            mmcif_dict["_pdbx_nonpoly_scheme.pdb_mon_id"].append(res_chemid)

            # Add relevant missing non-polymer residue types to the _chem_comp table.
            residue_constants = get_residue_constants(res_chem_index=res_chemindex)
            if res_chemid == residue_constants.unk_restype and res_chemid not in chem_comp_ids:
                chem_comp_ids.add(residue_constants.unk_restype)
                mmcif_dict["_chem_comp.id"].append(residue_constants.unk_restype)
                mmcif_dict["_chem_comp.formula"].append("?")
                mmcif_dict["_chem_comp.formula_weight"].append("0.0")
                mmcif_dict["_chem_comp.mon_nstd_flag"].append("no")
                mmcif_dict["_chem_comp.name"].append(residue_constants.unk_chemname)
                mmcif_dict["_chem_comp.type"].append(residue_constants.unk_chemtype)

    # Add all atom sites.
    if exists(unique_res_atom_names):
        unique_res_atom_names = [item[0] for item in unique_res_atom_names]
        assert len(unique_res_atom_names) == len(
            unique_restype
        ), f"Unique residue atom names array must have the same length ({len(unique_res_atom_names)}) as the unique residue types array ({len(unique_restype)})."
    atom_index = 1
    for i in range(unique_restype.shape[0]):
        # Determine the chemical type of the residue.
        res_chemindex = unique_chemtype[i]
        is_polymer_residue = (res_chemindex < 3).item()
        residue_constants = get_residue_constants(res_chem_index=res_chemindex)
        res_name_3 = unique_chemid[i]
        # Group "pseudoresidues" (e.g., ligand atoms) by parent residue.
        unique_atom_indices = collections.defaultdict(int)
        res_atom_positions = atom_positions[
            (chain_index == unique_chain_index[i]) & (residue_index == unique_residue_index[i])
        ]
        res_atom_mask = atom_mask[
            (chain_index == unique_chain_index[i]) & (residue_index == unique_residue_index[i])
        ]
        res_b_factors = b_factors[
            (chain_index == unique_chain_index[i]) & (residue_index == unique_residue_index[i])
        ]
        if (unique_restype[i] - residue_constants.min_restype_num) <= len(
            residue_constants.restypes
        ):
            res_atom_names = (
                unique_res_atom_names[i]
                if exists(unique_res_atom_names)
                else [residue_constants.atom_types for _ in range(len(res_atom_positions))]
            )
            assert len(res_atom_positions) == len(
                res_atom_names
            ), "Residue positions array and residue atom names array must have the same length."
        else:
            raise ValueError(
                "Residue types array contains entries with too many biomolecule types."
            )
        # Iterate over all "pseudoresidues" associated with a parent residue.
        for atom_name_, pos_, mask_, b_factor_ in zip(
            res_atom_names, res_atom_positions, res_atom_mask, res_b_factors
        ):
            # Iterate over each atom in a (pseudo)residue.
            for atom_name, pos, mask, b_factor in zip(atom_name_, pos_, mask_, b_factor_):
                if mask < 0.5:
                    continue
                type_symbol = atom_id_to_type(atom_name)

                unique_atom_indices[atom_name] += 1
                atom_index_postfix = "" if is_polymer_residue else unique_atom_indices[atom_name]
                unique_atom_name = (
                    atom_name
                    if exists(unique_res_atom_names)
                    else f"{atom_name}{atom_index_postfix}"
                )

                mmcif_dict["_atom_site.group_PDB"].append(
                    "ATOM" if is_polymer_residue else "HETATM"
                )
                mmcif_dict["_atom_site.id"].append(str(atom_index))
                mmcif_dict["_atom_site.type_symbol"].append(type_symbol)
                mmcif_dict["_atom_site.label_atom_id"].append(unique_atom_name)
                mmcif_dict["_atom_site.label_alt_id"].append(".")
                mmcif_dict["_atom_site.label_comp_id"].append(res_name_3)
                mmcif_dict["_atom_site.label_asym_id"].append(chain_ids[unique_chain_index[i]])
                mmcif_dict["_atom_site.label_entity_id"].append(
                    label_asym_id_to_entity_id[chain_ids[unique_chain_index[i]]]
                )
                mmcif_dict["_atom_site.label_seq_id"].append(str(unique_residue_index[i]))
                mmcif_dict["_atom_site.pdbx_PDB_ins_code"].append(".")
                mmcif_dict["_atom_site.Cartn_x"].append(f"{pos[0]:.3f}")
                mmcif_dict["_atom_site.Cartn_y"].append(f"{pos[1]:.3f}")
                mmcif_dict["_atom_site.Cartn_z"].append(f"{pos[2]:.3f}")
                mmcif_dict["_atom_site.occupancy"].append("1.00")
                mmcif_dict["_atom_site.B_iso_or_equiv"].append(f"{b_factor:.2f}")
                mmcif_dict["_atom_site.auth_seq_id"].append(str(unique_residue_index[i]))
                mmcif_dict["_atom_site.auth_comp_id"].append(res_name_3)
                mmcif_dict["_atom_site.auth_asym_id"].append(chain_ids[unique_chain_index[i]])
                mmcif_dict["_atom_site.auth_atom_id"].append(unique_atom_name)
                mmcif_dict["_atom_site.pdbx_PDB_model_num"].append("1")

                atom_index += 1

    init_metadata_dict = (
        remove_metadata_fields_by_prefixes(orig_mmcif_metadata, MMCIF_PREFIXES_TO_DROP_POST_AF3)
        if insert_alphafold_mmcif_metadata
        else remove_metadata_fields_by_prefixes(
            orig_mmcif_metadata, MMCIF_PREFIXES_TO_DROP_POST_PARSING
        )
    )
    metadata_dict = mmcif_metadata.add_metadata_to_mmcif(
        mmcif_dict, insert_alphafold_mmcif_metadata=insert_alphafold_mmcif_metadata
    )
    init_metadata_dict.update(metadata_dict)
    init_metadata_dict.update(mmcif_dict)

    return _create_mmcif_string(init_metadata_dict)


@typecheck
@functools.lru_cache(maxsize=256)
def _int_id_to_str_id(num: IntType) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    :param num: A positive integer.

    :return: A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


@typecheck
def _get_chain_seq(
    restypes: np.ndarray,
    residue_indices: np.ndarray,
    chain_indices: np.ndarray,
    chemids: np.ndarray,
    chemtypes: np.ndarray,
    gapless: bool = True,
    non_polymer_only: bool = False,
    non_polymer_chemtype: int = 3,
) -> Dict[IntType, Tuple[List[IntType], List[str], List[IntType]]]:
    """Constructs (as desired) gapless residue index, chemid, and chemtype lists for each chain.

    :param restypes: A numpy array with restypes.
    :param residue_indices: A numpy array with residue indices.
    :param chain_indices: A numpy array with chain indices.
    :param chemids: A numpy array with residue chemical IDs.
    :param chemtypes: A numpy array with residue chemical types.
    :param gapless: If True, the output will contain gapless residue indices.
    :param non_polymer_only: If True, only non-polymer residues are included in the output.
    :param non_polymer_chemtype: The chemical type index of non-polymer residues.

    :return: A dictionary mapping chain indices to a tuple with a list of residue indices,
      a list of chemids, and a list of chemtypes. Missing residues are filled with the
      unknown residue type of the corresponding residue chemical type (e.g., `N`, the
      unknown RNA type, for a majority-RNA chain). Ligand residues are not present in the
      output unless non_polymer_only is True.
    """
    if (
        restypes.shape[0] != residue_indices.shape[0]
        or restypes.shape[0] != chain_indices.shape[0]
    ):
        raise ValueError("restypes, residue_indices, chain_indices must have the same length.")

    # Mask the inputs.
    mask = (
        chemtypes == non_polymer_chemtype
        if non_polymer_only
        else chemtypes != non_polymer_chemtype
    )
    masked_chain_indices = chain_indices[mask]
    masked_residue_indices = residue_indices[mask]
    masked_restypes = restypes[mask]
    masked_chemtypes = chemtypes[mask]
    masked_chemids = chemids[mask]

    # Group the present residues by chain index.
    present = collections.defaultdict(list)
    if not all(
        x.shape[0] for x in (masked_chain_indices, masked_residue_indices, masked_restypes)
    ):
        return {}
    for chain_index, residue_index, restype in zip(
        masked_chain_indices, masked_residue_indices, masked_restypes
    ):
        present[chain_index].append((residue_index, restype))

    # Add any missing residues (from 1 to the first residue and for any gaps).
    chain_seq = {}
    for chain_index, present_residues in present.items():
        present_residue_indices = list(
            dict.fromkeys(x[0] for x in present_residues)
        )  # Preserve order.
        min_res_id = min(present_residue_indices)  # Could be negative.
        max_res_id = max(present_residue_indices)

        res_chemindex = np_mode(masked_chemtypes[masked_chain_indices == chain_index])[0].item()
        residue_constants = get_residue_constants(res_chem_index=res_chemindex)

        new_residue_indices = []
        new_chemids = []
        new_chemtypes = []
        present_index = 0
        residue_indices = (
            present_residue_indices if not gapless else range(min(1, min_res_id), max_res_id + 1)
        )
        for i in residue_indices:
            new_residue_indices.append(i)
            if not gapless or i in present_residue_indices:
                new_chemids.append(
                    masked_chemids[masked_chain_indices == chain_index][present_index]
                )
                new_chemtypes.append(
                    masked_chemtypes[masked_chain_indices == chain_index][present_index]
                )
                present_index += 1
            else:
                # Unknown residue type of the most common residue chemical type in the chain.
                new_chemids.append(residue_constants.unk_restype)
                new_chemtypes.append(res_chemindex)
        chain_seq[chain_index] = (new_residue_indices, new_chemids, new_chemtypes)
    return chain_seq


@typecheck
def _create_mmcif_string(mmcif_dict: Dict[str, Any]) -> str:
    """Converts mmCIF dictionary into mmCIF string."""
    mmcifio = MMCIFIO()
    mmcifio.set_dict(mmcif_dict)

    with io.StringIO() as file_handle:
        mmcifio.save(file_handle)
        return file_handle.getvalue()
