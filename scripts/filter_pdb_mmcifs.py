# %% [markdown]
# # Curating AlphaFold 3 PDB Dataset
#
# For training AlphaFold 3, we follow the training procedure outlined in Abramson et al (2024).
#
# Filtering of targets:
# 1. The structure must have been released to the PDB before the cutoff date of 2021-09-30.
# 2. The structure must have a reported resolution of 9 Å or less.
# 3. The maximum number of polymer chains in a considered structure is 300 for training and 1000 for evaluation.
# 4. Any polymer chain containing fewer than 4 resolved molecules is filtered out.
#
# Filtering of bioassemblies:
# 1. Hydrogens are removed.
# 2. Polymer chains with all unknown molecules are removed.
# 3. Clashing chains are removed. Clashing chains are defined as those with >30% of atoms within 1.7 Å of an atom
# in another chain. If two chains are clashing with each other, the chain with the greater percentage of clashing
# atoms will be removed. If the same fraction of atoms are clashing, the chain with fewer total atoms is removed.
# If the chains have the same number of atoms, then the chain with the larger chain id is removed.
# 4. For molecules or small molecules with CCD codes, atoms outside of the CCD code’s defined set of atom names are
# removed.
# 5. Leaving atoms (ligand atom or groups of atoms that detach when bonds form) for covalent ligands are filtered
# out.
# 6. Protein chains with consecutive Cα atoms >10 Å apart are filtered out.
# 7. For bioassemblies with greater than 20 chains, we select a random interface token (with a centre atom <15 Å to
# the centre atom of a token in another chain) and select the closest 20 chains to this token based on minimum
# distance between any tokens centre atom.
# 8. Crystallization aids are removed if the mmCIF method information indicates that crystallography was used (see
# Table 9).
#

# %%
from __future__ import annotations

import argparse
import glob
import os
import random
from operator import itemgetter
from typing import Dict, List, Literal, Set, Tuple
from datetime import datetime

import numpy as np
import timeout_decorator
from Bio.PDB.NeighborSearch import NeighborSearch
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.ccd_reader import CCDReaderResult
from tqdm.contrib.concurrent import process_map

from alphafold3_pytorch.common.paper_constants import (
    CRYSTALLOGRAPHY_METHODS,
    LIGAND_EXCLUSION_SET,
    NUCLEIC_ACID_RESIDUE_CENTER_ATOMS,
    PROTEIN_RESIDUE_CENTER_ATOMS,
)
from alphafold3_pytorch.data import mmcif_parsing, mmcif_writing
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject
from alphafold3_pytorch.tensor_typing import AtomType, ResidueType, TokenType, typecheck
from alphafold3_pytorch.utils.data_utils import (
    get_biopython_chain_residue_by_composite_id,
    is_polymer,
    is_water,
)
from alphafold3_pytorch.utils.utils import exists

# Constants

FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT = (
    600  # Maximum time allocated to filter a single structure (in seconds)
)

EVALUATION_SPLITS = {"eval", "test"}

# Helper functions


@typecheck
def impute_missing_assembly_metadata(
    mmcif_object: MmcifObject, asym_mmcif_object: MmcifObject
) -> MmcifObject:
    """Impute missing assembly metadata from the asymmetric unit mmCIF."""
    mmcif_object.header.update(asym_mmcif_object.header)
    mmcif_object.bonds.extend(asym_mmcif_object.bonds)

    # Impute structure method
    if (
        "_exptl.method" not in mmcif_object.raw_string
        and "_exptl.method" in asym_mmcif_object.raw_string
    ):
        mmcif_object.raw_string["_exptl.method"] = asym_mmcif_object.raw_string["_exptl.method"]

    # Impute release date
    if (
        "_pdbx_audit_revision_history.revision_date" not in mmcif_object.raw_string
        and "_pdbx_audit_revision_history.revision_date" in asym_mmcif_object.raw_string
    ):
        mmcif_object.raw_string[
            "_pdbx_audit_revision_history.revision_date"
        ] = asym_mmcif_object.raw_string["_pdbx_audit_revision_history.revision_date"]

    # Impute resolution
    if (
        "_refine.ls_d_res_high" not in mmcif_object.raw_string
        and "_refine.ls_d_res_high" in asym_mmcif_object.raw_string
    ):
        mmcif_object.raw_string["_refine.ls_d_res_high"] = asym_mmcif_object.raw_string[
            "_refine.ls_d_res_high"
        ]
    if (
        "_em_3d_reconstruction.resolution" not in mmcif_object.raw_string
        and "_em_3d_reconstruction.resolution" in asym_mmcif_object.raw_string
    ):
        mmcif_object.raw_string["_em_3d_reconstruction.resolution"] = asym_mmcif_object.raw_string[
            "_em_3d_reconstruction.resolution"
        ]
    if (
        "_reflns.d_resolution_high" not in mmcif_object.raw_string
        and "_reflns.d_resolution_high" in asym_mmcif_object.raw_string
    ):
        mmcif_object.raw_string["_reflns.d_resolution_high"] = asym_mmcif_object.raw_string[
            "_reflns.d_resolution_high"
        ]

    # Impute structure connectivity
    for key in asym_mmcif_object.raw_string:
        if key.startswith("_struct_conn.") and key not in mmcif_object.raw_string:
            mmcif_object.raw_string[key] = asym_mmcif_object.raw_string[key]

    return mmcif_object


@typecheck
def filter_pdb_release_date(
    mmcif_object: MmcifObject,
    min_cutoff_date: datetime = datetime(1970, 1, 1),
    max_cutoff_date: datetime = datetime(2021, 9, 30),
) -> bool:
    """Filter based on PDB release date."""
    return (
        "release_date" in mmcif_object.header
        and exists(mmcif_object.header["release_date"])
        and min_cutoff_date
        <= datetime.strptime(mmcif_object.header["release_date"], "%Y-%m-%d")
        <= max_cutoff_date
    )


@typecheck
def filter_resolution(mmcif_object: MmcifObject, max_resolution: float = 9.0) -> bool:
    """Filter based on resolution."""
    return (
        "resolution" in mmcif_object.header
        and exists(mmcif_object.header["resolution"])
        and mmcif_object.header["resolution"] <= max_resolution
    )


@typecheck
def filter_polymer_chains(
    mmcif_object: MmcifObject, max_chains: int = 1000, for_training: bool = False
) -> bool:
    """Filter based on number of polymer chains."""
    polymer_chains = [
        chain
        for chain in mmcif_object.structure.get_chains()
        if any(
            is_polymer(mmcif_object.all_chem_comp_details[chain.id][res_index].type)
            for res_index in range(len(mmcif_object.chain_to_seqres[chain.id]))
        )
    ]
    return len(polymer_chains) <= (300 if for_training else max_chains)


@typecheck
def filter_resolved_chains(
    mmcif_object: MmcifObject, minimum_polymer_residues: int = 4
) -> MmcifObject | None:
    """Filter polymer chains based on number of resolved residues."""
    chains_to_remove = {
        mmcif_object.structure[chain.id].get_full_id()
        for chain in mmcif_object.structure.get_chains()
        if any(
            is_polymer(mmcif_object.all_chem_comp_details[chain.id][res_index].type)
            for res_index in range(len(mmcif_object.chain_to_seqres[chain.id]))
        )
        and len(
            [
                res_index
                for res_index in range(len(mmcif_object.chain_to_seqres[chain.id]))
                if is_polymer(mmcif_object.all_chem_comp_details[chain.id][res_index].type)
                and not mmcif_object.seqres_to_structure[chain.id][res_index].is_missing
            ]
        )
        < minimum_polymer_residues
    }
    mmcif_object.chains_to_remove.update(chains_to_remove)

    return (
        None if len(mmcif_object.chains_to_remove) == len(mmcif_object.structure) else mmcif_object
    )


@typecheck
def prefilter_target(
    mmcif_object: MmcifObject,
    min_cutoff_date: datetime = datetime(1970, 1, 1),
    max_cutoff_date: datetime = datetime(2021, 9, 30),
) -> MmcifObject | None:
    """Pre-filter a target based on various criteria."""
    target_passes_prefilters = (
        filter_pdb_release_date(
            mmcif_object, min_cutoff_date=min_cutoff_date, max_cutoff_date=max_cutoff_date
        )
        and filter_resolution(mmcif_object)
        and filter_polymer_chains(mmcif_object)
    )
    return filter_resolved_chains(mmcif_object) if target_passes_prefilters else None


@typecheck
def remove_hydrogens(mmcif_object: MmcifObject, remove_waters: bool = False) -> MmcifObject:
    """Identify hydrogens (and optionally waters) to remove from a structure."""
    atoms_to_remove = set()
    residues_to_remove = set()
    chains_to_remove = set()

    for chain in mmcif_object.structure.get_chains():
        res_to_remove = set()
        for res in chain:
            res_atoms_to_remove = {
                atom.get_full_id() for atom in res.get_atoms() if atom.element == "H"
            }
            if remove_waters and is_water(res.resname):
                res_to_remove.add(res.get_full_id())
            if len(res_atoms_to_remove) == len(res):  # If no atoms are left in the residue
                res_to_remove.add(res.get_full_id())
            atoms_to_remove.update(res_atoms_to_remove)
        if len(res_to_remove) == len(chain):  # If no residues are left in the chain
            chains_to_remove.add(chain.get_full_id())
        residues_to_remove.update(res_to_remove)

    mmcif_object.atoms_to_remove.update(atoms_to_remove)
    mmcif_object.residues_to_remove.update(residues_to_remove)
    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_polymer_chains_with_all_unknown_residues(mmcif_object: MmcifObject) -> MmcifObject:
    """Identify polymer chains with all unknown residues to remove."""
    chains_to_remove = {
        chain.get_full_id()
        for chain in mmcif_object.structure.get_chains()
        if any(
            is_polymer(mmcif_object.all_chem_comp_details[chain.id][res_index].type)
            for res_index in range(len(mmcif_object.chain_to_seqres[chain.id]))
        )
        and not any(
            mmcif_object.chain_to_seqres[chain.id][res_index] != "X"
            for res_index in range(len(mmcif_object.chain_to_seqres[chain.id]))
        )
    }

    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_clashing_chains(
    mmcif_object: MmcifObject, clash_threshold: float = 1.7, clash_percentage: float = 0.3
) -> MmcifObject:
    """Identify clashing chains to remove."""
    all_atoms = list(mmcif_object.structure.get_atoms())
    neighbor_search = NeighborSearch(all_atoms)

    clashing_chains = []
    chains = list(mmcif_object.structure.get_chains())
    chain_atoms = {chain.id: set(chain.get_atoms()) for chain in chains}

    close_atoms = neighbor_search.search_all(clash_threshold, level="A")
    atom_pairs = [(atom1, atom2) for atom1, atom2 in close_atoms]

    # Find clashing chains
    for i, chain1 in enumerate(chains):
        for chain2 in chains[i + 1 :]:
            chain1_atoms = chain_atoms[chain1.id]
            chain2_atoms = chain_atoms[chain2.id]

            clash_count = sum(
                1 for atom1, atom2 in atom_pairs if atom1 in chain1_atoms and atom2 in chain2_atoms
            )
            if (
                clash_count / len(chain1_atoms) > clash_percentage
                or clash_count / len(chain2_atoms) > clash_percentage
            ):
                clashing_chains.append((chain1, chain2, clash_count))

    chains_to_remove = set()
    for chain1, chain2, clash_count in clashing_chains:
        len_chain1_atoms = len(chain_atoms[chain1.id])
        len_chain2_atoms = len(chain_atoms[chain2.id])

        chain1_clash_ratio = clash_count / len_chain1_atoms
        chain2_clash_ratio = clash_count / len_chain2_atoms

        if (
            chain1_clash_ratio > chain2_clash_ratio
            and chain1.get_full_id() not in chains_to_remove
        ):
            chains_to_remove.add(chain1.get_full_id())
        elif (
            chain2_clash_ratio > chain1_clash_ratio
            and chain2.get_full_id() not in chains_to_remove
        ):
            chains_to_remove.add(chain2.get_full_id())
        else:
            if (
                len_chain1_atoms < len_chain2_atoms
                and chain1.get_full_id() not in chains_to_remove
            ):
                chains_to_remove.add(chain1.get_full_id())
            elif (
                len_chain2_atoms < len_chain1_atoms
                and chain2.get_full_id() not in chains_to_remove
            ):
                chains_to_remove.add(chain2.get_full_id())
            else:
                if chain1.id > chain2.id and chain1.get_full_id() not in chains_to_remove:
                    chains_to_remove.add(chain1.get_full_id())
                elif chain2.get_full_id() not in chains_to_remove:
                    chains_to_remove.add(chain2.get_full_id())

    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_excluded_ligands(
    mmcif_object: MmcifObject, ligand_exclusion_set: Set[str]
) -> MmcifObject:
    """
    Identify ligands in the ligand exclusion set to be removed.

    NOTE: Here, we remove all excluded ligands, even though
    the AlphaFold 3 supplement doesn't mention removing them.
    """
    residues_to_remove = set()
    chains_to_remove = set()

    for chain in mmcif_object.structure.get_chains():
        res_to_remove = set()
        for res in chain:
            if res.resname in ligand_exclusion_set:
                res_to_remove.add(res.get_full_id())
        if len(res_to_remove) == len(chain):
            chains_to_remove.add(chain.get_full_id())
        residues_to_remove.update(res_to_remove)

    mmcif_object.residues_to_remove.update(residues_to_remove)
    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_non_ccd_atoms(
    mmcif_object: MmcifObject, ccd_reader_results: Dict[str, CCDReaderResult]
) -> MmcifObject:
    """Identify atoms not in the corresponding CCD code set to remove."""
    atoms_to_remove = set()
    residues_to_remove = set()
    chains_to_remove = set()

    for chain in mmcif_object.structure.get_chains():
        res_to_remove = set()
        for res in chain:
            if res.resname in ccd_reader_results:
                ccd_atoms = ccd_reader_results[res.resname].component.atoms_ids
                res_atoms_to_remove = {
                    atom.get_full_id() for atom in res.get_atoms() if atom.id not in ccd_atoms
                }
                if len(res_atoms_to_remove) == len(res):
                    res_to_remove.add(res.get_full_id())
                atoms_to_remove.update(res_atoms_to_remove)
        if len(res_to_remove) == len(chain):
            chains_to_remove.add(chain.get_full_id())
        residues_to_remove.update(res_to_remove)

    mmcif_object.atoms_to_remove.update(atoms_to_remove)
    mmcif_object.residues_to_remove.update(residues_to_remove)
    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_leaving_atoms(
    mmcif_object: MmcifObject, ccd_reader_results: Dict[str, CCDReaderResult]
) -> MmcifObject:
    """
    Identify leaving atoms to remove from covalent ligands.

    NOTE: We rely on the PDB's `struct_conn` and the CCD's
    `pdbx_leaving_atom_flag` metadata to discern leaving
    atoms within each covalent ligand once a covalent
    ligand has been structurally identified.

    NOTE: This implementation assumes that if a ligand atom is covalently
    bonded to any other atom, then any leaving atom within the residue
    to which the ligand atom belongs should be removed (in contrast to any
    leaving atom within the "entire chain" to which the ligand atom belongs).
    """
    atoms_to_remove = set()

    for bond in mmcif_object.bonds:
        if bond.conn_type_id.lower() != "covale":
            continue
        bond_chain_ids_in_structure = (
            bond.ptnr1_auth_asym_id in mmcif_object.structure
            and bond.ptnr2_auth_asym_id in mmcif_object.structure
        )
        if bond.pdbx_leaving_atom_flag in {"one", "both"} and bond_chain_ids_in_structure:
            # Identify the chemical types of the residues of the "partner 1" and "partner 2" bond atoms in the structure
            ptnr1_chain = mmcif_object.structure[bond.ptnr1_auth_asym_id]
            ptnr2_chain = mmcif_object.structure[bond.ptnr2_auth_asym_id]
            ptnr1_res = get_biopython_chain_residue_by_composite_id(
                ptnr1_chain, bond.ptnr1_auth_comp_id, int(bond.ptnr1_auth_seq_id)
            )
            ptnr2_res = get_biopython_chain_residue_by_composite_id(
                ptnr2_chain, bond.ptnr2_auth_comp_id, int(bond.ptnr2_auth_seq_id)
            )
            # NOTE: This is the main bottleneck of the function, since (for each chain)
            # we need a zero-based index into the residue's chemical component details
            ptnr1_res_index = list(ptnr1_chain).index(ptnr1_res)
            ptnr2_res_index = list(ptnr2_chain).index(ptnr2_res)
            ptnr1_res_is_ligand = not is_polymer(
                mmcif_object.chem_comp_details[ptnr1_chain.id][ptnr1_res_index].type
            )
            ptnr2_res_is_ligand = not is_polymer(
                mmcif_object.chem_comp_details[ptnr2_chain.id][ptnr2_res_index].type
            )

            # Remove all leaving atoms in the "partner 1" covalent ligand residue
            if ptnr1_res_is_ligand and bond.ptnr1_auth_comp_id in ccd_reader_results:
                ptnr1_atom_id_leaving_atom_table = ccd_reader_results[
                    bond.ptnr1_auth_comp_id
                ].component.ccd_cif_block.find(
                    "_chem_comp_atom.", ["atom_id", "pdbx_leaving_atom_flag"]
                )
                for row in ptnr1_atom_id_leaving_atom_table:
                    if row["pdbx_leaving_atom_flag"] == "Y" and row["atom_id"] in ptnr1_res:
                        atoms_to_remove.add(ptnr1_res[row["atom_id"]].get_full_id())

            # Remove all leaving atoms in the "partner 2" covalent ligand residue
            if ptnr2_res_is_ligand and bond.ptnr2_auth_comp_id in ccd_reader_results:
                ptnr2_atom_id_leaving_atom_table = ccd_reader_results[
                    bond.ptnr2_auth_comp_id
                ].component.ccd_cif_block.find(
                    "_chem_comp_atom.", ["atom_id", "pdbx_leaving_atom_flag"]
                )
                for row in ptnr2_atom_id_leaving_atom_table:
                    if row["pdbx_leaving_atom_flag"] == "Y" and row["atom_id"] in ptnr2_res:
                        atoms_to_remove.add(ptnr2_res[row["atom_id"]].get_full_id())

    mmcif_object.atoms_to_remove.update(atoms_to_remove)

    return mmcif_object


@typecheck
def filter_large_ca_distances(
    mmcif_object: MmcifObject,
    max_distance: float = 10.0,
    min_num_residues_for_protein_classification: int = 10,
) -> MmcifObject:
    """
    Identify chains (to be removed) with any large sequential Ca-Ca atom distances.

    NOTE: This function currently does not account for residues
    with alternative Ca atom locations.
    """
    chains_to_remove = set()

    for chain in mmcif_object.structure.get_chains():
        num_peptide_residues = sum(
            1
            for res_index in range(len(chain))
            if "peptide" in mmcif_object.chem_comp_details[chain.id][res_index].type.lower()
        )
        if num_peptide_residues < min_num_residues_for_protein_classification:
            # Only filter protein chains
            continue
        ca_atoms = [
            res["CA"]
            for (res_index, res) in enumerate(chain)
            if "peptide" in mmcif_object.chem_comp_details[chain.id][res_index].type.lower()
            and "CA" in res
        ]
        ca_ca_distance_in_range = True
        for i, ca1 in enumerate(ca_atoms[:-1]):
            ca2 = ca_atoms[i + 1]
            if (ca1 - ca2) > max_distance:
                ca_ca_distance_in_range = False
                break

        if len(ca_atoms) > 1 and not ca_ca_distance_in_range:
            chains_to_remove.add(chain.get_full_id())

    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def select_closest_chains(
    mmcif_object: MmcifObject,
    protein_residue_center_atoms: Dict[str, str],
    nucleic_acid_residue_center_atoms: Dict[str, str],
    max_chains: int = 20,
) -> MmcifObject:
    """Identify the closest chains in large bioassemblies."""

    @typecheck
    def get_tokens_from_residues(
        residues: List[ResidueType],
        protein_residue_center_atoms: Dict[str, str],
        nucleic_acid_residue_center_atoms: Dict[str, str],
    ) -> List[TokenType]:
        """Get tokens from residues."""
        tokens = []
        for res in residues:
            if (
                res.resname in protein_residue_center_atoms
                or res.resname in nucleic_acid_residue_center_atoms
            ):
                tokens.append(res)
            else:
                for atom in res:
                    tokens.append(atom)
        return tokens

    @typecheck
    def get_token_center_atom(
        token: TokenType,
        protein_residue_center_atoms: Dict[str, str],
        nucleic_acid_residue_center_atoms: Dict[str, str],
    ) -> AtomType:
        """Get center atom of a token."""
        if isinstance(token, ResidueType):
            if token.resname in protein_residue_center_atoms:
                token_center_atom = token[protein_residue_center_atoms[token.resname]]
            elif token.resname in nucleic_acid_residue_center_atoms:
                token_center_atom = token[nucleic_acid_residue_center_atoms[token.resname]]
        else:
            token_center_atom = token
        return token_center_atom

    @typecheck
    def get_token_center_atoms(
        tokens: List[TokenType],
        protein_residue_center_atoms: Dict[str, str],
        nucleic_acid_residue_center_atoms: Dict[str, str],
    ) -> List[AtomType]:
        """Get center atoms of tokens."""
        token_center_atoms = []
        for token in tokens:
            token_center_atom = get_token_center_atom(
                token, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
            )
            token_center_atoms.append(token_center_atom)
        return token_center_atoms

    @typecheck
    def get_token_chain_id(token: TokenType) -> str:
        """Get chain ID of a token."""
        if isinstance(token, AtomType):
            # For an Atom, navigate up twice to get to the Chain level
            return token.get_parent().get_parent().id
        elif isinstance(token, ResidueType):
            # For a Residue, navigate up once to get to the Chain level
            return token.get_parent().id

    @typecheck
    def get_interface_tokens(
        tokens: List[TokenType],
        protein_residue_center_atoms: Dict[str, str],
        nucleic_acid_residue_center_atoms: Dict[str, str],
        center_atom_interaction_distance: float = 15.0,
    ) -> List[TokenType]:
        """Get interface tokens."""
        token_center_atoms = get_token_center_atoms(
            tokens, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
        )
        token_center_atoms_coords_array = np.array([atom.coord for atom in token_center_atoms])
        token_chains_array = np.array([get_token_chain_id(token) for token in tokens])

        # Compute all pairwise distances
        differences = (
            token_center_atoms_coords_array[:, None, :]
            - token_center_atoms_coords_array[None, :, :]
        )
        distances = np.linalg.norm(differences, axis=2)

        # Create masks
        token_self_mask = distances != 0
        token_interface_mask = token_chains_array[:, None] != token_chains_array[None, :]
        token_interaction_mask = distances < center_atom_interaction_distance
        interface_distance_mask = token_self_mask & token_interface_mask & token_interaction_mask

        interface_tokens = set()
        for token_index, token in enumerate(tokens):
            if np.any(interface_distance_mask[token_index, :]):
                interface_tokens.add(token)

        return list(interface_tokens)

    chains_to_remove = set()
    if (len(mmcif_object.structure) - len(mmcif_object.chains_to_remove)) > max_chains:
        chains = [
            chain
            for chain in mmcif_object.structure.get_chains()
            if chain.get_full_id() not in mmcif_object.chains_to_remove
        ]
        residues = [res for chain in chains for res in chain]
        tokens = get_tokens_from_residues(
            residues, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
        )
        interface_tokens = get_interface_tokens(
            tokens, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
        )
        random_interface_token = random.choice(interface_tokens)
        chain_min_token_distances = []
        for chain in chains:
            chain_residues = list(chain)
            chain_tokens = get_tokens_from_residues(
                chain_residues, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
            )
            chain_token_center_atoms = get_token_center_atoms(
                chain_tokens, protein_residue_center_atoms, nucleic_acid_residue_center_atoms
            )
            chain_min_token_distance = min(
                atom
                - get_token_center_atom(
                    random_interface_token,
                    protein_residue_center_atoms,
                    nucleic_acid_residue_center_atoms,
                )
                for atom in chain_token_center_atoms
            )
            chain_min_token_distances.append((chain.id, chain_min_token_distance))

        chain_min_token_distances.sort(key=itemgetter(1))
        for chain_id, _ in chain_min_token_distances[max_chains:]:
            chains_to_remove.add(mmcif_object.structure[chain_id].get_full_id())

    mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
def remove_crystallization_aids(
    mmcif_object: MmcifObject, crystallography_methods: Dict[str, Set[str]]
) -> MmcifObject:
    """Identify crystallization aids to remove."""
    if (
        "structure_method" in mmcif_object.header
        and exists(mmcif_object.header["structure_method"])
        and mmcif_object.header["structure_method"].upper() in crystallography_methods
    ):
        residues_to_remove = set()
        chains_to_remove = set()

        structure_method_crystallization_aids = crystallography_methods[
            mmcif_object.header["structure_method"].upper()
        ]
        for chain in mmcif_object.structure.get_chains():
            res_to_remove = set()
            for res in chain:
                if res.resname in structure_method_crystallization_aids:
                    res_to_remove.add(res.get_full_id())
            if len(res_to_remove) == len(chain):
                chains_to_remove.add(chain.get_full_id())
            residues_to_remove.update(res_to_remove)

        mmcif_object.residues_to_remove.update(residues_to_remove)
        mmcif_object.chains_to_remove.update(chains_to_remove)

    return mmcif_object


@typecheck
@timeout_decorator.timeout(FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT, use_signals=False)
def filter_structure_with_timeout(
    filepath: str,
    output_dir: str,
    min_cutoff_date: datetime = datetime(1970, 1, 1),
    max_cutoff_date: datetime = datetime(2021, 9, 30),
    split: Literal["train", "eval", "test"] = "train",
):
    """
    Given an input mmCIF file, create a new filtered mmCIF file
    using AlphaFold 3's PDB dataset filtering criteria under a
    timeout constraint.
    """
    # Section 2.5.4 of the AlphaFold 3 supplement
    asym_filepath = os.path.join(
        os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
        os.path.basename(filepath).replace("-assembly1", ""),
    )
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")
    os.makedirs(output_file_dir, exist_ok=True)

    # Filtering of targets
    mmcif_object = mmcif_parsing.parse_mmcif_object(filepath, file_id)
    asym_mmcif_object = mmcif_parsing.parse_mmcif_object(asym_filepath, file_id)
    # NOTE: The assembly mmCIF does not contain the full header or the
    # structure connectivity (i.e., bond) information, so we impute
    # these from the asymmetric unit mmCIF
    mmcif_object = impute_missing_assembly_metadata(mmcif_object, asym_mmcif_object)
    mmcif_object = prefilter_target(
        mmcif_object, min_cutoff_date=min_cutoff_date, max_cutoff_date=max_cutoff_date
    )
    if not exists(mmcif_object):
        print(f"Skipping target due to prefiltering: {file_id}")
        return
    # Filtering of bioassemblies
    # NOTE: Here, we remove waters even though the AlphaFold 3 supplement doesn't mention removing them during filtering
    mmcif_object = remove_hydrogens(mmcif_object, remove_waters=True)
    mmcif_object = remove_polymer_chains_with_all_unknown_residues(mmcif_object)
    mmcif_object = remove_clashing_chains(mmcif_object)
    if split in EVALUATION_SPLITS:
        # NOTE: The AlphaFold 3 supplement suggests the training dataset retains these (excluded) ligands
        mmcif_object = remove_excluded_ligands(mmcif_object, LIGAND_EXCLUSION_SET)
    mmcif_object = remove_non_ccd_atoms(mmcif_object, CCD_READER_RESULTS)
    mmcif_object = remove_leaving_atoms(mmcif_object, CCD_READER_RESULTS)
    mmcif_object = filter_large_ca_distances(mmcif_object)
    mmcif_object = select_closest_chains(
        # NOTE: Modified amino acid and nucleotide residues are
        # treated as N-atom ligands in this (structural) filtering step
        mmcif_object,
        PROTEIN_RESIDUE_CENTER_ATOMS,
        NUCLEIC_ACID_RESIDUE_CENTER_ATOMS,
    )
    mmcif_object = remove_crystallization_aids(mmcif_object, CRYSTALLOGRAPHY_METHODS)
    if len(mmcif_object.chains_to_remove) < len(mmcif_object.structure):
        # Save a filtered structure as an mmCIF file along with its latest metadata
        mmcif_object = mmcif_parsing.filter_mmcif(mmcif_object)
        mmcif_writing.write_mmcif(
            mmcif_object,
            output_filepath,
            gapless_poly_seq=True,
            insert_orig_atom_names=True,
            insert_alphafold_mmcif_metadata=False,
        )
        print(f"Finished filtering structure: {mmcif_object.file_id}")


@typecheck
def filter_structure(args: Tuple[str, str, datetime, datetime, str]):
    """
    Given an input mmCIF file, create a new filtered mmCIF file
    using AlphaFold 3's PDB dataset filtering criteria.
    """
    filepath, output_dir, min_cutoff_date, max_cutoff_date, split = args
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")

    try:
        filter_structure_with_timeout(
            filepath,
            output_dir,
            min_cutoff_date=min_cutoff_date,
            max_cutoff_date=max_cutoff_date,
            split=split,
        )
    except Exception as e:
        print(f"Skipping structure filtering of {filepath} due to: {e}")
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except Exception as e:
                print(
                    f"Failed to remove partially filtered file {output_filepath} due to: {e}. Skipping its removal..."
                )


if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(
        description="Filter mmCIF files to curate the AlphaFold 3 PDB dataset."
    )
    parser.add_argument(
        "-i",
        "--mmcif_assembly_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        help="Path to the input directory containing `assembly1` mmCIF files to filter.",
    )
    parser.add_argument(
        "-a",
        "--mmcif_asym_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        help="Path to the input directory containing asymmetric unit mmCIF files with which to filter the `assembly1` mmCIF files.",
    )
    parser.add_argument(
        "-c",
        "--ccd_dir",
        type=str,
        default=os.path.join("data", "ccd_data"),
        help="Path to the directory containing CCD files to reference during data filtering.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "mmcifs"),
        help="Path to the output directory in which to store filtered mmCIF dataset files.",
    )
    parser.add_argument(
        "-f",
        "--min_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(1970, 1, 1),
        help="Minimum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-l",
        "--max_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(2021, 9, 30),
        help="Maximum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip filtering of existing output files.",
    )
    parser.add_argument(
        "-e",
        "--split",
        type=str,
        choices=["train", "eval", "test"],
        default="train",
        help="To which split the filtered dataset should be assigned (i.e., `train`, `eval`, or `test`).",
    )
    parser.add_argument(
        "-n",
        "--no_workers",
        type=int,
        default=16,
        help="Number of workers to use for filtering.",
    )
    parser.add_argument(
        "-w",
        "--chunksize",
        type=int,
        default=1,
        help="How many files should be distributed to each worker at a time.",
    )
    args = parser.parse_args()

    assert os.path.exists(
        args.mmcif_assembly_dir
    ), f"Input assembly directory {args.mmcif_assembly_dir} does not exist."
    assert os.path.exists(
        args.mmcif_asym_dir
    ), f"Input asymmetric unit directory {args.mmcif_asym_dir} does not exist."
    assert os.path.exists(args.ccd_dir), f"CCD directory {args.ccd_dir} does not exist."
    assert os.path.exists(
        os.path.join(args.ccd_dir, "chem_comp_model.cif")
    ), f"CCD ligands file not found in {args.ccd_dir}."
    assert os.path.exists(
        os.path.join(args.ccd_dir, "components.cif")
    ), f"CCD components file not found in {args.ccd_dir}."
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the Chemical Component Dictionary (CCD) into memory

    print("Loading the Chemical Component Dictionary (CCD) into memory...")
    CCD_READER_RESULTS = ccd_reader.read_pdb_components_file(
        # Load globally to share amongst all worker processes
        os.path.join(args.ccd_dir, "components.cif"),
        sanitize=False,  # Reduce loading time
    )
    print("Finished loading the Chemical Component Dictionary (CCD) into memory.")

    # Filter structures across all worker processes

    args_tuples = [
        (
            filepath,
            args.output_dir,
            args.min_cutoff_date,
            args.max_cutoff_date,
            args.split,
        )
        for filepath in glob.glob(os.path.join(args.mmcif_assembly_dir, "*", "*.cif"))
        if "assembly1" in os.path.basename(filepath)
        and os.path.exists(
            os.path.join(
                os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
                os.path.basename(filepath).replace("-assembly1", ""),
            )
        )
        and not (
            args.skip_existing
            and os.path.exists(
                os.path.join(
                    args.output_dir,
                    os.path.splitext(os.path.basename(filepath))[0][1:3],
                    f"{os.path.splitext(os.path.basename(filepath))[0]}.cif",
                )
            )
        )
    ]
    process_map(
        filter_structure,
        args_tuples,
        max_workers=args.no_workers,
        chunksize=args.chunksize,
    )
