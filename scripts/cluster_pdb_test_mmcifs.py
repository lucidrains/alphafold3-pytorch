# %% [markdown]
# # Clustering AlphaFold 3 PDB Evaluation Dataset
#
# For clustering AlphaFold 3's PDB evaluation dataset, we propose a modified (i.e., more stringent) clustering procedure
# inspired by Abramson et al (2024).
#
# With the full evaluation set curated by the PDB test set filtering script we create a “low homology” subset
# that is filtered on homology to the training and validation sets.
# Evaluation is done either on individual chains, or on specific interfaces extracted from the full complex prediction.
# For intra-chain metrics, we keep polymers that have less than 30% sequence identity to the training or validation sets.
# Here we define sequence identity as the percent of residues in the evaluation set chain that are identical to the
# training or validation set chain. For interface metrics the following filters are applied:
# • Polymer-polymer interfaces: If both polymers have greater than 30% sequence identity to two chains in the same
# complex in the training or validation sets, then this interface is filtered out.
# • Polymer-ligand interfaces: If both the polymer and ligand have greater than 30% sequence identity and 0.3 Tanimoto similarity,
# respectively, to two chains in the same complex in the training or validation sets, then this interface is filtered out.

# %%

import argparse
import glob
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import polars as pl
import timeout_decorator
from Bio.Data import PDBData
from Bio.PDB.NeighborSearch import NeighborSearch
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.models.components.inputs import CCD_COMPONENTS_SMILES
from alphafold3_pytorch.tensor_typing import IntType, typecheck
from alphafold3_pytorch.utils.data_utils import (
    RESIDUE_MOLECULE_TYPE,
    get_residue_molecule_type,
)
from alphafold3_pytorch.utils.utils import exists, np_mode

# Constants

CHAIN_SEQUENCES = List[Dict[str, Dict[str, str]]]
CHAIN_INTERFACES = Dict[str, List[str]]
INTERFACE_CLUSTERS = Dict[str, str]
CLUSTERING_MOLECULE_TYPE = Literal["protein", "nucleic_acid", "peptide", "ligand", "unknown"]

PROTEIN_LETTERS_3TO1 = {k.strip(): v.strip() for k, v in PDBData.protein_letters_3to1.items()}
NUCLEIC_LETTERS_3TO1 = {k.strip(): v.strip() for k, v in PDBData.nucleic_letters_3to1.items()}

PROTEIN_LETTERS_3TO1_EXTENDED = {
    k.strip(): v.strip() for k, v in PDBData.protein_letters_3to1_extended.items()
}
NUCLEIC_LETTERS_3TO1_EXTENDED = {
    k.strip(): v.strip() for k, v in PDBData.nucleic_letters_3to1_extended.items()
}

CLUSTERING_POLYMER_MOLECULE_TYPES = {"protein", "rna", "dna", "peptide"}
PROTEIN_LETTERS_1TO3 = {k.strip(): v.strip() for k, v in PDBData.protein_letters_1to3.items()}
RNA_LETTERS_1TO3 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "U",  # NOTE: This mapping is required for PDBs such as `41Od`
    "U": "U",
}
DNA_LETTERS_1TO3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
    "U": "DT",  # NOTE: This mapping is present as a precaution based on outlier PDBs such as `410d`
}

IS_NOVEL_LIGAND_MAX_SECONDS_PER_INPUT = (
    20  # Maximum time allocated to check a single ligand for novelty (in seconds)
)


# Helper functions


@typecheck
def convert_modified_residue_three_to_one(
    residue_id: str, residue_mol_type: RESIDUE_MOLECULE_TYPE
) -> Tuple[str, RESIDUE_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).
    Also return the chemically-specific molecule type of the residue.

    NOTE: All unknown residues or unmappable modified residues (be they protein, RNA, or DNA) are
    converted to the unknown residue type of the residue's chemical type (e.g., `N` for RNA).
    """
    # NOTE: If a modified residue cannot be found as a mapping key
    # or is mapped to a value longer than a single character, then
    # it will be mapped to the corresponding unknown residue type.
    is_modified_protein_residue = (
        residue_mol_type == "protein"
        and residue_id not in PROTEIN_LETTERS_3TO1
        and residue_id in PROTEIN_LETTERS_3TO1_EXTENDED
        and len(PROTEIN_LETTERS_3TO1_EXTENDED[residue_id]) == 1
    )
    is_modified_rna_residue = (
        residue_mol_type == "rna"
        and residue_id not in NUCLEIC_LETTERS_3TO1
        and residue_id in NUCLEIC_LETTERS_3TO1_EXTENDED
        and len(NUCLEIC_LETTERS_3TO1_EXTENDED[residue_id]) == 1
    )
    is_modified_dna_residue = (
        residue_mol_type == "dna"
        and residue_id not in NUCLEIC_LETTERS_3TO1
        and residue_id in NUCLEIC_LETTERS_3TO1_EXTENDED
        and len(NUCLEIC_LETTERS_3TO1_EXTENDED[residue_id]) == 1
    )

    # Map modified residues to their one-letter codes, if applicable
    if any((is_modified_protein_residue, is_modified_rna_residue, is_modified_dna_residue)):
        one_letter_mapped_residue = (
            PROTEIN_LETTERS_3TO1_EXTENDED[residue_id]
            if is_modified_protein_residue
            else NUCLEIC_LETTERS_3TO1_EXTENDED[residue_id]
        )
        if is_modified_protein_residue:
            mapped_residue = PROTEIN_LETTERS_1TO3[one_letter_mapped_residue]
        elif is_modified_rna_residue:
            mapped_residue = RNA_LETTERS_1TO3[one_letter_mapped_residue]
        elif is_modified_dna_residue:
            mapped_residue = DNA_LETTERS_1TO3[one_letter_mapped_residue]
    else:
        mapped_residue = residue_id

    if residue_mol_type == "protein":
        return (
            (
                PROTEIN_LETTERS_3TO1[mapped_residue]
                if mapped_residue in PROTEIN_LETTERS_3TO1
                else "X"
            ),
            "protein",
        )
    elif residue_mol_type in {"rna", "dna"}:
        return (
            (
                NUCLEIC_LETTERS_3TO1[mapped_residue]
                if mapped_residue in NUCLEIC_LETTERS_3TO1
                else "X"
            ),
            ("rna" if residue_mol_type == "rna" else "dna"),
        )
    else:
        return mapped_residue, "ligand"


def parse_chain_sequences_and_interfaces_from_mmcif(
    filepath: str,
    assume_one_based_residue_ids: bool = False,
    min_num_residues_for_protein_classification: int = 10,
    interface_distance_threshold: float = 5.0,
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Parse an mmCIF file and return a dictionary mapping chain IDs
    to sequences for all molecule types (i.e., proteins, rna, dna, peptides, ligands, etc.)
    as well as a set of chain ID pairs denoting structural interfaces.
    """
    assert filepath.endswith(".cif"), "The input file must be an mmCIF file."
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    mmcif_object = mmcif_parsing.parse_mmcif_object(filepath, file_id)
    model = mmcif_object.structure

    # NOTE: After dataset filtering, only heavy (non-hydrogen) atoms remain in the structure
    all_atoms = [atom for atom in model.get_atoms()]
    neighbor_search = NeighborSearch(all_atoms)

    sequences = {}
    interface_chain_ids = set()
    for chain in model:
        one_letter_seq_tokens = []
        token_molecule_types = []

        for res_index, res in enumerate(chain):
            # Convert each residue to a one-letter code if applicable
            res_chem_comp = mmcif_object.chem_comp_details[chain.id][res_index]
            res_id = res_chem_comp.id.strip()
            res_mol_type = get_residue_molecule_type(res_chem_comp.type)
            one_letter_residue, clustering_molecule_type = convert_modified_residue_three_to_one(
                res_id, res_mol_type
            )

            if clustering_molecule_type == "ligand":
                # NOTE: Since ligands are clustered based on their CCD codes,
                # we can group same-CCD molecules in the same chain together
                # as a single sequence
                sequences[f"{chain.id}:{clustering_molecule_type}-{res_id}"] = one_letter_residue
            else:
                one_letter_seq_tokens.append(one_letter_residue)
                token_molecule_types.append(clustering_molecule_type)

            # Find all interfaces defined as pairs of chains with minimum heavy atom (i.e. non-hydrogen) separation less than 5 Å
            for atom in res:
                for neighbor in neighbor_search.search(
                    atom.coord, interface_distance_threshold, "R"
                ):
                    neighbor_chain_id = neighbor.get_parent().get_id()
                    if chain.id == neighbor_chain_id:
                        continue
                    # NOTE: We can only make this `ID - 1` assumption because each chain's residue IDs
                    # are 1-indexed after performing PDB dataset filtering. If clustering is being performed
                    # on another (i.e., non-PDB) dataset of mmCIF files, then these zero-based residue indices
                    # need to be identified alternatively (e.g., by performing list indexing on the neighboring
                    # chain's Residue objects).
                    if assume_one_based_residue_ids:
                        neighbor_res_index = neighbor.get_id()[1] - 1
                    else:
                        neighbor_res_index = list(model[neighbor_chain_id]).index(
                            neighbor
                        )  # E.g., for non-PDB datasets
                    neighbor_res_chem_comp = mmcif_object.chem_comp_details[neighbor_chain_id][
                        neighbor_res_index
                    ]
                    neighbor_res_id = neighbor_res_chem_comp.id.strip()
                    neighbor_res_mol_type = get_residue_molecule_type(neighbor_res_chem_comp.type)

                    _, neighbor_clustering_molecule_type = convert_modified_residue_three_to_one(
                        neighbor_res_id, neighbor_res_mol_type
                    )

                    molecule_index_postfix = (
                        f"-{res_id}" if clustering_molecule_type == "ligand" else ""
                    )
                    neighbor_molecule_index_postfix = (
                        f"-{neighbor_res_id}"
                        if neighbor_clustering_molecule_type == "ligand"
                        else ""
                    )

                    # Avoid adding duplicate interface chain pairs
                    atom_interface_key = (
                        f"{chain.id}:{clustering_molecule_type}{molecule_index_postfix}"
                    )
                    neighbor_interface_key = f"{neighbor_chain_id}:{neighbor_clustering_molecule_type}{neighbor_molecule_index_postfix}"
                    if f"{neighbor_interface_key}+{atom_interface_key}" not in interface_chain_ids:
                        interface_chain_ids.add(f"{atom_interface_key}+{neighbor_interface_key}")

        if not one_letter_seq_tokens:
            # NOTE: This indicates that the current chain consists of only ligand residues
            continue

        unique_token_molecule_types = set(token_molecule_types)
        if len(unique_token_molecule_types) > 1:
            # Handle cases where a chain contains multiple polymer molecule types, such as in PDB `5a0f`
            molecule_type = np_mode(np.array(token_molecule_types))[0].item()
            logger.warning(
                f"More than one molecule type found (i.e., {unique_token_molecule_types}) in chain {chain.id} within the mmCIF file {filepath}."
                f" Assigning the most common molecule type to the chain (i.e., {molecule_type}), and setting the type of all outlier residues to the unknown residue type (i.e., X)."
            )
            for token_index in range(len(one_letter_seq_tokens)):
                if token_molecule_types[token_index] != molecule_type:
                    one_letter_seq_tokens[token_index] = "X"
        else:
            molecule_type = token_molecule_types[0]

        if (
            molecule_type == "protein"
            and len(one_letter_seq_tokens) < min_num_residues_for_protein_classification
        ):
            molecule_type = "peptide"

        one_letter_seq = "".join(one_letter_seq_tokens)
        sequences[f"{chain.id}:{molecule_type}"] = one_letter_seq

    return sequences, interface_chain_ids


def parse_chain_sequences_and_interfaces_from_mmcif_file(
    cif_filepath: str, assume_one_based_residue_ids: bool = False
) -> Tuple[str, Dict[str, str], Set[str]]:
    """Parse chain sequences and interfaces from an mmCIF file."""
    structure_id = os.path.splitext(os.path.basename(cif_filepath))[0]
    try:
        chain_sequences, interface_chain_ids = parse_chain_sequences_and_interfaces_from_mmcif(
            cif_filepath, assume_one_based_residue_ids=assume_one_based_residue_ids
        )
        return structure_id, chain_sequences, interface_chain_ids
    except Exception as e:
        logger.warning(
            f"Failed to parse chain sequences and interfaces from mmCIF file '{cif_filepath}' due to: {e}"
        )
        return structure_id, {}, set()


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_directory(
    mmcif_dir: str, max_workers: int = 2, assume_one_based_residue_ids: bool = False
) -> Tuple[CHAIN_SEQUENCES, CHAIN_INTERFACES]:
    """
    Parse all mmCIF files in a directory and return a list of dictionaries mapping chain IDs to sequences
    as well as a dictionary mapping complex IDs to a list of chain ID pairs denoting structural interfaces.
    """
    all_chain_sequences = []
    all_interface_chain_ids = {}

    mmcif_filepaths = list(glob.glob(os.path.join(mmcif_dir, "*", "*.cif")))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                parse_chain_sequences_and_interfaces_from_mmcif_file,
                cif_filepath,
                assume_one_based_residue_ids,
            ): cif_filepath
            for cif_filepath in mmcif_filepaths
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Parsing chain sequences and interfaces",
        ):
            structure_id, chain_sequences, interface_chain_ids = future.result()
            if chain_sequences:
                all_chain_sequences.append({structure_id: chain_sequences})
                all_interface_chain_ids[structure_id] = list(interface_chain_ids)

    return all_chain_sequences, all_interface_chain_ids


@typecheck
def separate_monomer_and_multimer_chain_sequences(
    all_chain_sequences: CHAIN_SEQUENCES,
) -> Tuple[CHAIN_SEQUENCES, CHAIN_SEQUENCES]:
    """Separate monomer and multimer chain sequences."""
    monomer_chain_sequences = []
    multimer_chain_sequences = []
    for chain_sequences in tqdm(
        all_chain_sequences, desc="Separating monomer and multimer chain sequences"
    ):
        chain_ids, molecule_ids, polymer_molecule_ids = [], [], []
        for chain_sequence_dicts in chain_sequences.values():
            for chain_sequence_dict in chain_sequence_dicts:
                chain_id, molecule_id = chain_sequence_dict.split(":")
                chain_ids.append(chain_id)
                molecule_ids.append(molecule_id)
                mol_is_polymer = any(
                    mol_id in molecule_id for mol_id in CLUSTERING_POLYMER_MOLECULE_TYPES
                )
                if mol_is_polymer:
                    polymer_molecule_ids.append(molecule_id)
        if len(polymer_molecule_ids) > 1:
            multimer_chain_sequences.append(chain_sequences)
        else:
            monomer_chain_sequences.append(chain_sequences)

    return monomer_chain_sequences, multimer_chain_sequences


@typecheck
def filter_to_low_homology_sequences(
    input_all_chain_sequences: CHAIN_SEQUENCES,
    reference_all_chain_sequences: CHAIN_SEQUENCES,
    input_interface_chain_ids: CHAIN_INTERFACES,
    input_fasta_filepath: str,
    reference_fasta_filepath: str,
    max_polymer_similarity: float = 0.3,
    max_ligand_similarity: float = 0.3,
    max_workers: int = 2,
) -> Tuple[CHAIN_SEQUENCES, CHAIN_INTERFACES]:
    """Filter targets to only low homology sequences."""
    input_monomer_fasta_filepath = input_fasta_filepath.replace(".fasta", "_monomer.fasta")
    input_multimer_fasta_filepath = input_fasta_filepath.replace(".fasta", "_multimer.fasta")

    reference_monomer_fasta_filepath = reference_fasta_filepath.replace(".fasta", "_monomer.fasta")
    reference_multimer_fasta_filepath = reference_fasta_filepath.replace(
        ".fasta", "_multimer.fasta"
    )

    # Separate monomer and multimer sequences

    (
        input_monomer_chain_sequences,
        input_multimer_chain_sequences,
    ) = separate_monomer_and_multimer_chain_sequences(input_all_chain_sequences)
    (
        reference_monomer_chain_sequences,
        reference_multimer_chain_sequences,
    ) = separate_monomer_and_multimer_chain_sequences(reference_all_chain_sequences)

    # Write monomer and multimer sequences to FASTA files

    write_sequences_to_fasta(
        input_monomer_chain_sequences, input_monomer_fasta_filepath, molecule_type="protein"
    )
    write_sequences_to_fasta(
        input_monomer_chain_sequences, input_monomer_fasta_filepath, molecule_type="nucleic_acid"
    )
    write_sequences_to_fasta(
        input_monomer_chain_sequences, input_monomer_fasta_filepath, molecule_type="peptide"
    )

    write_sequences_to_fasta(
        input_multimer_chain_sequences,
        input_multimer_fasta_filepath,
        molecule_type="protein",
        interface_chain_ids=input_interface_chain_ids,
    )
    write_sequences_to_fasta(
        input_multimer_chain_sequences,
        input_multimer_fasta_filepath,
        molecule_type="nucleic_acid",
        interface_chain_ids=input_interface_chain_ids,
    )
    write_sequences_to_fasta(
        input_multimer_chain_sequences,
        input_multimer_fasta_filepath,
        molecule_type="peptide",
        interface_chain_ids=input_interface_chain_ids,
    )

    write_sequences_to_fasta(
        reference_monomer_chain_sequences,
        reference_monomer_fasta_filepath,
        molecule_type="protein",
    )
    write_sequences_to_fasta(
        reference_monomer_chain_sequences,
        reference_monomer_fasta_filepath,
        molecule_type="nucleic_acid",
    )
    write_sequences_to_fasta(
        reference_monomer_chain_sequences,
        reference_monomer_fasta_filepath,
        molecule_type="peptide",
    )

    write_sequences_to_fasta(
        reference_multimer_chain_sequences,
        reference_multimer_fasta_filepath,
        molecule_type="protein",
    )
    write_sequences_to_fasta(
        reference_multimer_chain_sequences,
        reference_multimer_fasta_filepath,
        molecule_type="nucleic_acid",
    )
    write_sequences_to_fasta(
        reference_multimer_chain_sequences,
        reference_multimer_fasta_filepath,
        molecule_type="peptide",
    )

    # Use MMseqs2 to perform all-against-all sequence identity comparisons for monomers

    input_monomer_protein_sequence_names = search_sequences_using_mmseqs2(
        input_monomer_fasta_filepath,
        reference_monomer_fasta_filepath,
        args.output_dir,
        molecule_type="protein",
        max_seq_id=max_polymer_similarity,
        extra_parameters={
            # force protein mode
            "--dbtype": 1,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
        },
    )
    input_monomer_nucleic_acid_sequence_names = search_sequences_using_mmseqs2(
        input_monomer_fasta_filepath,
        reference_monomer_fasta_filepath,
        args.output_dir,
        molecule_type="nucleic_acid",
        max_seq_id=max_polymer_similarity,
        extra_parameters={
            # force nucleotide mode
            "--dbtype": 2,
            # force nucleotide search mode
            "--search-type": 3,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
            # 7 or 8 should work best, something to test
            "-k": 8,
            # there is currently an issue in mmseqs2 with nucleotide search and spaced k-mers
            "--spaced-kmer-mode": 0,
        },
    )
    input_monomer_peptide_sequence_names = search_sequences_using_mmseqs2(
        input_monomer_fasta_filepath,
        reference_monomer_fasta_filepath,
        args.output_dir,
        molecule_type="peptide",
        max_seq_id=max_polymer_similarity,
        # some of these parameters are from the spacepharer optimized parameters
        # these were for short CRISPR spacer recognition, so they should work well for arbitrary peptides
        extra_parameters={
            # force protein mode
            "--dbtype": 1,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
            # spacepharer optimized parameters
            "--gap-open": 16,
            "--gap-extend": 2,
            "--sub-mat": "VTML40.out",
            # we would like to try using ungapped prefilter mode to avoid
            # minimum consecutive k-mer match restrictions, but the cluster workflow doesn't expose this yet
            # let's use a real small k-mer size instead
            # "--prefilter-mode": 1,
            "-k": 5,
            "--spaced-kmer-mode": 0,
            # Don't try suppresing FP hits since the peptides are too short
            "--mask": 0,
            "--comp-bias-corr": 0,
            # let more things through the prefilter
            "--min-ungapped-score": 5,
            # Let's disable e-values as these are too short for reliable homology anyway
            # The most we can do is to collapse nearly identical peptides
            "-e": "inf",
        },
    )
    input_monomer_sequence_names = (
        input_monomer_protein_sequence_names
        | input_monomer_nucleic_acid_sequence_names
        | input_monomer_peptide_sequence_names
    )

    # Identify monomer sequences that passed the sequence identity criterion

    input_monomer_chain_sequences = filter_chains_by_sequence_names(
        input_monomer_chain_sequences, input_monomer_sequence_names
    )

    # Use MMseqs2 and RDKit to perform all-against-all sequence identity
    # and thresholded Tanimoto similarity comparisons for multimers

    input_multimer_protein_chain_mappings = search_sequences_using_mmseqs2(
        input_multimer_fasta_filepath,
        reference_multimer_fasta_filepath,
        args.output_dir,
        molecule_type="protein",
        max_seq_id=max_polymer_similarity,
        interface_chain_ids=input_interface_chain_ids,
        alignment_file_prefix="alnRes_multimer_",
        extra_parameters={
            # force protein mode
            "--dbtype": 1,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
        },
    )
    input_multimer_nucleic_acid_chain_mappings = search_sequences_using_mmseqs2(
        input_multimer_fasta_filepath,
        reference_multimer_fasta_filepath,
        args.output_dir,
        molecule_type="nucleic_acid",
        max_seq_id=max_polymer_similarity,
        interface_chain_ids=input_interface_chain_ids,
        alignment_file_prefix="alnRes_multimer_",
        extra_parameters={
            # force nucleotide mode
            "--dbtype": 2,
            # force nucleotide search mode
            "--search-type": 3,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
            # 7 or 8 should work best, something to test
            "-k": 8,
            # there is currently an issue in mmseqs2 with nucleotide search and spaced k-mers
            "--spaced-kmer-mode": 0,
        },
    )
    input_multimer_peptide_chain_mappings = search_sequences_using_mmseqs2(
        input_multimer_fasta_filepath,
        reference_multimer_fasta_filepath,
        args.output_dir,
        molecule_type="peptide",
        max_seq_id=max_polymer_similarity,
        interface_chain_ids=input_interface_chain_ids,
        alignment_file_prefix="alnRes_multimer_",
        # some of these parameters are from the spacepharer optimized parameters
        # these were for short CRISPR spacer recognition, so they should work well for arbitrary peptides
        extra_parameters={
            # force protein mode
            "--dbtype": 1,
            # force sensitivity level 8 per @milot-mirdita's suggestion
            "-s": 8,
            # spacepharer optimized parameters
            "--gap-open": 16,
            "--gap-extend": 2,
            "--sub-mat": "VTML40.out",
            # we would like to try using ungapped prefilter mode to avoid
            # minimum consecutive k-mer match restrictions, but the cluster workflow doesn't expose this yet
            # let's use a real small k-mer size instead
            # "--prefilter-mode": 1,
            "-k": 5,
            "--spaced-kmer-mode": 0,
            # Don't try suppresing FP hits since the peptides are too short
            "--mask": 0,
            "--comp-bias-corr": 0,
            # let more things through the prefilter
            "--min-ungapped-score": 5,
            # Let's disable e-values as these are too short for reliable homology anyway
            # The most we can do is to collapse nearly identical peptides
            "-e": "inf",
        },
    )

    input_multimer_chain_mappings_list = []
    if len(input_multimer_protein_chain_mappings):
        input_multimer_chain_mappings_list.append(input_multimer_protein_chain_mappings)
    if len(input_multimer_nucleic_acid_chain_mappings):
        input_multimer_chain_mappings_list.append(input_multimer_nucleic_acid_chain_mappings)
    if len(input_multimer_peptide_chain_mappings):
        input_multimer_chain_mappings_list.append(input_multimer_peptide_chain_mappings)
    input_multimer_chain_mappings = pl.concat(input_multimer_chain_mappings_list)

    # Identify multimer sequences and interfaces that passed the sequence identity and Tanimoto similarity criteria

    fpgen = AllChem.GetRDKitFPGenerator()
    reference_ligand_ccd_codes = filter_chains_by_molecule_type(
        reference_multimer_chain_sequences,
        molecule_type="ligand",
    )
    reference_ligand_fps = []
    for reference_ligand_ccd_code in reference_ligand_ccd_codes:
        reference_ligand_smiles = CCD_COMPONENTS_SMILES.get(reference_ligand_ccd_code, None)
        if not exists(reference_ligand_smiles):
            logger.warning(
                f"Could not find SMILES for reference CCD ligand: {reference_ligand_ccd_code}"
            )
            continue
        reference_ligand_mol = Chem.MolFromSmiles(reference_ligand_smiles)
        if not exists(reference_ligand_mol):
            logger.warning(
                f"Could not generate RDKit molecule for reference CCD ligand: {reference_ligand_ccd_code}"
            )
            continue
        reference_ligand_fp = fpgen.GetFingerprint(reference_ligand_mol)
        reference_ligand_fps.append(reference_ligand_fp)

    input_multimer_chain_sequences, input_interface_chain_ids = filter_chains_by_sequence_names(
        input_multimer_chain_sequences,
        input_multimer_chain_mappings.select(["query", "fident"]).to_numpy(),
        interface_chain_ids=input_interface_chain_ids,
        reference_ligand_fps=reference_ligand_fps,
        max_polymer_similarity=max_polymer_similarity,
        max_ligand_similarity=max_ligand_similarity,
        max_workers=max_workers,
    )

    # Assemble monomer and multimer chain sequences

    input_chain_sequences = input_monomer_chain_sequences + input_multimer_chain_sequences

    return input_chain_sequences, input_interface_chain_ids


@typecheck
def write_sequences_to_fasta(
    all_chain_sequences: CHAIN_SEQUENCES,
    fasta_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
    interface_chain_ids: CHAIN_INTERFACES | None = None,
) -> List[str]:
    """Write sequences of a particular molecule type to a FASTA file, and return all molecule IDs."""
    assert fasta_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    fasta_filepath = fasta_filepath.replace(".fasta", f"_{molecule_type}.fasta")

    molecule_ids = []
    with open(fasta_filepath, "w") as f:
        for structure_chain_sequences in tqdm(
            all_chain_sequences, desc=f"Writing {molecule_type} FASTA chain sequence file"
        ):
            for structure_id, chain_sequences in structure_chain_sequences.items():
                for chain_id, sequence in chain_sequences.items():
                    chain_id_, molecule_type_ = chain_id.split(":")
                    molecule_type_and_name = molecule_type_.split("-")
                    mol_type = (
                        molecule_type_and_name[0]
                        .replace("rna", "nucleic_acid")
                        .replace("dna", "nucleic_acid")
                    )
                    if mol_type == molecule_type:
                        molecule_index_postfix = (
                            f"-{molecule_type_and_name[1]}"
                            if len(molecule_type_and_name) == 2
                            else ""
                        )
                        molecule_id = f"{structure_id}{chain_id_}:{molecule_type_and_name[0]}{molecule_index_postfix}"

                        if exists(interface_chain_ids) and not any(
                            chain_id in interface_chain_id.split("+")
                            for interface_chain_id in interface_chain_ids[structure_id]
                        ):
                            continue

                        mapped_sequence = (
                            sequence.replace("X", "N")
                            if molecule_type == "nucleic_acid"
                            else sequence
                        )
                        f.write(f">{molecule_id}\n{mapped_sequence}\n")
                        molecule_ids.append(molecule_id)
    return molecule_ids


@timeout_decorator.timeout(IS_NOVEL_LIGAND_MAX_SECONDS_PER_INPUT, use_signals=False)
def is_novel_ligand(
    ligand_sequence: str,
    reference_ligand_fps: List[DataStructs.cDataStructs.ExplicitBitVect],
    max_sim: float = 0.3,
    verbose: bool = False,
) -> bool:
    """Check if a ligand sequence is novel based on Tanimoto similarity to a reference set of ligand sequences."""
    fpgen = AllChem.GetRDKitFPGenerator()
    ligand_smiles = CCD_COMPONENTS_SMILES.get(ligand_sequence, None)
    if not exists(ligand_smiles):
        if verbose:
            logger.warning(f"Could not find SMILES for ligand sequence: {ligand_sequence}")
        return True
    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    if not exists(ligand_mol):
        if verbose:
            logger.warning(
                f"Could not generate RDKit molecule for ligand sequence: {ligand_sequence}"
            )
        return True
    ligand_fp = fpgen.GetFingerprint(ligand_mol)

    for reference_ligand_fp in reference_ligand_fps:
        sim = DataStructs.TanimotoSimilarity(ligand_fp, reference_ligand_fp)
        if sim > max_sim:
            return False

    return True


def filter_structure_chain_sequences(
    structure_chain_sequences: Dict[str, Dict[str, str]],
    sequence_names: Set[str] | np.ndarray,
    interface_chain_ids: CHAIN_INTERFACES | None,
    reference_ligand_fps: List[DataStructs.cDataStructs.ExplicitBitVect] | None,
    max_polymer_similarity: float,
    max_ligand_similarity: float,
    filtered_structure_ids: Set[str],
):
    """Filter chain sequences based on either sequence names or Tanimoto similarity."""
    structure_id, chain_sequences = list(structure_chain_sequences.items())[0]
    interfaces_provided = exists(interface_chain_ids)

    if interfaces_provided:
        assert isinstance(
            sequence_names, np.ndarray
        ), "Sequence names must be provided as a NumPy array if interfaces are also provided."
        assert exists(
            reference_ligand_fps
        ), "Reference ligand fingerprints must be provided if interfaces are also provided."

    filtered_structure_chain_sequences = {}
    filtered_interface_chain_ids = defaultdict(set)

    for chain_id, sequence in chain_sequences.items():
        sequence_name = f"{structure_id}{chain_id}"
        molecule_type = chain_id.split(":")[1].split("-")[0]
        if interfaces_provided:
            matching_interfaces = [
                interface_chain_id
                for interface_chain_id in interface_chain_ids[structure_id]
                if chain_id in interface_chain_id.split("+")
            ]

            if any(matching_interfaces):
                interface_is_novel = True
                for interface in matching_interfaces:
                    ptnr1_chain_id, ptnr2_chain_id = interface.split("+")
                    ptnr1_sequence = chain_sequences.get(ptnr1_chain_id, None)
                    ptnr2_sequence = chain_sequences.get(ptnr2_chain_id, None)
                    ptnr1_molecule_type = ptnr1_chain_id.split(":")[1].split("-")[0]
                    ptnr2_molecule_type = ptnr2_chain_id.split(":")[1].split("-")[0]

                    if not (exists(ptnr1_sequence) and exists(ptnr2_sequence)):
                        continue

                    if ptnr1_molecule_type == "ligand":
                        # NOTE: We currently do not filter out interfaces
                        # involving a ligand with ranking model fit less
                        # than 0.5 or with multiple residues, due to a lack
                        # of available metadata within the context of this
                        # clustering script. This may be revisited in the future.
                        try:
                            ptnr1_is_novel = is_novel_ligand(
                                ptnr1_sequence,
                                reference_ligand_fps,
                                max_sim=max_ligand_similarity,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to check if partner 1 ligand is novel due to: {e}. Assuming it is not novel..."
                            )
                            ptnr1_is_novel = False
                    else:
                        matching_ptnr1_sequence_names = sequence_names[
                            sequence_names[:, 0] == f"{structure_id}{ptnr1_chain_id}"
                        ]
                        ptnr1_is_novel = not matching_ptnr1_sequence_names.size or (
                            matching_ptnr1_sequence_names[:, 1].max() <= max_polymer_similarity
                        )

                    if ptnr2_molecule_type == "ligand":
                        try:
                            ptnr2_is_novel = is_novel_ligand(
                                ptnr2_sequence,
                                reference_ligand_fps,
                                max_sim=max_ligand_similarity,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to check if partner 2 ligand is novel due to: {e}. Assuming it is not novel..."
                            )
                            ptnr2_is_novel = False
                    else:
                        matching_ptnr2_sequence_names = sequence_names[
                            sequence_names[:, 0] == f"{structure_id}{ptnr2_chain_id}"
                        ]
                        ptnr2_is_novel = not matching_ptnr2_sequence_names.size or (
                            matching_ptnr2_sequence_names[:, 1].max() <= max_polymer_similarity
                        )

                    if not (ptnr1_is_novel or ptnr2_is_novel):
                        # NOTE: An interface is only considered novel if at least one of its chains is novel
                        interface_is_novel = False
                        break

                if interface_is_novel:
                    # NOTE: Only if all of a chain's associated interfaces are novel will the chain be kept
                    filtered_structure_chain_sequences[chain_id] = sequence
            else:
                # NOTE: Chains within a multimeric structure that are not interfacing with any other chain are kept as is
                filtered_structure_chain_sequences[chain_id] = sequence

        elif not interfaces_provided and (
            sequence_name in sequence_names
            or (structure_id in filtered_structure_ids and molecule_type == "ligand")
        ):
            filtered_structure_chain_sequences[chain_id] = sequence

    if filtered_structure_chain_sequences and interfaces_provided:
        for interface_chain_id in interface_chain_ids[structure_id]:
            chain_id_1, chain_id_2 = interface_chain_id.split("+")
            if (
                chain_id_1 in filtered_structure_chain_sequences
                and chain_id_2 in filtered_structure_chain_sequences
                and f"{chain_id_2}:{chain_id_1}" not in filtered_interface_chain_ids[structure_id]
            ):
                filtered_interface_chain_ids[structure_id].add(interface_chain_id)

    return structure_id, filtered_structure_chain_sequences, filtered_interface_chain_ids


@typecheck
def filter_chains_by_sequence_names(
    all_chain_sequences: CHAIN_SEQUENCES,
    sequence_names: Set[str] | np.ndarray,
    interface_chain_ids: CHAIN_INTERFACES | None = None,
    reference_ligand_fps: List[DataStructs.cDataStructs.ExplicitBitVect] | None = None,
    max_polymer_similarity: float = 0.3,
    max_ligand_similarity: float = 0.3,
    max_workers: int = 2,
) -> CHAIN_SEQUENCES | Tuple[CHAIN_SEQUENCES, CHAIN_INTERFACES]:
    """Return only chains (and potentially interfaces) with sequence names in the given set."""
    filtered_structure_ids = set(
        name.split("-assembly1")[0] + "-assembly1"
        for name in (
            sequence_names[:, 0].tolist()
            if isinstance(sequence_names, np.ndarray)
            else sequence_names
        )
    )
    interfaces_provided = exists(interface_chain_ids)

    if interfaces_provided:
        assert isinstance(
            sequence_names, np.ndarray
        ), "Sequence names must be provided as a NumPy array if interfaces are also provided."
        assert exists(
            reference_ligand_fps
        ), "Reference ligand fingerprints must be provided if interfaces are also provided."

    filtered_chain_sequences = []
    filtered_interface_chain_ids = defaultdict(set)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_structure = {
            executor.submit(
                filter_structure_chain_sequences,
                structure_chain_sequences=structure_chain_sequences,
                sequence_names=sequence_names,
                interface_chain_ids=interface_chain_ids,
                reference_ligand_fps=reference_ligand_fps,
                max_polymer_similarity=max_polymer_similarity,
                max_ligand_similarity=max_ligand_similarity,
                filtered_structure_ids=filtered_structure_ids,
            ): structure_chain_sequences
            for structure_chain_sequences in all_chain_sequences
        }

        for future in tqdm(
            as_completed(future_to_structure),
            total=len(future_to_structure),
            desc="Filtering chain sequences by sequence names",
        ):
            (
                structure_id,
                filtered_structure_chain_sequences,
                filtered_structure_interface_ids,
            ) = future.result()
            if filtered_structure_chain_sequences:
                filtered_chain_sequences.append({structure_id: filtered_structure_chain_sequences})
                if interfaces_provided:
                    filtered_interface_chain_ids[structure_id] = filtered_structure_interface_ids[
                        structure_id
                    ]

    if interfaces_provided:
        filtered_chain_sequences = [
            sequences
            for sequences in filtered_chain_sequences
            if list(sequences.keys())[0] in filtered_interface_chain_ids
        ]
        filtered_interface_chain_ids = {
            k: list(v) for k, v in filtered_interface_chain_ids.items()
        }
        return filtered_chain_sequences, filtered_interface_chain_ids
    return filtered_chain_sequences


@typecheck
def filter_chains_by_molecule_type(
    all_chain_sequences: CHAIN_SEQUENCES,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
    interface_chain_ids: CHAIN_INTERFACES | None = None,
) -> List[str]:
    """Return only chains of a particular molecule type."""
    filtered_chain_sequences = set()
    for structure_chain_sequences in tqdm(
        all_chain_sequences, desc=f"Filtering for {molecule_type} chains"
    ):
        for structure_id, chain_sequences in structure_chain_sequences.items():
            for chain_id, sequence in chain_sequences.items():
                _, molecule_type_ = chain_id.split(":")
                molecule_type_and_name = molecule_type_.split("-")
                mol_type = (
                    molecule_type_and_name[0]
                    .replace("rna", "nucleic_acid")
                    .replace("dna", "nucleic_acid")
                )
                if mol_type == molecule_type:
                    if (
                        exists(interface_chain_ids)
                        and any(
                            chain_id in interface_chain_id.split("+")
                            for interface_chain_id in interface_chain_ids[structure_id]
                        )
                    ) or not exists(interface_chain_ids):
                        filtered_chain_sequences.add(sequence)
    return list(filtered_chain_sequences)


@typecheck
def extract_pdb_chain_and_molecule_ids_from_clustering_string(x: str) -> Tuple[str, str, str]:
    """Extract PDB, chain, and molecule IDs from a clustering output string."""
    pdb_id = (
        x.split(":")[0].split("-assembly1")[0] + "-assembly1"
        if "-assembly1" in x
        else x.split(":")[0][:4]
    )
    chain_id = x.split(":")[0].split("-assembly1")[1] if "assembly1" in x else x.split(":")[0][4:]
    molecule_id = x.split(":")[1]
    return pdb_id, chain_id, molecule_id


@typecheck
def search_sequences_using_mmseqs2(
    input_filepath: str,
    reference_filepath: str,
    output_dir: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
    max_seq_id: float = 0.3,
    interface_chain_ids: CHAIN_INTERFACES | None = None,
    alignment_file_prefix: str = "alnRes_",
    extra_parameters: Dict[str, int | float | str] | None = None,
) -> Set[str] | pl.DataFrame:
    """Run MMseqs2 on the input FASTA file and write the resulting search outputs to a local output directory."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert reference_filepath.endswith(".fasta"), "The reference file must be a FASTA file."

    input_filepath = input_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    reference_filepath = reference_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    output_alignment_filepath = os.path.join(
        output_dir, molecule_type, f"{alignment_file_prefix}{molecule_type}.m8"
    )
    tmp_output_dir = os.path.join(output_dir, molecule_type, "tmp")
    os.makedirs(os.path.join(output_dir, molecule_type), exist_ok=True)

    assert os.path.isfile(input_filepath), f"Input file '{input_filepath}' does not exist."
    assert os.path.isfile(
        reference_filepath
    ), f"Reference file '{reference_filepath}' does not exist."

    # Search sequences

    mmseqs_command = [
        "mmseqs",
        "easy-search",
        input_filepath,
        reference_filepath,
        output_alignment_filepath,
        tmp_output_dir,
    ]
    if extra_parameters:
        for key, value in extra_parameters.items():
            mmseqs_command.extend([key, str(value)])

    subprocess.run(mmseqs_command)
    if not os.path.isfile(output_alignment_filepath):
        logger.warning(
            f"Output alignment file '{output_alignment_filepath}' does not exist. No input sequences were found."
        )
        return set()
    try:
        chain_search_mapping = pl.read_csv(
            output_alignment_filepath,
            separator="\t",
            has_header=False,
            new_columns=[
                "query",
                "target",
                "fident",
                "alnlen",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "evalue",
                "bits",
            ],
        )
    except Exception as e:
        logger.warning(
            f"Failed to read MMseqs2 alignment file '{output_alignment_filepath}' due to: {e}"
        )
        return set()

    # For monomers, filter out sequences with reference sequence identity greater than the maximum threshold;
    # For multimers, return the chain search results for all input-reference combinations

    if exists(interface_chain_ids):
        return chain_search_mapping
    else:
        input_queries = set()
        with open(input_filepath, "r") as f:
            for line in f:
                if line.startswith(">"):
                    input_queries.add(line.strip().lstrip(">"))

        # Re-insert the names of input queries for which MMseqs2 could not find a match,
        # as these are safely not homologous to any reference sequence (due to MMseqs2's
        # default sensitivity settings)

        mappable_queries = set(chain_search_mapping.get_column("query").to_list())
        unmappable_queries = input_queries - mappable_queries

        return (
            set(
                chain_search_mapping.group_by("query")
                .agg(pl.max("fident"))
                .filter(pl.col("fident") <= max_seq_id)
                .get_column("query")
                .to_list()
            )
            | unmappable_queries
        )


@typecheck
def cluster_sequences_using_mmseqs2(
    input_filepath: str,
    output_dir: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
    min_seq_id: float = 0.5,
    coverage: float = 0.8,
    coverage_mode: Literal[0, 1, 2, 3] = 1,
    extra_parameters: Dict[str, int | float | str] | None = None,
) -> Dict[str, int]:
    """Run MMseqs2 on the input FASTA file and write the resulting clusters to a local output directory."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."

    input_filepath = input_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    output_db_filepath = os.path.join(output_dir, molecule_type, f"DB_{molecule_type}")
    tmp_output_dir = os.path.join(output_dir, molecule_type, "tmp")
    output_cluster_filepath = os.path.join(
        args.output_dir, molecule_type, f"DB_{molecule_type}_cluster.tsv"
    )
    os.makedirs(os.path.join(output_dir, molecule_type), exist_ok=True)

    assert os.path.isfile(input_filepath), f"Input file '{input_filepath}' does not exist."

    # Cluster sequences

    mmseqs_command = [
        "mmseqs",
        "easy-cluster",
        input_filepath,
        output_db_filepath,
        tmp_output_dir,
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(coverage_mode),
    ]
    if extra_parameters:
        for key, value in extra_parameters.items():
            mmseqs_command.extend([key, str(value)])

    subprocess.run(mmseqs_command)
    if not os.path.isfile(output_cluster_filepath):
        logger.warning(
            f"Output cluster file '{output_cluster_filepath}' does not exist. No input sequences were clustered."
        )
        return {}

    chain_cluster_mapping = pl.read_csv(
        output_cluster_filepath,
        separator="\t",
        has_header=False,
        new_columns=["cluster_rep", "cluster_member"],
    )
    chain_cluster_mapping.insert_column(
        len(chain_cluster_mapping.columns),
        chain_cluster_mapping.get_column("cluster_rep")
        .cast(pl.Categorical)
        .to_physical()
        .rename("cluster_id"),
    )
    chain_cluster_mappings = dict(
        zip(
            chain_cluster_mapping.get_column("cluster_member"),
            chain_cluster_mapping.get_column("cluster_id"),
        )
    )

    # Cache chain cluster mappings to local (CSV) storage

    local_chain_cluster_mapping = pl.DataFrame(
        chain_cluster_mapping.get_column("cluster_member")
        .map_elements(
            extract_pdb_chain_and_molecule_ids_from_clustering_string, return_dtype=pl.List
        )
        .to_list(),
        schema=["pdb_id", "chain_id", "molecule_id"],
        orient="row",
    )
    local_chain_cluster_mapping.insert_column(
        len(local_chain_cluster_mapping.columns),
        chain_cluster_mapping.get_column("cluster_id"),
    )
    local_chain_cluster_mapping.write_csv(
        os.path.join(output_dir, f"{molecule_type}_chain_cluster_mapping.csv")
    )

    return chain_cluster_mappings


@typecheck
def cluster_ligands_by_ccd_code(
    all_chain_sequences: CHAIN_SEQUENCES, output_dir: str
) -> Dict[str, int]:
    """Cluster ligands based on their CCD codes and write the resulting clusters to a local output directory."""
    # Parse the ligand sequences from all chain sequences, while clustering them based on their CCD codes
    chain_cluster_mapping = {}
    ccd_code_to_cluster_mapping = {}
    for structure_chain_sequences in tqdm(
        all_chain_sequences, desc="Clustering ligands by CCD code"
    ):
        for structure_id, chain_sequences in structure_chain_sequences.items():
            for chain_id, sequence in chain_sequences.items():
                chain_id_, molecule_type_ = chain_id.split(":")
                molecule_type_and_name = molecule_type_.split("-")
                if molecule_type_and_name[0] == "ligand":
                    molecule_index_postfix = (
                        f"-{molecule_type_and_name[1]}" if len(molecule_type_and_name) == 2 else ""
                    )
                    if sequence in ccd_code_to_cluster_mapping:
                        cluster_id = ccd_code_to_cluster_mapping[sequence]
                    else:
                        cluster_id = len(ccd_code_to_cluster_mapping)
                        ccd_code_to_cluster_mapping[sequence] = cluster_id
                    chain_cluster_mapping[
                        f"{structure_id}{chain_id_}:{molecule_type_and_name[0]}{molecule_index_postfix}"
                    ] = cluster_id

    # Cache chain cluster mappings to local (CSV) storage
    local_chain_cluster_mapping = pl.DataFrame(
        [
            (*extract_pdb_chain_and_molecule_ids_from_clustering_string(k), v)
            for (k, v) in chain_cluster_mapping.items()
        ],
        schema=["pdb_id", "chain_id", "molecule_id", "cluster_id"],
        orient="row",
    )
    local_chain_cluster_mapping.write_csv(
        os.path.join(output_dir, "ligand_chain_cluster_mapping.csv"),
    )

    return chain_cluster_mapping


@typecheck
def map_pdb_chain_id_to_chain_cluster_id(
    pdb_chain_id: str,
    molecule_id: str,
    protein_chain_cluster_mapping: Dict[str, IntType],
    nucleic_acid_chain_cluster_mapping: Dict[str, IntType],
    peptide_chain_cluster_mapping: Dict[str, IntType],
    ligand_chain_cluster_mapping: Dict[str, IntType],
) -> str:
    """Map a PDB chain ID and molecule ID to a chain cluster ID based on the chain's (majority) molecule type."""
    if "protein" in pdb_chain_id and pdb_chain_id in protein_chain_cluster_mapping:
        chain_cluster = f"{molecule_id}-cluster-{protein_chain_cluster_mapping[pdb_chain_id]}"
    elif (
        "protein" in pdb_chain_id
        and pdb_chain_id.replace("protein", "peptide") in peptide_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-protein (or protein-X) interaction is actually a peptide interaction, e.g., PDB `148l`
        chain_cluster = f"{molecule_id}-cluster-{peptide_chain_cluster_mapping[pdb_chain_id.replace('protein', 'peptide')]}"
    elif (
        "protein" in pdb_chain_id
        and pdb_chain_id.replace("protein", "rna") in nucleic_acid_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-protein (or protein-X) interaction is actually a nucleic acid interaction, e.g., PDB `1b23`
        chain_cluster = f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id.replace('protein', 'rna')]}"
    elif (
        "protein" in pdb_chain_id
        and pdb_chain_id.replace("protein", "dna") in nucleic_acid_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-protein (or protein-X) interaction is actually a nucleic acid interaction, e.g., PDB `1b23`
        chain_cluster = f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id.replace('protein', 'dna')]}"
    elif (
        "rna" in pdb_chain_id or "dna" in pdb_chain_id
    ) and pdb_chain_id in nucleic_acid_chain_cluster_mapping:
        chain_cluster = f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id]}"
    elif (
        "rna" in pdb_chain_id
        and pdb_chain_id.replace("rna", "dna") in nucleic_acid_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-RNA (or RNA-X) interaction is actually a DNA interaction, e.g., PDB `216d`
        chain_cluster = f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id.replace('rna', 'dna')]}"
    elif (
        "dna" in pdb_chain_id
        and pdb_chain_id.replace("dna", "rna") in nucleic_acid_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-DNA (or DNA-X) interaction is actually an RNA interaction, e.g., PDB `216d`
        chain_cluster = f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id.replace('dna', 'rna')]}"
    elif (
        "rna" in pdb_chain_id
        and pdb_chain_id.replace("rna", "protein") in protein_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-nucleic acid (or nucleic acid-X) interaction is actually a protein interaction, e.g., PDB `3a1s`
        chain_cluster = f"{molecule_id}-cluster-{protein_chain_cluster_mapping[pdb_chain_id.replace('rna', 'protein')]}"
    elif (
        "dna" in pdb_chain_id
        and pdb_chain_id.replace("dna", "protein") in protein_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-nucleic acid (or nucleic acid-X) interaction is actually a protein interaction, e.g., PDB `3a1s`
        chain_cluster = f"{molecule_id}-cluster-{protein_chain_cluster_mapping[pdb_chain_id.replace('dna', 'protein')]}"
    elif (
        "rna" in pdb_chain_id
        and pdb_chain_id.replace("rna", "peptide") in peptide_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-nucleic acid (or nucleic acid-X) interaction is actually a peptide interaction, e.g., PDB `2aiz`
        chain_cluster = f"{molecule_id}-cluster-{peptide_chain_cluster_mapping[pdb_chain_id.replace('rna', 'peptide')]}"
    elif (
        "dna" in pdb_chain_id
        and pdb_chain_id.replace("dna", "peptide") in peptide_chain_cluster_mapping
    ):
        # Based on (majority) chain molecule types, handle instances where
        # a X-nucleic acid (or nucleic acid-X) interaction is actually a peptide interaction, e.g., PDB `2aiz`
        chain_cluster = f"{molecule_id}-cluster-{peptide_chain_cluster_mapping[pdb_chain_id.replace('dna', 'peptide')]}"
    elif "peptide" in pdb_chain_id and pdb_chain_id in peptide_chain_cluster_mapping:
        chain_cluster = f"{molecule_id}-cluster-{peptide_chain_cluster_mapping[pdb_chain_id]}"
    elif "ligand" in pdb_chain_id and pdb_chain_id in ligand_chain_cluster_mapping:
        chain_cluster = f"{molecule_id}-cluster-{ligand_chain_cluster_mapping[pdb_chain_id]}"
    else:
        raise ValueError(f"Chain {pdb_chain_id} not found in any chain cluster mapping.")

    return chain_cluster


@typecheck
def cluster_interfaces(
    protein_chain_cluster_mapping: Dict[str, IntType],
    nucleic_acid_chain_cluster_mapping: Dict[str, IntType],
    peptide_chain_cluster_mapping: Dict[str, IntType],
    ligand_chain_cluster_mapping: Dict[str, IntType],
    interface_chain_ids: CHAIN_INTERFACES,
    output_dir: str,
) -> INTERFACE_CLUSTERS:
    """Cluster interfaces based on the cluster IDs of the chains involved."""
    interface_chains_cluster_mapping = {}
    interface_clusters = {}

    for pdb_id in tqdm(interface_chain_ids, desc="Clustering interfaces"):
        for chain_id_pair in interface_chain_ids[pdb_id]:
            chain_ids = chain_id_pair.split("+")
            chain_clusters = []
            for chain_id in chain_ids:
                pdb_chain_id = f"{pdb_id}{chain_id}"
                molecule_id = chain_id.split(":")[-1]
                chain_clusters.append(
                    map_pdb_chain_id_to_chain_cluster_id(
                        pdb_chain_id,
                        molecule_id,
                        protein_chain_cluster_mapping,
                        nucleic_acid_chain_cluster_mapping,
                        peptide_chain_cluster_mapping,
                        ligand_chain_cluster_mapping,
                    )
                )
            # Ensure that each interface cluster is unique
            if (
                len(chain_clusters) == 2
                and (chain_clusters[0], chain_clusters[1]) not in interface_chains_cluster_mapping
                and (chain_clusters[1], chain_clusters[0]) not in interface_chains_cluster_mapping
            ):
                # Assign a unique interface cluster ID as a join on the constituent chain cluster IDs,
                # such that two interfaces I and J are in the same interface cluster C^interface only if
                # their constituent chain pairs {I_1,I_2},{J_1,J_2} have the same chain cluster pairs {C_1^chain ,C_2^chain}.
                interface_chains_cluster_mapping[(chain_clusters[0], chain_clusters[1])] = len(
                    interface_chains_cluster_mapping
                )
            elif len(chain_clusters) != 2:
                raise ValueError(
                    f"Invalid number of chains in interface {chain_id_pair} for PDB ID {pdb_id}."
                )
            chain_cluster_0 = chain_clusters[0].split("-")[-1]
            chain_cluster_1 = chain_clusters[1].split("-")[-1]
            interface_cluster_mapping = (
                interface_chains_cluster_mapping[(chain_clusters[1], chain_clusters[0])]
                if (chain_clusters[1], chain_clusters[0]) in interface_chains_cluster_mapping
                else interface_chains_cluster_mapping[(chain_clusters[0], chain_clusters[1])]
            )
            interface_clusters[
                f"{pdb_id}~{chain_id_pair}"
            ] = f"{chain_cluster_0},{chain_cluster_1}:{interface_cluster_mapping}"

    # Cache interface cluster mappings to local (CSV) storage
    pl.DataFrame(
        (
            (
                k.split("~")[0],
                k.split("+")[0].split("~")[-1].split(":")[0],
                k.split("+")[1].split("~")[-1].split(":")[0],
                k.split("+")[0].split("~")[-1].split(":")[1],
                k.split("+")[1].split("~")[-1].split(":")[1],
                int(v.split(":")[0].split(",")[0]),
                int(v.split(":")[0].split(",")[1]),
                int(v.split(":")[1]),
            )
            for k, v in interface_clusters.items()
        ),
        schema=[
            "pdb_id",
            "interface_chain_id_1",
            "interface_chain_id_2",
            "interface_molecule_id_1",
            "interface_molecule_id_2",
            "interface_chain_cluster_id_1",
            "interface_chain_cluster_id_2",
            "interface_cluster_id",
        ],
    ).write_csv(os.path.join(output_dir, "interface_cluster_mapping.csv"))

    return interface_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster chains and interfaces within the AlphaFold 3 PDB validation dataset's filtered mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "test_mmcifs"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "--reference_1_clustering_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "train_clusterings"),
        help="Path to the first reference clustering directory.",
    )
    parser.add_argument(
        "--reference_2_clustering_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "val_clusterings"),
        help="Path to the second reference clustering directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "test_clusterings"),
        help="Path to the output clustering directory.",
    )
    parser.add_argument(
        "--clustering_filtered_pdb_dataset",
        action="store_true",
        help="Whether the clustering is being performed on a filtered PDB dataset.",
    )
    parser.add_argument(
        "-n",
        "--no_workers",
        type=int,
        default=16,
        help="Number of workers to use for clustering.",
    )
    args = parser.parse_args()

    # Validate input arguments
    assert os.path.isdir(args.mmcif_dir), f"mmCIF directory '{args.mmcif_dir}' does not exist."
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine paths for intermediate files

    fasta_filepath = os.path.join(args.output_dir, "sequences.fasta")
    reference_1_fasta_filepath = os.path.join(args.reference_1_clustering_dir, "sequences.fasta")
    reference_2_fasta_filepath = os.path.join(args.reference_2_clustering_dir, "sequences.fasta")

    # Attempt to load existing chain sequences and interfaces from local storage

    if os.path.exists(
        os.path.join(args.output_dir, "all_chain_sequences.json")
    ) and os.path.exists(os.path.join(args.output_dir, "interface_chain_ids.json")):
        with open(os.path.join(args.output_dir, "all_chain_sequences.json"), "r") as f:
            all_chain_sequences = json.load(f)

        with open(os.path.join(args.output_dir, "interface_chain_ids.json"), "r") as f:
            interface_chain_ids = json.load(f)
    else:
        # Parse all chain sequences and interfaces from mmCIF files

        (
            all_chain_sequences,
            interface_chain_ids,
        ) = parse_chain_sequences_and_interfaces_from_mmcif_directory(
            args.mmcif_dir,
            max_workers=args.no_workers,
            assume_one_based_residue_ids=args.clustering_filtered_pdb_dataset,
        )

        # Cache chain sequences and interfaces to local storage

        with open(os.path.join(args.output_dir, "all_chain_sequences.json"), "w") as f:
            json.dump(all_chain_sequences, f)

        with open(os.path.join(args.output_dir, "interface_chain_ids.json"), "w") as f:
            json.dump(interface_chain_ids, f)

    # Attempt to filter chain sequences and interfaces according to the AlphaFold 3 supplement

    if os.path.exists(
        os.path.join(args.output_dir, "filtered_all_chain_sequences.json")
    ) and os.path.exists(os.path.join(args.output_dir, "filtered_interface_chain_ids.json")):
        with open(os.path.join(args.output_dir, "filtered_all_chain_sequences.json"), "r") as f:
            all_chain_sequences = json.load(f)

        with open(os.path.join(args.output_dir, "filtered_interface_chain_ids.json"), "r") as f:
            interface_chain_ids = json.load(f)
    else:
        with open(
            os.path.join(args.reference_1_clustering_dir, "all_chain_sequences.json"), "r"
        ) as f:
            reference_1_all_chain_sequences = json.load(f)
        with open(
            os.path.join(args.reference_2_clustering_dir, "filtered_all_chain_sequences.json"), "r"
        ) as f:
            reference_2_all_chain_sequences = json.load(f)

        (
            all_chain_sequences,
            interface_chain_ids,
        ) = filter_to_low_homology_sequences(
            all_chain_sequences,
            reference_1_all_chain_sequences,
            interface_chain_ids,
            fasta_filepath,
            reference_1_fasta_filepath,
            max_workers=args.no_workers,
        )
        (
            all_chain_sequences,
            interface_chain_ids,
        ) = filter_to_low_homology_sequences(
            all_chain_sequences,
            reference_2_all_chain_sequences,
            interface_chain_ids,
            fasta_filepath,
            reference_2_fasta_filepath,
            max_workers=args.no_workers,
        )

        # Cache (filtered) chain sequences and interfaces to local storage

        with open(os.path.join(args.output_dir, "filtered_all_chain_sequences.json"), "w") as f:
            json.dump(all_chain_sequences, f)

        with open(os.path.join(args.output_dir, "filtered_interface_chain_ids.json"), "w") as f:
            json.dump(interface_chain_ids, f)

    # Attempt to load existing chain cluster mappings from local storage

    protein_chain_cluster_mapping = {}
    nucleic_acid_chain_cluster_mapping = {}
    peptide_chain_cluster_mapping = {}
    ligand_chain_cluster_mapping = {}

    if os.path.exists(os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")):
        protein_chain_cluster_mapping = pl.read_csv(
            os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")
        ).with_columns(
            pl.format("{}{}:{}", "pdb_id", "chain_id", "molecule_id").alias("combined_key")
        )
        protein_chain_cluster_mapping = dict(
            zip(
                protein_chain_cluster_mapping.get_column("combined_key"),
                protein_chain_cluster_mapping.get_column("cluster_id"),
            )
        )
    if os.path.exists(os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv")):
        nucleic_acid_chain_cluster_mapping = pl.read_csv(
            os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv")
        ).with_columns(
            pl.format("{}{}:{}", "pdb_id", "chain_id", "molecule_id").alias("combined_key")
        )
        nucleic_acid_chain_cluster_mapping = dict(
            zip(
                nucleic_acid_chain_cluster_mapping.get_column("combined_key"),
                nucleic_acid_chain_cluster_mapping.get_column("cluster_id"),
            )
        )
    if os.path.exists(os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv")):
        peptide_chain_cluster_mapping = pl.read_csv(
            os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv")
        ).with_columns(
            pl.format("{}{}:{}", "pdb_id", "chain_id", "molecule_id").alias("combined_key")
        )
        peptide_chain_cluster_mapping = dict(
            zip(
                peptide_chain_cluster_mapping.get_column("combined_key"),
                peptide_chain_cluster_mapping.get_column("cluster_id"),
            )
        )
    if os.path.exists(os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv")):
        ligand_chain_cluster_mapping = pl.read_csv(
            os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv")
        ).with_columns(
            pl.format("{}{}:{}", "pdb_id", "chain_id", "molecule_id").alias("combined_key")
        )
        ligand_chain_cluster_mapping = dict(
            zip(
                ligand_chain_cluster_mapping.get_column("combined_key"),
                ligand_chain_cluster_mapping.get_column("cluster_id"),
            )
        )

    # Cluster sequences separately for each molecule type

    if not protein_chain_cluster_mapping:
        protein_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="protein"
        )
        protein_chain_cluster_mapping = cluster_sequences_using_mmseqs2(
            # Cluster proteins at 40% sequence homology
            fasta_filepath,
            args.output_dir,
            molecule_type="protein",
            min_seq_id=0.4,
            coverage=0.8,
            coverage_mode=0,
            extra_parameters={
                # force protein mode
                "--dbtype": 1,
                # cluster reassign improves clusters by reassigning sequences to the best cluster
                # and fixes transitivity issues of the cascade clustering
                "--cluster-reassign": 1,
            },
        )

    if not nucleic_acid_chain_cluster_mapping:
        nucleic_acid_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="nucleic_acid"
        )
        nucleic_acid_chain_cluster_mapping = cluster_sequences_using_mmseqs2(
            # Cluster nucleic acids at 100% sequence homology
            fasta_filepath,
            args.output_dir,
            molecule_type="nucleic_acid",
            min_seq_id=1.0,
            coverage=0.8,
            coverage_mode=0,
            extra_parameters={
                # force nucleotide mode
                "--dbtype": 2,
                # 7 or 8 should work best, something to test
                "-k": 8,
                # there is currently an issue in mmseqs2 with nucleotide search and spaced k-mers
                "--spaced-kmer-mode": 0,
                # see above
                "--cluster-reassign": 1,
            },
        )

    if not peptide_chain_cluster_mapping:
        peptide_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="peptide"
        )
        peptide_chain_cluster_mapping = cluster_sequences_using_mmseqs2(
            # Cluster peptides at 100% sequence homology
            fasta_filepath,
            args.output_dir,
            molecule_type="peptide",
            min_seq_id=1.0,
            coverage=0.8,
            coverage_mode=0,
            # some of these parameters are from the spacepharer optimized parameters
            # these were for short CRISPR spacer recognition, so they should work well for arbitrary peptides
            extra_parameters={
                # force protein mode
                "--dbtype": 1,
                # spacepharer optimized parameters
                "--gap-open": 16,
                "--gap-extend": 2,
                "--sub-mat": "VTML40.out",
                # we would like to try using ungapped prefilter mode to avoid
                # minimum consecutive k-mer match restrictions, but the cluster workflow doesn't expose this yet
                # let's use a real small k-mer size instead
                # "--prefilter-mode": 1,
                "-k": 5,
                "--spaced-kmer-mode": 0,
                # Don't try suppresing FP hits since the peptides are too short
                "--mask": 0,
                "--comp-bias-corr": 0,
                # let more things through the prefilter
                "--min-ungapped-score": 5,
                # Let's disable e-values as these are too short for reliable homology anyway
                # The most we can do is to collapse nearly identical peptides
                "-e": "inf",
                # see above
                "--cluster-reassign": 1,
            },
        )

    if not ligand_chain_cluster_mapping:
        ligand_chain_cluster_mapping = cluster_ligands_by_ccd_code(
            # Cluster ligands based on their CCD codes (i.e., identical ligands share a cluster)
            all_chain_sequences,
            args.output_dir,
        )

    # Cluster interfaces based on the cluster IDs of the chains involved, and save the interface cluster mapping to local (CSV) storage

    assert all(
        (
            protein_chain_cluster_mapping,
            ligand_chain_cluster_mapping,
        )
    ), "At least protein and ligand molecule type-specific chain cluster mappings must be available to cluster interfaces."

    cluster_interfaces(
        protein_chain_cluster_mapping,
        nucleic_acid_chain_cluster_mapping,
        peptide_chain_cluster_mapping,
        ligand_chain_cluster_mapping,
        interface_chain_ids,
        args.output_dir,
    )

    # Ensure each cluster mapping has a corresponding CSV file

    if not protein_chain_cluster_mapping:
        pl.DataFrame([], schema=["pdb_id", "chain_id", "molecule_id", "cluster_id"]).write_csv(
            os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")
        )
    if not nucleic_acid_chain_cluster_mapping:
        pl.DataFrame([], schema=["pdb_id", "chain_id", "molecule_id", "cluster_id"]).write_csv(
            os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv")
        )
    if not peptide_chain_cluster_mapping:
        pl.DataFrame([], schema=["pdb_id", "chain_id", "molecule_id", "cluster_id"]).write_csv(
            os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv")
        )
    if not ligand_chain_cluster_mapping:
        pl.DataFrame([], schema=["pdb_id", "chain_id", "molecule_id", "cluster_id"]).write_csv(
            os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv")
        )

# %%
