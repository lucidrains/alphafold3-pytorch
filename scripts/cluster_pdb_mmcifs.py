# %% [markdown]
# # Clustering AlphaFold 3 PDB Dataset
#
# For clustering AlphaFold 3's PDB dataset, we follow the clustering procedure outlined in Abramson et al (2024).
#
# In order to reduce bias in the training and evaluation sets, clustering was performed on PDB chains and interfaces, as
# follows.
# • Chain-based clustering occur at 40% sequence homology for proteins, 100% homology for nucleic acids, 100%
# homology for peptides (<10 residues) and according to CCD identity for small molecules (i.e. only identical
# molecules share a cluster).
# • Chain-based clustering of polymers with modified residues is first done by mapping the modified residues to
# a standard residue using SCOP [23, 24] convention (https://github.com/biopython/biopython/
# blob/5ee5e69e649dbe17baefe3919e56e60b54f8e08f/Bio/Data/SCOPData.py). If the mod-
# ified residue could not be found as a mapping key or was mapped to a value longer than a single character, it was
# mapped to type unknown.
# • Interface-based clustering is a join on the cluster IDs of the constituent chains, such that interfaces I and J are
# in the same interface cluster C^interface only if their constituent chain pairs {I_1,I_2},{J_1,J_2} have the same chain
# cluster pairs {C_1^chain ,C_2^chain}.

# %%

import argparse
import glob
import os
import subprocess
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.Data import PDBData
from Bio.PDB.NeighborSearch import NeighborSearch
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from alphafold3_pytorch.tensor_typing import IntType, typecheck
from alphafold3_pytorch.utils.utils import exists
from scripts.filter_pdb_mmcifs import parse_mmcif_object

# Constants

CHAIN_SEQUENCES = List[Dict[str, Dict[str, str]]]
CHAIN_INTERFACES = Dict[str, List[str]]
INTERFACE_CLUSTERS = Dict[str, str]
RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand"]
CLUSTERING_MOLECULE_TYPE = Literal["protein", "nucleic_acid", "peptide", "ligand", "unknown"]

PROTEIN_LETTERS_3TO1 = {k.strip(): v.strip() for k, v in PDBData.protein_letters_3to1.items()}
NUCLEIC_LETTERS_3TO1 = {k.strip(): v.strip() for k, v in PDBData.nucleic_letters_3to1.items()}

PROTEIN_LETTERS_3TO1_EXTENDED = {
    k.strip(): v.strip() for k, v in PDBData.protein_letters_3to1_extended.items()
}
NUCLEIC_LETTERS_3TO1_EXTENDED = {
    k.strip(): v.strip() for k, v in PDBData.nucleic_letters_3to1_extended.items()
}

PROTEIN_LETTERS_1TO3 = {k.strip(): v.strip() for k, v in PDBData.protein_letters_1to3.items()}
RNA_LETTERS_1TO3 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}
DNA_LETTERS_1TO3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
}


# Helper functions


@typecheck
def get_residue_molecule_type(res_chem_type: str) -> RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue."""
    if "peptide" in res_chem_type.lower():
        return "protein"
    elif "rna" in res_chem_type.lower():
        return "rna"
    elif "dna" in res_chem_type.lower():
        return "dna"
    else:
        return "ligand"


@typecheck
def convert_modified_residue_three_to_one(
    residue_id: str, residue_mol_type: RESIDUE_MOLECULE_TYPE
) -> Tuple[str, CLUSTERING_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).
    Also return the clustering-specific molecule type of the residue.

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
            PROTEIN_LETTERS_3TO1[mapped_residue]
            if mapped_residue in PROTEIN_LETTERS_3TO1
            else "X",
            "protein",
        )
    elif residue_mol_type in {"rna", "dna"}:
        return (
            NUCLEIC_LETTERS_3TO1[mapped_residue]
            if mapped_residue in NUCLEIC_LETTERS_3TO1
            else "X",
            "nucleic_acid",
        )
    else:
        return mapped_residue, "ligand"


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_file(
    filepath: str,
    assume_one_based_residue_ids: bool = False,
    min_num_residues_for_protein_classification: int = 10,
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Parse an mmCIF file and return a dictionary mapping chain IDs
    to sequences for all molecule types (i.e., proteins, nucleic acids, peptides, ligands, etc)
    as well as a set of chain ID pairs denoting structural interfaces.
    """
    assert filepath.endswith(".cif"), "The input file must be an mmCIF file."
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    mmcif_object = parse_mmcif_object(filepath, file_id)
    model = mmcif_object.structure

    # NOTE: After dataset filtering, only heavy (non-hydrogen) atoms remain in the structure
    all_atoms = [atom for atom in model.get_atoms()]
    neighbor_search = NeighborSearch(all_atoms)

    sequences = {}
    interface_chain_ids = set()
    for chain in model:
        one_letter_seq_tokens = []
        token_molecule_types = set()

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
                token_molecule_types.add(clustering_molecule_type)

            # Find all interfaces defined as pairs of chains with minimum heavy atom (i.e. non-hydrogen) separation less than 5 Å
            for atom in res:
                for neighbor in neighbor_search.search(atom.coord, 5.0, "R"):
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

        assert (
            len(one_letter_seq_tokens) > 0
        ), f"No residues found in chain {chain.id} within the mmCIF file {filepath}."

        token_molecule_types = list(token_molecule_types)
        assert (
            len(token_molecule_types) == 1
        ), f"More than one molecule type found (i.e., {token_molecule_types}) in chain {chain.id} within the mmCIF file {filepath}."

        molecule_type = token_molecule_types[0]
        if (
            molecule_type == "protein"
            and len(one_letter_seq_tokens) < min_num_residues_for_protein_classification
        ):
            molecule_type = "peptide"

        one_letter_seq = "".join(one_letter_seq_tokens)
        sequences[f"{chain.id}:{molecule_type}"] = one_letter_seq

    return sequences, interface_chain_ids


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_directory(
    mmcif_dir: str, assume_one_based_residue_ids: bool = False
) -> Tuple[CHAIN_SEQUENCES, CHAIN_INTERFACES]:
    """
    Parse all mmCIF files in a directory and return a dictionary for each complex mapping chain IDs to sequences
    as well as a set of chain ID pairs denoting structural interfaces for each complex."""
    all_chain_sequences = []
    all_interface_chain_ids = {}

    mmcif_filepaths = list(glob.glob(os.path.join(mmcif_dir, "*", "*.cif")))
    for cif_filepath in tqdm(mmcif_filepaths, desc="Parsing chain sequences"):
        structure_id = os.path.splitext(os.path.basename(cif_filepath))[0]
        (
            chain_sequences,
            interface_chain_ids,
        ) = parse_chain_sequences_and_interfaces_from_mmcif_file(
            cif_filepath, assume_one_based_residue_ids=assume_one_based_residue_ids
        )
        all_chain_sequences.append({structure_id: chain_sequences})
        all_interface_chain_ids[structure_id] = list(interface_chain_ids)

    return all_chain_sequences, all_interface_chain_ids


@typecheck
def write_sequences_to_fasta(
    all_chain_sequences: CHAIN_SEQUENCES,
    fasta_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
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
                    if molecule_type_and_name[0] == molecule_type:
                        molecule_index_postfix = (
                            f"-{molecule_type_and_name[1]}"
                            if len(molecule_type_and_name) == 2
                            else ""
                        )
                        molecule_id = f"{structure_id}{chain_id_}:{molecule_type_and_name[0]}{molecule_index_postfix}"

                        f.write(f">{molecule_id}\n{sequence}\n")
                        molecule_ids.append(molecule_id)
    return molecule_ids


@typecheck
def run_clustalo(
    input_filepath: str,
    output_filepath: str,
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
):
    """Run Clustal Omega on the input FASTA file and write the aligned FASTA sequences and corresponding distance matrix to respective output files."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert output_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    output_filepath = output_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")

    assert os.path.isfile(input_filepath), f"Input file '{input_filepath}' does not exist."

    subprocess.run(
        [
            "clustalo",
            "-i",
            input_filepath,
            "-o",
            output_filepath,
            f"--distmat-out={distmat_filepath}",
            "--percent-id",
            "--full",
            "--force",
        ]
    )


@typecheck
def cluster_ligands_by_ccd_code(input_filepath: str, distmat_filepath: str):
    """Cluster ligands based on their CCD codes and write the resulting sequence distance matrix to a file."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", "_ligand.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", "_ligand.txt")

    # Parse the ligand FASTA input file into a dictionary
    ligands = {}
    with open(input_filepath, "r") as f:
        structure_id = None
        for line in f:
            if line.startswith(">"):
                structure_id = line[1:].strip()
                ligands[structure_id] = ""
            else:
                ligands[structure_id] += line.strip()

    # Convert ligands to a list of tuples for easier indexing
    ligand_structure_ids = list(ligands.keys())
    ligand_sequences = list(ligands.values())
    n = len(ligand_structure_ids)

    # Initialize the distance matrix efficiently
    distance_matrix = np.zeros((n, n))

    # Fill the distance matrix using only the upper triangle (symmetric)
    for i in range(n):
        for j in range(i, n):
            if ligand_sequences[i] == ligand_sequences[j]:
                distance_matrix[i, j] = 100.0
                distance_matrix[j, i] = 100.0

    # Write the ligand distance matrix to a NumPy-compatible text file
    with open(distmat_filepath, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = [ligand_structure_ids[i]] + list(map(str, distance_matrix[i]))
            f.write(" ".join(row) + "\n")


@typecheck
def read_distance_matrix(
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> Optional[np.ndarray]:
    """Read a distance matrix from a file and return it as a NumPy array."""
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")
    if not os.path.isfile(distmat_filepath):
        logger.warning(f"Distance matrix file '{distmat_filepath}' does not exist.")
        return None

    # Convert sequence matching percentages to distances through complementation
    df = pd.read_csv(distmat_filepath, sep="\s+", header=None, skiprows=1)
    matrix = 100.0 - df.values[:, 1:].astype(float) if len(df) > 0 else None

    return matrix


@typecheck
def cluster_interfaces(
    protein_chain_cluster_mapping: Dict[str, IntType],
    nucleic_acid_chain_cluster_mapping: Dict[str, IntType],
    peptide_chain_cluster_mapping: Dict[str, IntType],
    ligand_chain_cluster_mapping: Dict[str, IntType],
    interface_chain_ids: CHAIN_INTERFACES,
) -> INTERFACE_CLUSTERS:
    """Cluster interfaces based on the cluster IDs of the chains involved."""
    interface_chains_cluster_mapping = {}
    interface_clusters = {}

    for pdb_id in interface_chain_ids:
        for chain_id_pair in interface_chain_ids[pdb_id]:
            chain_ids = chain_id_pair.split("+")
            chain_clusters = []
            for chain_id in chain_ids:
                pdb_chain_id = f"{pdb_id}{chain_id}"
                molecule_id = chain_id.split(":")[-1]
                if "protein" in pdb_chain_id and pdb_chain_id in protein_chain_cluster_mapping:
                    chain_clusters.append(
                        f"{molecule_id}-cluster-{protein_chain_cluster_mapping[pdb_chain_id]}"
                    )
                elif (
                    "nucleic_acid" in pdb_chain_id
                    and pdb_chain_id in nucleic_acid_chain_cluster_mapping
                ):
                    chain_clusters.append(
                        f"{molecule_id}-cluster-{nucleic_acid_chain_cluster_mapping[pdb_chain_id]}"
                    )
                elif "peptide" in pdb_chain_id and pdb_chain_id in peptide_chain_cluster_mapping:
                    chain_clusters.append(
                        f"{molecule_id}-cluster-{peptide_chain_cluster_mapping[pdb_chain_id]}"
                    )
                elif "ligand" in pdb_chain_id and pdb_chain_id in ligand_chain_cluster_mapping:
                    chain_clusters.append(
                        f"{molecule_id}-cluster-{ligand_chain_cluster_mapping[pdb_chain_id]}"
                    )
                else:
                    raise ValueError(
                        f"Chain {pdb_chain_id} not found in any cluster mapping for PDB ID {pdb_id}."
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

    return interface_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster chains and interfaces within the AlphaFold 3 PDB dataset's filtered mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "mmcifs"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "clusterings"),
        help="Path to the output FASTA file.",
    )
    parser.add_argument(
        "--clustering_filtered_pdb_dataset",
        action="store_true",
        help="Whether the clustering is being performed on the filtered PDB dataset.",
    )
    args = parser.parse_args()

    # Validate input arguments
    assert os.path.isdir(args.mmcif_dir), f"mmCIF directory '{args.mmcif_dir}' does not exist."
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine paths for intermediate files

    fasta_filepath = os.path.join(args.output_dir, "sequences.fasta")
    aligned_fasta_filepath = os.path.join(args.output_dir, "aligned_sequences.fasta")
    distmat_filepath = os.path.join(args.output_dir, "distmat.txt")

    # Parse all chain sequences from mmCIF files

    (
        all_chain_sequences,
        interface_chain_ids,
    ) = parse_chain_sequences_and_interfaces_from_mmcif_directory(
        args.mmcif_dir, assume_one_based_residue_ids=args.clustering_filtered_pdb_dataset
    )

    # Attempt to load existing chain cluster mappings from local storage

    protein_chain_cluster_mapping = {}
    nucleic_acid_chain_cluster_mapping = {}
    peptide_chain_cluster_mapping = {}
    ligand_chain_cluster_mapping = {}

    if os.path.exists(os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")):
        protein_chain_cluster_mapping = (
            pd.read_csv(os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv"))
            .assign(
                combined_key=lambda df: df.apply(
                    lambda row: f"{row['pdb_id']}{row['chain_id']}:{row['molecule_id']}", axis=1
                )
            )
            .set_index("combined_key")["cluster_id"]
            .to_dict()
        )
    if os.path.exists(os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv")):
        nucleic_acid_chain_cluster_mapping = (
            pd.read_csv(os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv"))
            .assign(
                combined_key=lambda df: df.apply(
                    lambda row: f"{row['pdb_id']}{row['chain_id']}:{row['molecule_id']}", axis=1
                )
            )
            .set_index("combined_key")["cluster_id"]
            .to_dict()
        )
    if os.path.exists(os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv")):
        peptide_chain_cluster_mapping = (
            pd.read_csv(os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv"))
            .assign(
                combined_key=lambda df: df.apply(
                    lambda row: f"{row['pdb_id']}{row['chain_id']}:{row['molecule_id']}", axis=1
                )
            )
            .set_index("combined_key")["cluster_id"]
            .to_dict()
        )
    if os.path.exists(os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv")):
        ligand_chain_cluster_mapping = (
            pd.read_csv(os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv"))
            .assign(
                combined_key=lambda df: df.apply(
                    lambda row: f"{row['pdb_id']}{row['chain_id']}:{row['molecule_id']}", axis=1
                )
            )
            .set_index("combined_key")["cluster_id"]
            .to_dict()
        )

    # Align sequences separately for each molecule type and compute each respective distance matrix

    if not all(
        (
            protein_chain_cluster_mapping,
            nucleic_acid_chain_cluster_mapping,
            peptide_chain_cluster_mapping,
            ligand_chain_cluster_mapping,
        )
    ):
        protein_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="protein"
        )
        nucleic_acid_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="nucleic_acid"
        )
        peptide_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="peptide"
        )
        ligand_molecule_ids = write_sequences_to_fasta(
            all_chain_sequences, fasta_filepath, molecule_type="ligand"
        )

        run_clustalo(
            fasta_filepath,
            aligned_fasta_filepath,
            distmat_filepath,
            molecule_type="protein",
        )
        run_clustalo(
            fasta_filepath,
            aligned_fasta_filepath,
            distmat_filepath,
            molecule_type="nucleic_acid",
        )
        run_clustalo(
            fasta_filepath,
            aligned_fasta_filepath,
            distmat_filepath,
            molecule_type="peptide",
        )
        cluster_ligands_by_ccd_code(
            fasta_filepath,
            distmat_filepath,
        )

        protein_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="protein")
        nucleic_acid_dist_matrix = read_distance_matrix(
            distmat_filepath, molecule_type="nucleic_acid"
        )
        peptide_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="peptide")
        ligand_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="ligand")

        # Cluster residues at sequence homology levels corresponding to each molecule type

        protein_cluster_labels = (
            # Cluster proteins at 40% sequence homology
            AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=40.0 + 1e-6,
                metric="precomputed",
                linkage="complete",
            ).fit_predict(protein_dist_matrix)
            if exists(protein_dist_matrix)
            else None
        )

        nucleic_acid_cluster_labels = (
            # Cluster nucleic acids at 100% sequence homology
            AgglomerativeClustering(
                n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
            ).fit_predict(nucleic_acid_dist_matrix)
            if exists(nucleic_acid_dist_matrix)
            else None
        )

        peptide_cluster_labels = (
            # Cluster peptides at 100% sequence homology
            AgglomerativeClustering(
                n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
            ).fit_predict(peptide_dist_matrix)
            if exists(peptide_dist_matrix)
            else None
        )

        ligand_cluster_labels = (
            # Cluster ligands based on their CCD codes (i.e., identical ligands share a cluster)
            AgglomerativeClustering(
                n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
            ).fit_predict(ligand_dist_matrix)
            if exists(ligand_dist_matrix)
            else None
        )

        # Map PDB IDs and their constituent chain and molecule IDs to (molecule type-specific) cluster IDs, and save the mappings to local (CSV) storage

        protein_chain_cluster_mapping = (
            dict(zip(protein_molecule_ids, protein_cluster_labels))
            if exists(protein_cluster_labels)
            else {}
        )
        nucleic_acid_chain_cluster_mapping = (
            dict(zip(nucleic_acid_molecule_ids, nucleic_acid_cluster_labels))
            if exists(nucleic_acid_cluster_labels)
            else {}
        )
        peptide_chain_cluster_mapping = (
            dict(zip(peptide_molecule_ids, peptide_cluster_labels))
            if exists(peptide_cluster_labels)
            else {}
        )
        ligand_chain_cluster_mapping = (
            dict(zip(ligand_molecule_ids, ligand_cluster_labels))
            if exists(ligand_cluster_labels)
            else {}
        )

        if protein_chain_cluster_mapping:
            pd.DataFrame(
                (
                    (k.split(":")[0][:4], k.split(":")[0][4:], k.split(":")[1], v)
                    for k, v in protein_chain_cluster_mapping.items()
                ),
                columns=["pdb_id", "chain_id", "molecule_id", "cluster_id"],
            ).to_csv(
                os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv"), index=False
            )
        if nucleic_acid_chain_cluster_mapping:
            pd.DataFrame(
                (
                    (k.split(":")[0][:4], k.split(":")[0][4:], k.split(":")[1], v)
                    for k, v in nucleic_acid_chain_cluster_mapping.items()
                ),
                columns=["pdb_id", "chain_id", "molecule_id", "cluster_id"],
            ).to_csv(
                os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv"),
                index=False,
            )
        if peptide_chain_cluster_mapping:
            pd.DataFrame(
                (
                    (k.split(":")[0][:4], k.split(":")[0][4:], k.split(":")[1], v)
                    for k, v in peptide_chain_cluster_mapping.items()
                ),
                columns=["pdb_id", "chain_id", "molecule_id", "cluster_id"],
            ).to_csv(
                os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv"), index=False
            )
        if ligand_chain_cluster_mapping:
            pd.DataFrame(
                (
                    (k.split(":")[0][:4], k.split(":")[0][4:], k.split(":")[1], v)
                    for k, v in ligand_chain_cluster_mapping.items()
                ),
                columns=["pdb_id", "chain_id", "molecule_id", "cluster_id"],
            ).to_csv(
                os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv"), index=False
            )

    # Cluster interfaces based on the cluster IDs of the chains involved, and save the interface cluster mapping to local (CSV) storage

    interface_cluster_mapping = cluster_interfaces(
        protein_chain_cluster_mapping,
        nucleic_acid_chain_cluster_mapping,
        peptide_chain_cluster_mapping,
        ligand_chain_cluster_mapping,
        interface_chain_ids,
    )

    pd.DataFrame(
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
            for k, v in interface_cluster_mapping.items()
        ),
        columns=[
            "pdb_id",
            "interface_chain_id_1",
            "interface_chain_id_2",
            "interface_molecule_id_1",
            "interface_molecule_id_2",
            "interface_chain_cluster_id_1",
            "interface_chain_cluster_id_2",
            "interface_cluster_id",
        ],
    ).to_csv(os.path.join(args.output_dir, "interface_cluster_mapping.csv"), index=False)

# %%
