# %% [markdown]
# # Clustering AlphaFold 3 PDB Validation Dataset
#
# For clustering AlphaFold 3's PDB validation dataset, we propose a modified (i.e., more stringent) version of the
# validation dataset's clustering procedure outlined in Abramson et al (2024).
#
# The process for selecting these targets was broken up into two separate stages. The first was for selecting multimers,
# the second for selecting monomers. Multimer selection proceeded as follows:
#
# # ... (see the PDB validation set filtering script)
# 2. Filter to only low homology interfaces, which are defined as those where no target in the training set contains
# a chain with high homology to either chain involved in the interface, where high homology here means >
# 40% sequence identity for polymers or > 0.85 Tanimoto similarity for ligands. For non-interfacing polymers within a
# multimeric structure, for clustering we retain only those polymers that are of low homology to the training set along
# with any remaining ligands within the filtered structure.
# 3. Assign interfaces to clusters as per subsubsection 2.5.3.
# 4. Take the following interface types only, possibly reducing number of clusters by sampling a subset of clusters
# (number of samples given in brackets if reduced): protein-protein (600), protein-DNA (100), DNA-DNA (100),
# Protein-ligand (600), DNA-ligand (50), ligand-ligand (200), protein-RNA, RNA-RNA, DNA-RNA, RNA-ligand.
# 5. Take the set of all PDB targets containing the remaining interfaces and make the set of scored chains
# and interfaces equal to all low homology chains and interfaces in those targets.
#
# Monomer selection proceeded similarly:
#
# ... (see the PDB validation set filtering script)
# 2. Filter to only low homology polymers and any ligands within the filtered structure, where low homology polymers
# are defined as those where no target in the training set contains a chain with high homology to the polymer and
# where high homology here means > 40% sequence identity for the polymers.
# 3. Assign polymers to clusters as per subsubsection 2.5.3.
# 4. Sample 40 protein monomers and take all DNA and RNA monomers.
# 5. Make the set of scored chains and interfaces equal to all low homology chains and interfaces in the remaining targets.

# %%

import argparse
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from beartype.typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
import timeout_decorator
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from alphafold3_pytorch.inputs import CCD_COMPONENTS_SMILES
from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists
from scripts.cluster_pdb_train_mmcifs import (
    CHAIN_INTERFACES,
    CHAIN_SEQUENCES,
    CLUSTERING_MOLECULE_TYPE,
    cluster_interfaces,
    cluster_ligands_by_ccd_code,
    cluster_sequences_using_mmseqs2,
    parse_chain_sequences_and_interfaces_from_mmcif_directory,
    write_sequences_to_fasta,
)

# Constants

CLUSTERING_POLYMER_MOLECULE_TYPES = {"protein", "rna", "dna", "peptide"}

INTERFACE_SAMPLE_SIZES = {
    "protein-protein": 600,
    "dna-protein": 100,
    "dna-dna": 100,
    "ligand-protein": 600,
    "dna-ligand": 50,
    "ligand-ligand": 200,
    # NOTE: `None` implies all rows are taken
    "protein-rna": None,
    "rna-rna": None,
    "dna-rna": None,
    "ligand-rna": None,
}

IS_NOVEL_LIGAND_MAX_SECONDS_PER_INPUT = (
    20  # Maximum time allocated to check a single ligand for novelty (in seconds)
)


# Helper functions


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
def search_sequences_using_mmseqs2(
    input_filepath: str,
    reference_filepath: str,
    output_dir: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
    max_seq_id: float = 0.4,
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
        # high sensitivity, e.g., 8)

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


@timeout_decorator.timeout(IS_NOVEL_LIGAND_MAX_SECONDS_PER_INPUT, use_signals=False)
def is_novel_ligand(
    ligand_sequence: str,
    reference_ligand_fps: List[DataStructs.cDataStructs.ExplicitBitVect],
    max_sim: float = 0.85,
    verbose: bool = False,
) -> bool:
    """Check if a ligand sequence is novel based on Tanimoto similarity to a reference set of ligand sequences."""
    fpgen = AllChem.GetRDKitFPGenerator()
    ligand_smiles = CCD_COMPONENTS_SMILES.get(ligand_sequence, None)
    if not exists(ligand_smiles):
        if verbose:
            logger.warning(f"Could not find SMILES for ligand sequence: {ligand_sequence}")
        return False
    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    if not exists(ligand_mol):
        if verbose:
            logger.warning(
                f"Could not generate RDKit molecule for ligand sequence: {ligand_sequence}"
            )
        return False
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
) -> Tuple[str, Dict[str, str], CHAIN_INTERFACES]:
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

                    # NOTE: Only if both of the interface's chains are novel
                    # will the interface be kept. For the validation dataset,
                    # this is the only filter that screens for novel ligands.
                    interface_is_novel = ptnr1_is_novel and ptnr2_is_novel
                    if interface_is_novel:
                        # NOTE: If at least one of a chain's associated interfaces
                        # are novel, the chain will be kept.
                        filtered_structure_chain_sequences[chain_id] = sequence
                        if (
                            f"{ptnr2_chain_id}:{ptnr1_chain_id}"
                            not in filtered_interface_chain_ids[structure_id]
                        ):
                            filtered_interface_chain_ids[structure_id].add(interface)

            elif sequence_name in sequence_names or (
                structure_id in filtered_structure_ids and molecule_type == "ligand"
            ):
                # NOTE: For the validation dataset, non-interfacing polymer chains within a
                # multimeric structure are kept only if the polymer chain is novel, and any
                # ligand chains within the (polymer-)filtered structure are kept. In other
                # words, here we do not filter for only novel ligands, but in the context
                # of the evaluation dataset, we will only keep novel ligands.
                filtered_structure_chain_sequences[chain_id] = sequence

        elif sequence_name in sequence_names or (
            structure_id in filtered_structure_ids and molecule_type == "ligand"
        ):
            # NOTE: For the validation dataset's monomers, sequence non-redundant polymers or
            # any ligand chains within the (polymer-)filtered structure are kept. In other words,
            # here we do not filter for only novel ligands, but in the context of the evaluation
            # dataset, we will only keep novel ligands.
            filtered_structure_chain_sequences[chain_id] = sequence

    return structure_id, filtered_structure_chain_sequences, filtered_interface_chain_ids


@typecheck
def filter_chains_by_sequence_names(
    all_chain_sequences: CHAIN_SEQUENCES,
    sequence_names: Set[str] | np.ndarray,
    interface_chain_ids: CHAIN_INTERFACES | None = None,
    reference_ligand_fps: List[DataStructs.cDataStructs.ExplicitBitVect] | None = None,
    max_polymer_similarity: float = 0.4,
    max_ligand_similarity: float = 0.85,
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
def filter_to_low_homology_sequences(
    input_all_chain_sequences: CHAIN_SEQUENCES,
    reference_all_chain_sequences: CHAIN_SEQUENCES,
    input_interface_chain_ids: CHAIN_INTERFACES,
    input_fasta_filepath: str,
    reference_fasta_filepath: str,
    max_polymer_similarity: float = 0.4,
    max_ligand_similarity: float = 0.85,
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
    )
    write_sequences_to_fasta(
        input_multimer_chain_sequences,
        input_multimer_fasta_filepath,
        molecule_type="nucleic_acid",
    )
    write_sequences_to_fasta(
        input_multimer_chain_sequences,
        input_multimer_fasta_filepath,
        molecule_type="peptide",
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
        input_monomer_chain_sequences,
        input_monomer_sequence_names,
        max_polymer_similarity=max_polymer_similarity,
        max_ligand_similarity=max_ligand_similarity,
        max_workers=max_workers,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster chains and interfaces within the AlphaFold 3 PDB validation dataset's filtered mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "val_mmcifs"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "--reference_clustering_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "train_clusterings"),
        help="Path to the reference clustering directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "val_clusterings"),
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
    reference_fasta_filepath = os.path.join(args.reference_clustering_dir, "sequences.fasta")

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
            os.path.join(args.reference_clustering_dir, "all_chain_sequences.json"), "r"
        ) as f:
            reference_all_chain_sequences = json.load(f)

        (
            all_chain_sequences,
            interface_chain_ids,
        ) = filter_to_low_homology_sequences(
            all_chain_sequences,
            reference_all_chain_sequences,
            interface_chain_ids,
            fasta_filepath,
            reference_fasta_filepath,
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

    # Load current clusters to subsample chain and interface clusters according to Step(s) 4 of Section 5.8 of the AF3 supplement

    protein_chain_clusters = pl.read_csv(
        os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")
    )
    nucleic_acid_chain_clusters = pl.read_csv(
        os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv")
    )
    peptide_chain_clusters = pl.read_csv(
        os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv")
    )
    ligand_chain_clusters = pl.read_csv(
        os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv")
    )
    interface_clusters = pl.read_csv(
        os.path.join(args.output_dir, "interface_cluster_mapping.csv")
    )

    # Subsample protein monomer chain clusters to 40 random clusters,
    # and leave the RNA and DNA monomer chain clusters unchanged

    (
        _,
        multimer_chain_sequences,
    ) = separate_monomer_and_multimer_chain_sequences(all_chain_sequences)
    multimer_pdb_ids = set(
        list(multimer_chain_sequence.keys())[0]
        for multimer_chain_sequence in multimer_chain_sequences
    )
    # Retain all protein multimer chain clusters during monomer chain subsampling
    protein_multimer_chain_clusters = protein_chain_clusters.filter(
        pl.col("pdb_id").is_in(multimer_pdb_ids)
    )

    protein_chain_clusters_grouped = protein_chain_clusters.group_by("cluster_id").agg(
        pl.col("*").sample(1, seed=42)
    )
    protein_random_chain_clusters = np.random.choice(
        protein_chain_clusters_grouped.shape[0], 40, replace=False
    )
    protein_chain_clusters = (
        protein_chain_clusters_grouped[protein_random_chain_clusters]
        .with_columns(
            [
                pl.col("pdb_id").explode().alias("pdb_id"),
                pl.col("chain_id").explode().alias("chain_id"),
                pl.col("molecule_id").explode().alias("molecule_id"),
            ]
        )
        .select(["pdb_id", "chain_id", "molecule_id", "cluster_id"])
    )
    protein_chain_clusters = pl.concat([protein_chain_clusters, protein_multimer_chain_clusters])
    protein_chain_clusters.write_csv(
        os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv")
    )

    # Subsample interface cluster types

    interface_clusters = interface_clusters.with_columns(
        pl.concat_list(
            [
                pl.col("interface_molecule_id_1").str.split_exact("-", 0).struct.field("field_0"),
                pl.col("interface_molecule_id_2").str.split_exact("-", 0).struct.field("field_0"),
            ]
        )
        .list.sort()
        .list.join("-")
        .alias("interface_type")
    )

    sampled_interface_dataframes = []
    for interface_type, sample_size in INTERFACE_SAMPLE_SIZES.items():
        filtered_interface_df = interface_clusters.filter(
            pl.col("interface_type") == interface_type
        )
        if exists(sample_size):
            interface_chain_clusters_grouped = filtered_interface_df.group_by(
                "interface_cluster_id"
            ).agg(pl.col("*").sample(1, seed=42))
            interface_random_chain_clusters = np.random.choice(
                interface_chain_clusters_grouped.shape[0],
                min(len(interface_chain_clusters_grouped), sample_size),
                replace=False,
            )
            sampled_interface_df = interface_chain_clusters_grouped[
                interface_random_chain_clusters
            ].with_columns(
                [
                    pl.col("pdb_id").explode().alias("pdb_id"),
                    pl.col("interface_chain_id_1").explode().alias("interface_chain_id_1"),
                    pl.col("interface_chain_id_2").explode().alias("interface_chain_id_2"),
                    pl.col("interface_molecule_id_1").explode().alias("interface_molecule_id_1"),
                    pl.col("interface_molecule_id_2").explode().alias("interface_molecule_id_2"),
                    pl.col("interface_chain_cluster_id_1")
                    .explode()
                    .alias("interface_chain_cluster_id_1"),
                    pl.col("interface_chain_cluster_id_2")
                    .explode()
                    .alias("interface_chain_cluster_id_2"),
                ]
            )
        else:
            sampled_interface_df = filtered_interface_df
        sampled_interface_dataframes.append(
            sampled_interface_df.select(
                [
                    "pdb_id",
                    "interface_chain_id_1",
                    "interface_chain_id_2",
                    "interface_molecule_id_1",
                    "interface_molecule_id_2",
                    "interface_chain_cluster_id_1",
                    "interface_chain_cluster_id_2",
                    "interface_cluster_id",
                ]
            )
        )
    interface_clusters = pl.concat(sampled_interface_dataframes)
    interface_clusters.write_csv(os.path.join(args.output_dir, "interface_cluster_mapping.csv"))
