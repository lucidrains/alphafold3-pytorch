"""General-purpose data pipeline."""

import os
from typing import Dict, List, MutableMapping, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from alphafold3_pytorch.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
    to_mmcif,
)
from alphafold3_pytorch.data import mmcif_parsing, msa_parsing
from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

# Constants

GAP_ID = get_residue_constants("peptide").MSA_CHAR_TO_ID["-"]

FeatureDict = MutableMapping[str, np.ndarray | Tensor]


@typecheck
def make_sequence_features(sequence: str, description: str) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=object)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=object)
    return features


@typecheck
def make_msa_mask(features: FeatureDict) -> FeatureDict:
    """
    Make MSA mask features.
    From: https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_transforms.py#L379
    NOTE: Openfold Mask features are all ones, but will later be zero-padded.

    :param features: The features dictionary.
    :return: The features dictionary with new MSA mask features.
    """
    features["msa_mask"] = torch.ones(features["msa"].shape[1], dtype=torch.float32)
    features["msa_row_mask"] = torch.ones((features["msa"].shape[0]), dtype=torch.float32)
    return features


@typecheck
def make_msa_features(
    msas: Dict[str, msa_parsing.Msa | None],
    chain_id_to_residue: Dict[str, Dict[str, List[int]]],
    ligand_chemtype_index: int = 3,
    raise_missing_exception: bool = False,
) -> FeatureDict:
    """
    Construct a feature dictionary of MSA features.
    From: https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_pipeline.py#L224

    :param msas: The mapping of chain IDs to lists of (optional) MSAs for each chain.
    :param chain_id_to_residue: The mapping of chain IDs to residue information.
    :param ligand_index: The index of the ligand in the chemical type list.
    :param raise_missing_exception: Whether to raise an exception if no MSAs are provided for any chain.
    :return: The MSA feature dictionary.
    """
    if not msas:
        raise ValueError("At least one chain's MSA must be provided.")

    # Infer MSA metadata.
    max_alignments = 1
    for msa in msas.values():
        if exists(msa) and exists(msa.sequences) and exists(msa.sequences[0]):
            max_alignments = max(max_alignments, len(msa.sequences) if msa else 1)

    # Collect MSAs.
    int_msa_list = []
    deletion_matrix_list = []
    species_ids_list = []

    for chain_id, msa in msas.items():
        int_msa = []
        deletion_matrix = []
        species_ids = []
        seen_sequences = set()

        chain_chemtype = chain_id_to_residue[chain_id]["chemtype"]
        chain_residue_index = chain_id_to_residue[chain_id]["residue_index"]

        num_res = len(chain_chemtype)
        assert num_res == len(chain_residue_index), (
            f"Residue features count mismatch for chain {chain_id}: "
            f"{num_res} != {len(chain_residue_index)}"
        )

        msa_residue_constants = (
            get_residue_constants(msa.msa_type.replace("protein", "peptide"))
            if exists(msa)
            else None
        )

        gap_ids = [[GAP_ID] * num_res]
        deletion_values = [[0] * num_res]
        species = ["".encode("utf-8")]

        if not msa and raise_missing_exception:
            raise ValueError(f"MSA for chain {chain_id} must contain at least one sequence.")
        elif not msa:
            # Pad the MSA to the maximum number of alignments
            # if the chain does not have any associated alignments.
            int_msa_list.append(torch.tensor(gap_ids * max_alignments, dtype=torch.long))
            deletion_matrix_list.append(
                torch.tensor(deletion_values * max_alignments, dtype=torch.float32)
            )
            species_ids_list.append(np.array(species * max_alignments, dtype=object))
            continue

        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)

            # Convert the MSA to integers while handling
            # ligands and modified polymer residues.
            msa_res_types = []
            msa_deletion_values = []

            polymer_residue_index = -1

            for idx, (chemtype, residue_index) in enumerate(
                zip(chain_chemtype, chain_residue_index)
            ):
                is_polymer = chemtype < ligand_chemtype_index
                is_ligand = not is_polymer

                chem_residue_constants = get_residue_constants(res_chem_index=chemtype)

                # NOTE: For modified polymer residues, we only increment the polymer residue index
                # when the current (atomized) modified polymer residue's atom sequence ends.
                increment_index = (
                    0 < idx < num_res and chain_residue_index[idx - 1] != residue_index
                )
                polymer_residue_index += 1 if is_polymer and (idx == 0 or increment_index) else 0

                if is_ligand:
                    # NOTE: For ligands, we use the unknown amino acid type.
                    msa_res_type = chem_residue_constants.restype_num
                    msa_deletion_value = 0
                else:
                    # NOTE: For polymer residues of a different chemical type than the chain's MSA,
                    # we use the unknown residue type of the corresponding chemical type
                    # (e.g., `DN` for DNA residues in a protein chain's MSA). This should
                    # provide the model with partial information about the residue's identity.
                    if chem_residue_constants != msa_residue_constants:
                        msa_res_type = chem_residue_constants.restype_num
                    else:
                        res = sequence[polymer_residue_index]
                        msa_res_type = msa_residue_constants.MSA_CHAR_TO_ID.get(
                            res, msa_residue_constants.restype_num
                        )

                    msa_deletion_value = msa.deletion_matrix[sequence_index][polymer_residue_index]

                msa_res_types.append(msa_res_type)
                msa_deletion_values.append(msa_deletion_value)

            assert polymer_residue_index + 1 == len(
                sequence
            ), f"Polymer residue index length mismatch for MSA chain {chain_id}: {polymer_residue_index + 1} != {len(sequence)}"

            int_msa.append(msa_res_types)
            deletion_matrix.append(msa_deletion_values)

            identifiers = msa_parsing.get_identifiers(msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode("utf-8"))

        # Pad the MSA to the maximum number of alignments.
        num_padding_alignments = max_alignments - len(int_msa)

        padding_msa = gap_ids * num_padding_alignments
        padding_deletion_matrix = deletion_values * num_padding_alignments
        padding_species_ids = species * num_padding_alignments

        int_msa_list.append(torch.tensor(int_msa + padding_msa, dtype=torch.long))
        deletion_matrix_list.append(
            torch.tensor(deletion_matrix + padding_deletion_matrix, dtype=torch.float32)
        )
        species_ids_list.append(np.array(species_ids + padding_species_ids, dtype=object))

    features = {
        "msa": torch.cat(int_msa_list, dim=-1),
        "deletion_matrix": torch.cat(deletion_matrix_list, dim=-1),
        "msa_species_identifiers": np.stack(species_ids_list),
        "num_alignments": max_alignments,
    }
    return features


@typecheck
def get_assembly(biomol: Biomolecule, assembly_id: Optional[str] = None) -> Biomolecule:
    """Get a specified (Biomolecule) assembly of a given Biomolecule.

    Adapted from: https://github.com/biotite-dev/biotite

    :param biomol: The Biomolecule from which to extract the requested assembly.
    :param assembly_id: The index of the assembly to get.
    :return: The requested assembly.
    """
    # Get mmCIF metadata categories
    assembly_category = mmcif_parsing.mmcif_loop_to_dict(
        "_pdbx_struct_assembly.", "_pdbx_struct_assembly.id", biomol.mmcif_metadata
    )
    assembly_gen_category = mmcif_parsing.mmcif_loop_to_dict(
        "_pdbx_struct_assembly_gen.",
        "_pdbx_struct_assembly_gen.assembly_id",
        biomol.mmcif_metadata,
    )
    struct_oper_category = mmcif_parsing.mmcif_loop_to_dict(
        "_pdbx_struct_oper_list.", "_pdbx_struct_oper_list.id", biomol.mmcif_metadata
    )

    if not all((assembly_category, assembly_gen_category, struct_oper_category)):
        logger.warning(
            "Not all required assembly information was found in the mmCIF file. Returning the input biomolecule."
        )
        return biomol

    assembly_ids = sorted(list(assembly_gen_category.keys()))
    if assembly_id is None:
        # NOTE: Sorting ensures that the default assembly is the first biological assembly.
        assembly_id = assembly_ids[0]
    elif assembly_id not in assembly_ids:
        raise KeyError(
            f"Biomolecule has no assembly ID `{assembly_id}`. Available assembly IDs: `{assembly_ids}`."
        )

    # Calculate all possible transformations
    transformations = mmcif_parsing.get_transformations(struct_oper_category)

    # Get transformations and apply them to the affected asym IDs
    assembly = None
    for id in assembly_gen_category:
        op_expr = assembly_gen_category[id]["_pdbx_struct_assembly_gen.oper_expression"]
        asym_id_expr = assembly_gen_category[id]["_pdbx_struct_assembly_gen.asym_id_list"]

        # Find the operation expressions for given assembly ID,
        # where we already asserted that the ID is actually present
        if id == assembly_id:
            operations = mmcif_parsing.parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")
            # Filter affected asym IDs
            sub_assembly = mmcif_parsing.apply_transformations(
                biomol.subset_chains(asym_ids),
                transformations,
                operations,
            )
            # Merge the chains with asym IDs for this operation
            # with chains from other operations
            if assembly is None:
                assembly = sub_assembly
            else:
                assembly += sub_assembly

    assert exists(assembly), f"Assembly with ID `{assembly_id}` not found."
    return assembly


@typecheck
def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject,
) -> Tuple[FeatureDict, Biomolecule]:
    """Make features from an mmCIF object."""
    input_sequence = "".join(
        mmcif_object.chain_to_seqres[chain_id] for chain_id in mmcif_object.chain_to_seqres
    )
    description = mmcif_object.file_id

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
        )
    )

    # As necessary, expand the first bioassembly/model sequence and structure, to obtain a biologically relevant complex (AF3 Supplement, Section 2.1).
    # Reference: https://github.com/biotite-dev/biotite/blob/1045f43f80c77a0dc00865e924442385ce8f83ab/src/biotite/structure/io/pdbx/convert.py#L1441

    assembly = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in description
        else get_assembly(_from_mmcif_object(mmcif_object))
    )

    mmcif_feats["all_atom_positions"] = assembly.atom_positions
    mmcif_feats["all_atom_mask"] = assembly.atom_mask
    mmcif_feats["b_factors"] = assembly.b_factors
    mmcif_feats["chain_index"] = assembly.chain_index
    mmcif_feats["chain_id"] = assembly.chain_id
    mmcif_feats["chemid"] = assembly.chemid
    mmcif_feats["chemtype"] = assembly.chemtype
    mmcif_feats["residue_index"] = assembly.residue_index
    mmcif_feats["restype"] = assembly.restype

    mmcif_feats["bonds"] = mmcif_object.bonds

    mmcif_feats["resolution"] = np.array([mmcif_object.header["resolution"]], dtype=np.float32)

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    )

    mmcif_feats["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_feats, assembly


if __name__ == "__main__":
    filepath = os.path.join("data", "test", "7a4d-assembly1.cif")
    file_id = os.path.splitext(os.path.basename(filepath))[0]

    mmcif_object = mmcif_parsing.parse_mmcif_object(
        filepath=filepath,
        file_id=file_id,
    )
    mmcif_feats, assembly = make_mmcif_features(mmcif_object)
    cropped_assembly, _, _ = assembly.crop(
        contiguous_weight=0.2,
        spatial_weight=0.4,
        spatial_interface_weight=0.4,
        n_res=384,
        chain_1="A",
        chain_2="B",
    )
    mmcif_string = to_mmcif(
        # assembly,
        cropped_assembly,
        file_id=file_id,
        gapless_poly_seq=True,
        insert_alphafold_mmcif_metadata=False,
        # unique_res_atom_names=assembly.unique_res_atom_names,
        unique_res_atom_names=cropped_assembly.unique_res_atom_names,
    )
    with open(os.path.basename(filepath).replace(".cif", "_reconstructed.cif"), "w") as f:
        f.write(mmcif_string)

    print(f"Successfully reconstructed {filepath} after mmCIF featurization.")
