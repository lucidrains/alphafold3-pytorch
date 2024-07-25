"""General-purpose data pipeline."""

import os

from loguru import logger
from typing import MutableMapping, Optional, Tuple

import numpy as np

from alphafold3_pytorch.common.biomolecule import Biomolecule, _from_mmcif_object, to_mmcif
from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.utils import exists

FeatureDict = MutableMapping[str, np.ndarray]


def make_sequence_features(sequence: str, description: str) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=object)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=object)
    return features


def get_assembly(biomol: Biomolecule, assembly_id: Optional[str] = None) -> Biomolecule:
    """
    Get a specified (Biomolecule) assembly of a given Biomolecule.

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
    cropped_assembly = assembly.crop(
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
