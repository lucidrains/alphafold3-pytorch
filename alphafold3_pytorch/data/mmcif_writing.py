"""An mmCIF file format writer."""

import numpy as np

from alphafold3_pytorch.common.biomolecule import (
    _from_mmcif_object,
    to_mmcif,
)
from alphafold3_pytorch.data.data_pipeline import get_assembly
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject, parse_mmcif_object
from alphafold3_pytorch.utils.utils import exists


def write_mmcif_from_filepath_and_id(
    input_filepath: str, output_filepath: str, file_id: str, **kwargs
):
    """Write an input mmCIF file to an output mmCIF filepath using the provided keyword arguments
    (e.g., sampled coordinates)."""
    mmcif_object = parse_mmcif_object(filepath=input_filepath, file_id=file_id)
    return write_mmcif(mmcif_object, output_filepath=output_filepath, **kwargs)


def write_mmcif(
    mmcif_object: MmcifObject,
    output_filepath: str,
    gapless_poly_seq: bool = True,
    insert_orig_atom_names: bool = True,
    insert_alphafold_mmcif_metadata: bool = True,
    sampled_atom_positions: np.ndarray | None = None,
):
    """Write a BioPython `Structure` object to an mmCIF file using an intermediate `Biomolecule` object."""
    biomol = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in mmcif_object.file_id
        else get_assembly(_from_mmcif_object(mmcif_object))
    )
    if exists(sampled_atom_positions):
        atom_mask = biomol.atom_mask.astype(bool)
        assert biomol.atom_positions[atom_mask].shape == sampled_atom_positions.shape, (
            f"Expected sampled atom positions to have masked shape {biomol.atom_positions[atom_mask].shape}, "
            f"but got {sampled_atom_positions.shape}."
        )
        biomol.atom_positions[atom_mask] = sampled_atom_positions
    unique_res_atom_names = biomol.unique_res_atom_names if insert_orig_atom_names else None
    mmcif_string = to_mmcif(
        biomol,
        mmcif_object.file_id,
        gapless_poly_seq=gapless_poly_seq,
        insert_alphafold_mmcif_metadata=insert_alphafold_mmcif_metadata,
        unique_res_atom_names=unique_res_atom_names,
    )
    with open(output_filepath, "w") as f:
        f.write(mmcif_string)
