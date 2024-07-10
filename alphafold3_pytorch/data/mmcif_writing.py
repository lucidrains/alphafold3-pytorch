"""An mmCIF file format writer."""

from alphafold3_pytorch.common.biomolecule import (
    _from_mmcif_object,
    to_mmcif,
)
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject


def write_mmcif(
    mmcif_object: MmcifObject,
    output_filepath: str,
    gapless_poly_seq: bool = True,
    insert_orig_atom_names: bool = True,
    insert_alphafold_mmcif_metadata: bool = True,
):
    """Write a BioPython `Structure` object to an mmCIF file using an intermediate `Biomolecule` object."""
    biomol = _from_mmcif_object(mmcif_object)
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
