"""An mmCIF file format writer."""

from typing import List

from alphafold3_pytorch.common.biomolecule import (
    _from_mmcif_object,
    get_residue_constants,
    to_mmcif,
)
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject
from alphafold3_pytorch.utils.data_utils import is_polymer


def get_unique_res_atom_names(mmcif_object: MmcifObject) -> List[List[List[str]]]:
    """Get atom names for each (e.g. ligand) "pseudoresidue" of each residue in each chain."""
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
            unique_res_atom_names.append(atoms_to_append)
    return unique_res_atom_names


def write_mmcif(
    mmcif_object: MmcifObject,
    output_filepath: str,
    gapless_poly_seq: bool = True,
    insert_orig_atom_names: bool = True,
    insert_alphafold_mmcif_metadata: bool = True,
):
    """Write a BioPython `Structure` object to an mmCIF file using an intermediate `Biomolecule` object."""
    biomol = _from_mmcif_object(mmcif_object)
    unique_res_atom_names = (
        get_unique_res_atom_names(mmcif_object) if insert_orig_atom_names else None
    )
    mmcif_string = to_mmcif(
        biomol,
        mmcif_object.file_id,
        gapless_poly_seq=gapless_poly_seq,
        insert_alphafold_mmcif_metadata=insert_alphafold_mmcif_metadata,
        unique_res_atom_names=unique_res_atom_names,
    )
    with open(output_filepath, "w") as f:
        f.write(mmcif_string)
