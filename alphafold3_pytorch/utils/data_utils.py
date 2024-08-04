from typing import Any, Dict, Literal, Set

import numpy as np

from alphafold3_pytorch.tensor_typing import ChainType, ResidueType, typecheck

# constants

RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand"]
PDB_INPUT_RESIDUE_MOLECULE_TYPE = Literal[
    "protein", "rna", "dna", "mod_protein", "mod_rna", "mod_dna", "ligand"
]


@typecheck
def is_polymer(
    res_chem_type: str, polymer_chem_types: Set[str] = {"peptide", "dna", "rna"}
) -> bool:
    """
    Check if a residue is polymeric using its chemical type string.

    :param res_chem_type: The chemical type of the residue as a descriptive string.
    :param polymer_chem_types: The set of polymer chemical types.
    :return: Whether the residue is polymeric.
    """
    return any(chem_type in res_chem_type.lower() for chem_type in polymer_chem_types)


@typecheck
def is_water(res_name: str, water_res_names: Set[str] = {"HOH", "WAT"}) -> bool:
    """
    Check if a residue is a water residue using its residue name string.

    :param res_name: The name of the residue as a descriptive string.
    :param water_res_names: The set of water residue names.
    :return: Whether the residue is a water residue.
    """
    return any(water_res_name in res_name.upper() for water_res_name in water_res_names)


@typecheck
def is_atomized_residue(
    res_name: str, atomized_res_mol_types: Set[str] = {"ligand", "mod"}
) -> bool:
    """Check if a residue is an atomized residue using its residue molecule type string.

    :param res_name: The name of the residue as a descriptive string.
    :param atomized_res_mol_types: The set of atomized residue molecule types as strings.
    :return: Whether the residue is an atomized residue.
    """
    return any(mol_type in res_name.lower() for mol_type in atomized_res_mol_types)


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
def get_pdb_input_residue_molecule_type(
    res_chem_type: str, is_modified_polymer_residue: bool = False
) -> PDB_INPUT_RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue."""
    if "peptide" in res_chem_type.lower():
        return "mod_protein" if is_modified_polymer_residue else "protein"
    elif "rna" in res_chem_type.lower():
        return "mod_rna" if is_modified_polymer_residue else "rna"
    elif "dna" in res_chem_type.lower():
        return "mod_dna" if is_modified_polymer_residue else "dna"
    else:
        return "ligand"


@typecheck
def get_biopython_chain_residue_by_composite_id(
    chain: ChainType, res_name: str, res_id: int
) -> ResidueType:
    """
    Get a Biopython `Residue` or `DisorderedResidue` object
    by its residue name-residue index composite ID.

    :param chain: Biopython `Chain` object
    :param res_name: Residue name
    :param res_id: Residue index
    :return: Biopython `Residue` or `DisorderedResidue` object
    """
    if ("", res_id, " ") in chain:
        res = chain[("", res_id, " ")]
    elif (" ", res_id, " ") in chain:
        res = chain[(" ", res_id, " ")]
    elif (
        f"H_{res_name}",
        res_id,
        " ",
    ) in chain:
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                " ",
            )
        ]
    else:
        assert (
            f"H_{res_name}",
            res_id,
            "A",
        ) in chain, f"Version A of residue {res_name} of ID {res_id} in chain {chain.id} was missing from the chain's structure."
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                "A",
            )
        ]
    return res


@typecheck
def matrix_rotate(v: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Perform a rotation using a rotation matrix.

    :param v: The coordinates to rotate.
    :param matrix: The rotation matrix.
    :return: The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


@typecheck
def deep_merge_dicts(
    dict1: Dict[Any, Any], dict2: Dict[Any, Any], value_op: Literal["union", "concat"]
) -> Dict[Any, Any]:
    """
    Deeply merge two dictionaries, merging values where possible.

    :param dict1: The first dictionary to merge.
    :param dict2: The second dictionary to merge.
    :param value_op: The merge operation to perform on the values of matching keys.
    :return: The merged dictionary.
    """
    # Iterate over items in dict2
    for key, value in dict2.items():
        # If key is in dict1, merge the values
        if key in dict1:
            merged_value = dict1[key] + value
            if value_op == "union":
                dict1[key] = list(dict.fromkeys(merged_value))  # Preserve order
            else:
                dict1[key] = merged_value
        else:
            # Otherwise, set/overwrite the key in dict1 with dict2's value
            dict1[key] = value
    return dict1
