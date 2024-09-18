import csv

import numpy as np
import torch
from beartype.typing import Any, Dict, Iterable, List, Literal, Set, Tuple
from torch import Tensor

from alphafold3_pytorch.tensor_typing import ChainType, ResidueType, typecheck
from alphafold3_pytorch.utils.utils import exists

# constants

RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand"]
PDB_INPUT_RESIDUE_MOLECULE_TYPE = Literal[
    "protein", "rna", "dna", "mod_protein", "mod_rna", "mod_dna", "ligand"
]
MMCIF_METADATA_FIELD = Literal[
    "structure_method", "release_date", "resolution", "structure_connectivity"
]


@typecheck
def is_polymer(
    res_chem_type: str, polymer_chem_types: Set[str] = {"peptide", "dna", "rna"}
) -> bool:
    """Check if a residue is polymeric using its chemical type string.

    :param res_chem_type: The chemical type of the residue as a descriptive string.
    :param polymer_chem_types: The set of polymer chemical types.
    :return: Whether the residue is polymeric.
    """
    return any(chem_type in res_chem_type.lower() for chem_type in polymer_chem_types)


@typecheck
def is_water(res_name: str, water_res_names: Set[str] = {"HOH", "WAT"}) -> bool:
    """Check if a residue is a water residue using its residue name string.

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
    """Get a Biopython `Residue` or `DisorderedResidue` object by its residue name-residue index
    composite ID.

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
    """Perform a rotation using a rotation matrix.

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
    """Deeply merge two dictionaries, merging values where possible.

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


@typecheck
def coerce_to_float(obj: Any) -> float | None:
    """Coerce an object to a float, returning `None` if the object is not coercible.

    :param obj: The object to coerce to a float.
    :return: The object coerced to a float if possible, otherwise `None`.
    """
    try:
        if isinstance(obj, (int, float, str)):
            return float(obj)
        elif isinstance(obj, list):
            return float(obj[0])
        else:
            return None
    except (ValueError, TypeError):
        return None


@typecheck
def extract_mmcif_metadata_field(
    mmcif_object: Any,
    metadata_field: MMCIF_METADATA_FIELD,
    min_resolution: float = 0.0,
    max_resolution: float = 1000.0,
) -> str | float | None:
    """Extract a metadata field from an mmCIF object. If the field is not found, return `None`.

    :param mmcif_object: The mmCIF object to extract the metadata field from.
    :param metadata_field: The metadata field to extract.
    :return: The extracted metadata field.
    """
    # Extract structure method
    if metadata_field == "structure_method" and "_exptl.method" in mmcif_object.raw_string:
        return mmcif_object.raw_string["_exptl.method"]

    # Extract release date
    if (
        metadata_field == "release_date"
        and "_pdbx_audit_revision_history.revision_date" in mmcif_object.raw_string
    ):
        # Return the earliest release date
        return min(mmcif_object.raw_string["_pdbx_audit_revision_history.revision_date"])

    # Extract resolution
    if metadata_field == "resolution" and "_refine.ls_d_res_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_refine.ls_d_res_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    elif (
        metadata_field == "resolution"
        and "_em_3d_reconstruction.resolution" in mmcif_object.raw_string
    ):
        resolution = coerce_to_float(mmcif_object.raw_string["_em_3d_reconstruction.resolution"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    elif metadata_field == "resolution" and "_reflns.d_resolution_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_reflns.d_resolution_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution


@typecheck
def make_one_hot(x: Tensor, num_classes: int) -> Tensor:
    """Convert a tensor of indices to a one-hot tensor.

    :param x: A tensor of indices.
    :param num_classes: The number of classes.
    :return: A one-hot tensor.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


@typecheck
def make_one_hot_np(x: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an array of indices to a one-hot encoded array.

    :param x: A NumPy array of indices.
    :param num_classes: The number of classes.
    :return: A one-hot encoded NumPy array.
    """
    x_one_hot = np.zeros((*x.shape, num_classes), dtype=np.int64)
    np.put_along_axis(x_one_hot, np.expand_dims(x, axis=-1), 1, axis=-1)
    return x_one_hot


@typecheck
def get_sorted_tuple_indices(
    tuples_list: List[Tuple[str, Any]], order_list: List[str]
) -> List[int]:
    """Get the indices of the tuples in the order specified by the order_list.

    :param tuples_list: A list of tuples containing a string and a value.
    :param order_list: A list of strings specifying the order of the tuples.
    :return: A list of indices of the tuples in the order specified by the order list.
    """
    # Create a mapping from the string values to their indices
    index_map = {value: index for index, (value, _) in enumerate(tuples_list)}

    # Generate the indices in the order specified by the order_list
    sorted_indices = [index_map[value] for value in order_list]

    return sorted_indices


@typecheck
def load_tsv_to_dict(filepath):
    """Load a two-column TSV file into a dictionary.

    :param filepath: The path to the TSV file.
    :return: A dictionary containing the TSV data.
    """
    result = {}
    with open(filepath, mode="r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            result[row[0]] = row[1]
    return result


@typecheck
def join(arr: Iterable[Any], delimiter: str = "") -> str:
    """Join the elements of an iterable into a string using a delimiter.

    :param arr: The iterable to join.
    :param delimiter: The delimiter to use.
    :return: The joined string.
    """
    # Re-do an ugly part of python
    return delimiter.join(arr)
