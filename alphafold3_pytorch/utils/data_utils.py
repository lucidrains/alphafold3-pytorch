from typing import Set

from alphafold3_pytorch.tensor_typing import ChainType, ResidueType, typecheck


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
