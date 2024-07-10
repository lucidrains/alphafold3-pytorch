"""This file prepares unit tests for data parsing (e.g., mmCIF file I/O)."""

import glob
import os
import random

import pytest

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object
from alphafold3_pytorch.data import mmcif_parsing

os.environ["TYPECHECK"] = "True"

# constants

ERRONEOUS_PDB_IDS = [
    "3tob"  # NOTE: At residue index 97, ALY and LYS are assigned the same residue ID of 118 by the authors.
]


@pytest.mark.parametrize(
    "mmcif_dir",
    [
        os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        os.path.join("data", "pdb_data", "mmcifs"),
    ],
)
@pytest.mark.parametrize(
    "complex_id", ["100d", "1k7a", "384d", "4xij", "6adq", "7a4d", "7akd", "8a3j"]
)
def test_mmcif_object_parsing(mmcif_dir: str, complex_id: str) -> None:
    """Tests parsing and `Biomolecule` object creation for mmCIF files.

    :param mmcif_dir: A directory containing PDB mmCIF files.
    :param complex_id: The PDB ID of the complex to be tested.
    """
    complex_filepath = os.path.join(mmcif_dir, complex_id[1:3], f"{complex_id}.cif")
    complex_filepaths = glob.glob(f"{complex_filepath[:-4]}*.cif")

    if not complex_filepaths:
        pytest.skip(f"File '{complex_filepath}' does not exist.")

    complex_filepath = random.choice(complex_filepaths)
    with open(complex_filepath, "r") as f:
        mmcif_string = f.read()

    parsing_result = mmcif_parsing.parse(
        file_id=complex_id,
        mmcif_string=mmcif_string,
        auth_chains=True,
        auth_residues=True,
    )

    if parsing_result.mmcif_object is None:
        print(f"Failed to parse file '{complex_filepath}'.")
        raise list(parsing_result.errors.values())[0]
    else:
        try:
            biomol = _from_mmcif_object(parsing_result.mmcif_object)
        except Exception as e:
            if "mmCIF contains an insertion code" in str(e):
                pytest.skip(f"File '{complex_filepath}' contains an insertion code.")
            else:
                raise e
        assert (
            len(biomol.atom_positions) > 0
        ), f"Failed to parse file '{complex_filepath}' into a `Biomolecule` object."


@pytest.mark.parametrize(
    "mmcif_dir",
    [
        os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        os.path.join("data", "pdb_data", "mmcifs"),
    ],
)
@pytest.mark.parametrize("num_random_complexes_to_parse", [500])
@pytest.mark.parametrize("random_seed", [1])
def test_random_mmcif_objects_parsing(
    mmcif_dir: str,
    num_random_complexes_to_parse: int,
    random_seed: int,
) -> None:
    """Tests parsing and `Biomolecule` object creation for a random batch
    of mmCIF files.

    :param mmcif_dir: A directory containing PDB mmCIF files.
    :param num_random_complexes_to_parse: The number of random complexes to parse.
    :param random_seed: The random seed for reproducibility.
    """
    random.seed(random_seed)

    if not os.path.exists(mmcif_dir):
        pytest.skip(f"Directory '{mmcif_dir}' does not exist.")

    parsing_errors = []
    failed_complex_indices = []
    failed_random_complex_filepaths = []
    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
        and os.listdir(os.path.join(mmcif_dir, subdir))
    ]
    for complex_index in range(num_random_complexes_to_parse):
        random_mmcif_subdir = random.choice(mmcif_subdirs)
        mmcif_subdir_files = [
            os.path.join(random_mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(random_mmcif_subdir)
            if os.path.isfile(os.path.join(random_mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        random_complex_filepath = random.choice(mmcif_subdir_files)
        complex_id = os.path.splitext(os.path.basename(random_complex_filepath))[0]

        if not os.path.exists(random_complex_filepath):
            print(f"File '{random_complex_filepath}' does not exist.")
            continue

        if any(
            id in os.path.basename(random_complex_filepath)[:4].lower() for id in ERRONEOUS_PDB_IDS
        ):
            continue

        with open(random_complex_filepath, "r") as f:
            mmcif_string = f.read()

        parsing_result = mmcif_parsing.parse(
            file_id=complex_id,
            mmcif_string=mmcif_string,
            auth_chains=True,
            auth_residues=True,
        )

        if parsing_result.mmcif_object is None:
            parsing_errors.append(list(parsing_result.errors.values())[0])
            failed_complex_indices.append(complex_index)
            failed_random_complex_filepaths.append(random_complex_filepath)
        else:
            try:
                biomol = _from_mmcif_object(parsing_result.mmcif_object)
            except Exception as e:
                if "mmCIF contains an insertion code" in str(e):
                    continue
                else:
                    parsing_errors.append(e)
                    failed_complex_indices.append(complex_index)
                    failed_random_complex_filepaths.append(random_complex_filepath)
                    continue
            if len(biomol.atom_positions) == 0:
                parsing_errors.append(
                    AssertionError(
                        f"Failed to parse file '{random_complex_filepath}' into a `Biomolecule` object."
                    )
                )
                failed_complex_indices.append(complex_index)
                failed_random_complex_filepaths.append(random_complex_filepath)

    if parsing_errors:
        print(
            f"Failed to parse {len(parsing_errors)} files at indices {failed_complex_indices}: '{failed_random_complex_filepaths}'."
        )
        for error in parsing_errors:
            print(error)
        raise error
