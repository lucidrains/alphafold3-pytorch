"""Ligand constants used in AlphaFold."""

from typing import Final

from alphafold3_pytorch.common import amino_acid_constants, dna_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    # NOTE: Taken from: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L117C13-L126C18
    "AL",
    "AS",
    "AU",
    "B",
    "BE",
    "BR",
    "C",
    "CA",
    "CL",
    "CO",
    "CR",
    "CU",
    "F",
    "FE",
    "HG",
    "I",
    "IR",
    "K",
    "LI",
    "MG",
    "MN",
    "MO",
    "N",
    "NI",
    "O",
    "OS",
    "P",
    "PB",
    "PD",
    "PR",
    "PT",
    "RE",
    "RH",
    "RU",
    "S",
    "SB",
    "SE",
    "SI",
    "SN",
    "TB",
    "TE",
    "U",
    "W",
    "V",
    "Y",
    "ZN",
    "ATM",
]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 47.


# All ligand residues are mapped to the unknown amino acid type index (:= 20).
restypes = ["UNL"]
min_restype_num = len(amino_acid_constants.restypes)  # := 20.
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}
restype_num = len(amino_acid_constants.restypes)  # := 20.

BIOMOLECULE_CHAIN: Final[str] = "other"
POLYMER_CHAIN: Final[str] = "non-polymer"


# NB: restype_3to1 serves as a placeholder for mapping all
# ligand residues to the unknown amino acid type index (:= 20).
restype_3to1 = {}

# Define residue metadata for all unknown ligand residues.
unk_restype = "UNL"
unk_chemtype = "non-polymer"
unk_chemname = "UNKNOWN LIGAND RESIDUE"

# This represents the residue chemical type (i.e., `chemtype`) index of ligand residues.
chemtype_num = dna_constants.chemtype_num + 1  # := 3.