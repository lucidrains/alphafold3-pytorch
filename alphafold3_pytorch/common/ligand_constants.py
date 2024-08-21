"""Ligand constants used in AlphaFold."""

from typing import Final

import numpy as np

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
element_types = [
    # NOTE: Taken from: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L117C13-L126C18
    "Al",
    "As",
    "Au",
    "B",
    "Be",
    "Br",
    "C",
    "Ca",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe",
    "Hg",
    "I",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Ni",
    "O",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Sn",
    "Tb",
    "Te",
    "U",
    "W",
    "V",
    "Y",
    "Zn",
    "ATM",
]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 47.
res_rep_atom_index = (
    len(atom_types) - 1
)  # := 46  # The index of the atom used to represent the center of a ligand pseudoresidue.


# All ligand residues are mapped to the unknown amino acid type index (:= 20).
restypes = ["X"]
min_restype_num = len(amino_acid_constants.restypes)  # := 20.
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}
restype_num = len(amino_acid_constants.restypes)  # := 20.


restype_1to3 = {"X": "UNL"}

MSA_CHAR_TO_ID = {
    "X": 20,
    "-": 31,
}

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

# A compact atom encoding with 47 columns for ligand residues.
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "UNL": atom_types,
}

restype_atom47_to_compact_atom = np.zeros([1, 47], dtype=int)


def _make_constants():
    """Fill the array(s) above."""
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname in restype_name_to_compact_atom_names[resname]:
            if not atomname:
                continue
            atomtype = atom_order[atomname]
            compact_atom_idx = restype_name_to_compact_atom_names[resname].index(atomname)
            restype_atom47_to_compact_atom[restype, atomtype] = compact_atom_idx


_make_constants()
