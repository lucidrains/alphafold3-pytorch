"""Ribonucleic acid (RNA) constants used in AlphaFold."""

from typing import Final

import numpy as np

from alphafold3_pytorch.common import amino_acid_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# From: https://files.rcsb.org/ligands/view/A.cif - https://files.rcsb.org/ligands/view/U.cif
# Derived via: `list(dict.fromkeys([name for atom_names in rna_constants.restype_name_to_compact_atom_names.values() for name in atom_names if name]))`
atom_types = [
    "OP3",
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "N1",
    "C2",
    "N3",
    "C4",
    "O2",
    "N4",
    "O6",
    "N2",
    "O4",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",  # 19 null types.
]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28 + 19 null types := 47.
res_rep_atom_index = 12  # The index of the atom used to represent the center of the residue.


# This is the standard residue order when coding RNA type as a number.
# Reproduce it by taking 3-letter RNA codes and sorting them alphabetically.
restypes = ["A", "C", "G", "U"]
min_restype_num = len(amino_acid_constants.restypes) + 1  # := 21.
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}
restype_num = min_restype_num + len(restypes)  # := 21 + 4 := 25.


restype_1to3 = {"A": "A", "C": "C", "G": "G", "U": "U", "X": "N"}

BIOMOLECULE_CHAIN: Final[str] = "polyribonucleotide"
POLYMER_CHAIN: Final[str] = "polymer"

# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.nucleic_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name.
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define residue metadata for all unknown RNA residues.
unk_restype = "N"
unk_chemtype = "RNA linking"
unk_chemname = "UNKNOWN RNA RESIDUE"

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# This represents the residue chemical type (i.e., `chemtype`) index of RNA residues.
chemtype_num = amino_acid_constants.chemtype_num + 1  # := 1.

# A compact atom encoding with 24 columns for RNA residues.
# From: https://files.rcsb.org/ligands/view/A.cif - https://files.rcsb.org/ligands/view/U.cif
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "A": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "",
    ],
    "C": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
        "",
        "",
        "",
    ],
    "G": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "U": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
        "",
        "",
        "",
    ],
    "N": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "",
    ],
}

restype_atom47_to_compact_atom = np.zeros([5, 47], dtype=int)


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
