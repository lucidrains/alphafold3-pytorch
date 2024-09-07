"""Deoxyribonucleic acid (DNA) constants used in AlphaFold."""

from beartype.typing import Final

import numpy as np

from alphafold3_pytorch.common import amino_acid_constants, rna_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# From: https://files.rcsb.org/ligands/view/DA.cif - https://files.rcsb.org/ligands/view/DT.cif
# Derived via: `list(dict.fromkeys([name for atom_names in dna_constants.restype_name_to_compact_atom_names.values() for name in atom_names if name]))`
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
    "C7",
    "ATM",  # NOTE: This represents a catch-all atom type for non-standard or modified residues.
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
element_types = [atom_type if atom_type == "ATM" else atom_type[0] for atom_type in atom_types]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28 + 19 null types := 47.
res_rep_atom_index = 11  # The index of the atom used to represent the center of the residue.


# This is the standard residue order when coding DNA type as a number.
# Reproduce it by taking 3-letter DNA codes and sorting them alphabetically.
restypes = ["A", "C", "G", "T"]
min_restype_num = (len(amino_acid_constants.restypes) + 1) + (
    len(rna_constants.restypes) + 1
)  # := 26.
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}
restype_num = min_restype_num + len(restypes)  # := 26 + 4 := 30.


restype_1to3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
    "X": "DN",
}

MSA_CHAR_TO_ID = {
    "A": 26,
    "C": 27,
    "G": 28,
    "T": 29,
    "X": 30,
    "-": 31,
}

BIOMOLECULE_CHAIN: Final[str] = "polydeoxyribonucleotide"
POLYMER_CHAIN: Final[str] = "polymer"


# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.nucleic_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name.
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define residue metadata for all unknown DNA residues.
unk_restype = "DN"
unk_chemtype = "DNA linking"
unk_chemname = "UNKNOWN DNA RESIDUE"

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# This represents the residue chemical type (i.e., `chemtype`) index of DNA residues.
chemtype_num = rna_constants.chemtype_num + 1  # := 2.

# A compact atom encoding with 24 columns for DNA residues.
# From: https://files.rcsb.org/ligands/view/DA.cif - https://files.rcsb.org/ligands/view/DT.cif
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "DA": [
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
        "",
    ],
    "DC": [
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
        "",
    ],
    "DG": [
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
        "",
    ],
    "DT": [
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
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
        "",
        "",
        "",
    ],
    "DN": [
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
        "",
    ],
}

restype_atom47_to_compact_atom = np.zeros([5, 47], dtype=int)


def _make_constants():
    """Fill the array(s) above."""
    for restype, restype_letter in enumerate(restype_1to3.keys()):
        resname = restype_1to3[restype_letter]
        for atomname in restype_name_to_compact_atom_names[resname]:
            if not atomname:
                continue
            atomtype = atom_order[atomname]
            compact_atom_idx = restype_name_to_compact_atom_names[resname].index(atomname)
            restype_atom47_to_compact_atom[restype, atomtype] = compact_atom_idx


_make_constants()
