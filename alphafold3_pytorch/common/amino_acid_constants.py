"""Amino acid constants used in AlphaFold."""

from typing import Final

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    # NOTE: Taken from: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/common/residue_constants.py#L492C1-L497C2
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",  # 10 null types.
]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37 + 10 null types := 47.


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
min_restype_num = 0  # := 0.
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}
restype_num = min_restype_num + len(restypes)  # := 0 + 20 := 20.


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

BIOMOLECULE_CHAIN: Final[str] = "polypeptide(L)"
POLYMER_CHAIN: Final[str] = "polymer"


# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.protein_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name
# (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define residue metadata for all unknown amino acid residues.
unk_restype = "UNK"
unk_chemtype = "peptide linking"
unk_chemname = "UNKNOWN AMINO ACID RESIDUE"

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# This represents the residue chemical type (i.e., `chemtype`) index of amino acid residues.
chemtype_num = 0