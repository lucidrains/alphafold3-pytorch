"""Deoxyribonucleic acid (DNA) constants used in AlphaFold."""

from typing import Final

from alphafold3_pytorch.common import amino_acid_constants, rna_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    # NOTE: Taken from: https://github.com/Profluent-Internships/MMDiff/blob/e21192bb8e815c765eaa18ee0f7bacdcc6af4044/src/data/components/pdb/nucleotide_constants.py#L670C1-L698C2
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    "O5'",
    "O4'",
    "O3'",
    "O2'",
    "P",
    "OP1",
    "OP2",
    "N1",
    "N2",
    "N3",
    "N4",
    "N6",
    "N7",
    "N9",
    "C2",
    "C4",
    "C5",
    "C6",
    "C8",
    "O2",
    "O4",
    "O6",
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
    "_",
    "_",  # 20 null types.
]
atom_types_set = set(atom_types)
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 27 + 20 null types := 47.


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