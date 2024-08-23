"""MSA loading functions used in AlphaFold2."""

# From: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/data/parsers.py#L157

import dataclasses
import random
import re
import string
from typing import Literal, Optional, Sequence, Tuple

from alphafold3_pytorch.tensor_typing import typecheck

DeletionMatrix = Sequence[Sequence[int]]

# Constants

MSA_TYPE = Literal["protein", "dna", "rna"]

# Utilities for extracting identifiers from MSA sequence descriptions.


# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE,
)


@dataclasses.dataclass(frozen=True)
class Identifiers:
    species_id: str = ""


@typecheck
def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
    """Gets species from an msa sequence identifier.

    The sequence identifier has the format specified by
    _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
    An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

    Args:
      msa_sequence_identifier: a sequence identifier.

    Returns:
      An `Identifiers` instance with species_id. These
      can be empty in the case where no identifier was found.
    """
    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifiers(species_id=matches.group("SpeciesIdentifier"))
    return Identifiers()


@typecheck
def _extract_sequence_identifier(description: str) -> Optional[str]:
    """Extracts sequence identifier from description.

    Returns None if no match.
    """
    split_description = description.split()
    if split_description:
        return split_description[0].partition("/")[0]
    else:
        return None


@typecheck
def get_identifiers(description: str) -> Identifiers:
    """Computes extra MSA features from the description."""
    sequence_identifier = _extract_sequence_identifier(description)
    if sequence_identifier is None:
        return Identifiers()
    else:
        return _parse_sequence_identifier(sequence_identifier)


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""

    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]
    msa_type: MSA_TYPE

    def __post_init__(self):
        """Checks that all fields have the same length."""
        if not (len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got {len(self.sequences)} sequences, "
                f"{len(self.deletion_matrix)} rows in the deletion matrix, and "
                f"{len(self.descriptions)} descriptions."
            )

    def __len__(self):
        """Returns the number of sequences in the MSA."""
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        """Truncates the MSA to the first `max_seqs` sequences."""
        max_seqs = min(len(self.sequences), max_seqs)
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
            msa_type=self.msa_type,
        )

    def random_truncate(self, max_seqs: int):
        """Truncates the MSA to a random range of `max_seqs` sequences."""
        max_seqs = min(len(self.sequences), max_seqs)
        start = random.randint(0, len(self.sequences) - max_seqs)  # nosec
        return Msa(
            sequences=self.sequences[start : start + max_seqs],
            deletion_matrix=self.deletion_matrix[start : start + max_seqs],
            descriptions=self.descriptions[start : start + max_seqs],
            msa_type=self.msa_type,
        )


@typecheck
def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


@typecheck
def parse_a3m(a3m_string: str, msa_type: MSA_TYPE) -> Msa:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
      a3m_string: The string contents of a a3m file. The first sequence in the
        file should be the query sequence.
      msa_type: The type of the sequences in the MSA. This can be 'protein',
        'dna', or 'rna'.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
        * A list of descriptions, one per sequence, from the a3m file.
        * The type of the sequences in the MSA.
    """

    sequences, descriptions = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return Msa(
        sequences=aligned_sequences,
        deletion_matrix=deletion_matrix,
        descriptions=descriptions,
        msa_type=msa_type,
    )
