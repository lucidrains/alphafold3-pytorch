# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Python wrapper for Kalign."""
import os
import subprocess  # nosec
import tempfile
from loguru import logger
from beartype.typing import Mapping, Sequence, Tuple

from alphafold3_pytorch.data import msa_parsing
from alphafold3_pytorch.data.template_parsing import (
    TEMPLATE_TYPE,
    QueryToTemplateAlignError,
)
from alphafold3_pytorch.utils.utils import exists


def _to_a3m(sequences: Sequence[str]) -> str:
    """Converts sequences to an a3m file."""
    names = ["sequence %d" % i for i in range(1, len(sequences) + 1)]
    a3m = []
    for sequence, name in zip(sequences, names):
        a3m.append(">" + name + "\n")
        a3m.append(sequence + "\n")
    return "".join(a3m)


def _realign_pdb_template_to_query(
    query_sequence: str,
    template_sequence: str,
    old_mapping: Mapping[int, int],
    kalign_binary_path: str | None,
    template_type: TEMPLATE_TYPE,
    min_frac_matching: float = 0.1,
    verbose: bool = False,
) -> Tuple[str, Mapping[int, int]]:
    """Aligns template sequence to the query sequence.

    Adapted from:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/templates.py#L16

    :param query_sequence: The actual query sequence extracted from a query
        Biomolecule object.
    :param template_sequence: The actual template sequence extracted from a template
        Biomolecule object.
    :param old_mapping: A mapping from the query sequence to the template sequence.
        This mapping will be used to compute the new mapping from the actual query
        sequence to the actual template sequence by aligning the `query_sequence`
        and the `template_sequence` strings.
    :param kalign_binary_path: The path to a kalign executable.
    :param template_type: The type of the template (e.g. "protein", "rna", "dna").
    :param min_frac_matching: The minimum fraction of matching residues between the
        actual template sequence and the actual query sequence. If the fraction is
        less than this value, a QueryToTemplateAlignError will be raised.
    :param verbose: Whether to log verbose output.

    :return: A tuple (new_template_sequence, new_query_to_template_mapping) where:
        * new_template_sequence is the actual template sequence that was given.
        * new_query_to_template_mapping is the new mapping from the query to the
            actual template sequence that was given.

    :raise: QueryToTemplateAlignError:
        * If there was an error thrown by the alignment tool.
        * Or if the actual template sequence differs (by default) by more than 90%
            from the `query_sequence` string.
    """
    assert exists(kalign_binary_path) and os.path.exists(
        kalign_binary_path
    ), f"Kalign binary not found at {kalign_binary_path}"
    aligner = Kalign(binary_path=kalign_binary_path)

    try:
        parsed_a3m = msa_parsing.parse_a3m(
            aligner.align([query_sequence, template_sequence]),
            template_type,
        )
        old_aligned_template, new_aligned_template = parsed_a3m.sequences
    except Exception as e:
        raise QueryToTemplateAlignError(
            "Could not align old template %s to template %s. Error: %s"
            % (
                query_sequence,
                template_sequence,
                str(e),
            )
        )

    if verbose:
        logger.info(
            "Old aligned template: %s\nNew aligned template: %s",
            old_aligned_template,
            new_aligned_template,
        )

    old_to_new_template_mapping = {}
    old_template_index = -1
    new_template_index = -1
    num_same = 0
    for old_template_res, new_template_res in zip(old_aligned_template, new_aligned_template):
        if old_template_res != "-":
            old_template_index += 1
        if new_template_res != "-":
            new_template_index += 1
        if old_template_res != "-" and new_template_res != "-":
            old_to_new_template_mapping[old_template_index] = new_template_index
            if old_template_res == new_template_res:
                num_same += 1

    # Require at least (by default) 10 % sequence identity w.r.t. to the shorter of the sequences.
    frac_matching = float(num_same) / min(len(query_sequence), len(template_sequence))
    if frac_matching < min_frac_matching:
        raise QueryToTemplateAlignError(
            "Insufficient similarity of the sequence in the database: %s to the "
            "actual sequence in the mmCIF file: %s. We require at least "
            f"{min_frac_matching * 100} %% similarity w.r.t. to the shorter of the sequences. This is not a "
            "problem unless you think this is a template that should be included."
            % (
                query_sequence,
                template_sequence,
            )
        )

    new_query_to_template_mapping = {}
    for query_index, old_template_index in old_mapping.items():
        new_query_to_template_mapping[query_index] = old_to_new_template_mapping.get(
            old_template_index, -1
        )

    template_sequence = template_sequence.replace("-", "")

    return template_sequence, new_query_to_template_mapping


class Kalign:
    """Python wrapper of the Kalign binary."""

    def __init__(self, *, binary_path: str):
        """Initializes the Python Kalign wrapper.

        Args:
          binary_path: The path to the Kalign binary.

        Raises:
          RuntimeError: If Kalign binary not found within the path.
        """
        self.binary_path = binary_path

    def align(self, sequences: Sequence[str]) -> str:
        """Aligns the sequences and returns the alignment in A3M string.

        Args:
          sequences: A list of query sequence strings. The sequences have to be at
            least 6 residues long (Kalign requires this). Note that the order in
            which you give the sequences might alter the output slightly as
            different alignment tree might get constructed.

        Returns:
          A string with the alignment in a3m format.

        Raises:
          RuntimeError: If Kalign fails.
          ValueError: If any of the sequences is less than 6 residues long.
        """
        logger.info("Aligning %d sequences", len(sequences))

        for s in sequences:
            if len(s) < 6:
                raise ValueError(
                    "Kalign requires all sequences to be at least 6 "
                    "residues long. Got %s (%d residues)." % (s, len(s))
                )

        with tempfile.TemporaryDirectory() as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, "input.fasta")
            output_a3m_path = os.path.join(query_tmp_dir, "output.a3m")

            with open(input_fasta_path, "w") as f:
                f.write(_to_a3m(sequences))

            cmd = [
                self.binary_path,
                "-i",
                input_fasta_path,
                "-o",
                output_a3m_path,
                "-format",
                "fasta",
            ]

            logger.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )  # nosec

            stdout, stderr = process.communicate()
            retcode = process.wait()
            logger.info(
                "Kalign stdout:\n%s\n\nstderr:\n%s\n",
                stdout.decode("utf-8"),
                stderr.decode("utf-8"),
            )

            if retcode:
                raise RuntimeError(
                    "Kalign failed\nstdout:\n%s\n\nstderr:\n%s\n"
                    % (stdout.decode("utf-8"), stderr.decode("utf-8"))
                )

            with open(output_a3m_path) as f:
                a3m = f.read()

            return a3m
