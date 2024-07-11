"""An mmCIF file format parser."""

import dataclasses
import functools
import io
import itertools
import logging
from collections import defaultdict
from operator import itemgetter
from typing import Any, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from Bio import PDB
from Bio.Data import PDBData

from alphafold3_pytorch.utils.data_utils import is_polymer, is_water, matrix_rotate

# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

AtomFullId = Tuple[str, int, str, Tuple[str, int, str], Tuple[str, str]]
ResidueFullId = Tuple[str, int, str, Tuple[str, int, str]]
ChainFullId = Tuple[str, int, str]


@dataclasses.dataclass(frozen=True)
class Monomer:
    """Represents a monomer in a polymer chain."""

    id: str
    num: int


@dataclasses.dataclass(frozen=True)
class ChemComp:
    """Represents a chemical composition."""

    id: str
    formula: str
    formula_weight: str
    mon_nstd_flag: str
    name: str
    type: str


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    """Represents an atom site in an mmCIF file."""

    mmcif_entity_num: int
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


@dataclasses.dataclass(frozen=True)
class Bond:
    """Represents a structural connection between two atoms."""

    ptnr1_auth_seq_id: str
    ptnr1_auth_comp_id: str
    ptnr1_auth_asym_id: str
    ptnr1_label_atom_id: str
    pdbx_ptnr1_label_alt_id: str

    ptnr2_auth_seq_id: str
    ptnr2_auth_comp_id: str
    ptnr2_auth_asym_id: str
    ptnr2_label_atom_id: str
    pdbx_ptnr2_label_alt_id: str

    pdbx_leaving_atom_flag: str
    pdbx_dist_value: str
    pdbx_role: str
    conn_type_id: str


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    """Represents a residue position in a chain."""

    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    """Represents a residue at a given position in a chain."""

    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
        file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
            files being processed.
        header: Biopython header.
        structure: Biopython structure.
        chem_comp_details: Dict mapping chain_id to a list of (non-missing) ChemComp. E.g.
            {'A': [ChemComp, ChemComp, ...]}
        all_chem_comp_details: Dict mapping chain_id to a list of ChemComp. E.g.
            {'A': [ChemComp, ChemComp, ...]}
        chain_to_seqres: Dict mapping chain_id to 1 letter sequence. E.g.
            {'A': 'ABCDEFG'}
        seqres_to_structure: Dict; for each chain_id contains a mapping between
            SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                            1: ResidueAtPosition,
                                                            ...}}
        entity_to_chain: Dict mapping entity_id to a list of chain_ids. E.g.
            {1: ['A', 'B']}
        mmcif_to_author_chain: Dict mapping internal mmCIF chain ids to author chain ids. E.g.
            {'A': 'B', 'B', 'B'}
        bonds: List of Bond objects.
        raw_string: The raw string used to construct the MmcifObject.
        atoms_to_remove: Optional set of atoms to remove.
        residues_to_remove: Optional set of residues to remove.
        chains_to_remove: Optional set of chains to remove.
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chem_comp_details: Mapping[ChainId, Sequence[ChemComp]]
    all_chem_comp_details: Mapping[ChainId, Sequence[ChemComp]]
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    entity_to_chain: Mapping[int, Sequence[str]]
    mmcif_to_author_chain: Mapping[str, str]
    bonds: Sequence[Bond]
    raw_string: Any
    atoms_to_remove: Set[AtomFullId]
    residues_to_remove: Set[ResidueFullId]
    chains_to_remove: Set[ChainFullId]


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(prefix: str, parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    :param prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
    :param parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    :return: A list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        "mmCIF error: Not all loops are the same length: %s" % cols
    )

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(
    prefix: str,
    index: str,
    parsed_info: MmCIFDict,
) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    :param prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
    :param index: Which item of loop data should serve as the key.
    :param parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    :return: A dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def apply_transformations(
    biomol: Any,
    transformation_dict: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    operations: Sequence[Tuple[str, str]],
) -> Any:
    """
    Gets subassembly by applying the given operations to the input
    Biomolecule containing affected asym IDs.

    Adapted from: https://github.com/biotite-dev/biotite

    :param biomol: The Biomolecule to which to apply assembly transformations.
    :param transformation_dict: A dictionary, where the keys are operation IDs
        and the values are tuples of rotation matrix and translation vector.
    :param operations: A list of tuples, where each tuple contains operation IDs
        of a single step.
    :return: The subassembly Biomolecule.
    """
    # Additional first dimension for 'repeat_biomol()'
    assembly_coord = np.zeros((len(operations),) + biomol.atom_positions.shape)

    # Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = biomol.atom_positions.copy()
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            rotation_matrix, translation_vector = transformation_dict[op_step]
            # Rotate
            coord = matrix_rotate(coord, rotation_matrix)
            # Translate
            coord += translation_vector
        assembly_coord[i] = coord

    return biomol.repeat(assembly_coord)


def get_transformations(
    struct_oper: Mapping[str, str]
) -> Mapping[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Gets transformation operations in terms of rotation matrix and
    translation for each operation ID in `pdbx_struct_oper_list`.

    Adapted from: https://github.com/biotite-dev/biotite

    :param struct_oper: A dictionary representing each row in the
        `_pdbx_struct_oper_list` category of an mmCIF file.
    :return: A dictionary, where the keys are operation IDs and the values
        are tuples of rotation matrix and translation vector.
    """
    transformation_dict = {}
    for id in struct_oper:
        rotation_matrix = np.array(
            [
                [struct_oper[id][f"_pdbx_struct_oper_list.matrix[{i}][{j}]"] for j in (1, 2, 3)]
                for i in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        translation_vector = np.array(
            [struct_oper[id][f"_pdbx_struct_oper_list.vector[{i}]"] for i in (1, 2, 3)],
            dtype=np.float64,
        )
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict


def parse_operation_expression(expression: str) -> Sequence[Tuple[str, str]]:
    """
    Gets successive operation steps (IDs) for the given
    ``oper_expression``. Forms the cartesian product, if necessary.

    Adapted from: https://github.com/biotite-dev/biotite

    :param expression: The operation expression string.
    :return: A list of tuples, where each tuple contains operation IDs
        of a single step.
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    # example: '(X0)(1-10,21-25)' from PDB `1a34`
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for one_step_expr in expressions_per_step:
        one_step_op_ids = []
        for expr in one_step_expr.split(","):
            if "-" in expr:
                # Range of operation IDs, they must be integers
                first, last = expr.split("-")
                one_step_op_ids.extend([str(id) for id in range(int(first), int(last) + 1)])
            else:
                # Single operation ID
                one_step_op_ids.append(expr)
        operations.append(one_step_op_ids)

    # Cartesian product of operations
    return list(itertools.product(*operations))


@functools.lru_cache(16, typed=False)
def parse(
    *,
    file_id: str,
    mmcif_string: str,
    catch_all_errors: bool = True,
    auth_chains: bool = True,
    auth_residues: bool = True,
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    :param file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
    :param mmcif_string: Contents of an mmCIF file.
    :param catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.
    :param auth_chains: If True, use author-assigned chain ids. If False, use internal
        mmCIF chain ids.
    :param auth_residues: If True, use author-assigned residue numbers. If False, use
        internal mmCIF residue numbers.

    :return: A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True, auth_chains=auth_chains, auth_residues=auth_residues)
        with io.StringIO(mmcif_string) as handle:
            full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the complex chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_complex_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ""): "No complex chains found in this file."})
        mmcif_seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, (seq, _) in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate a mapping:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # by the authors / Biopython).
        mmcif_to_author_chain_id = {}
        atom_site_list = _get_atom_site_list(parsed_info)
        for atom in atom_site_list:
            # NOTE: This is a potential bottleneck, which may be addressed by
            # instead referencing pairs of internal mmCIF chain IDs and author
            # chain IDs on a residue or chain-wise level. However, it does not
            # seem that such information is always available in a given mmCIF file.
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue
            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

        # Determine each (author) complex chain's start number
        # according to the author-assigned numbering scheme.
        author_seq_start_num = defaultdict(lambda: float("inf"))
        for chain_id, (seq, _) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            author_seq_start_num[author_chain] = min(
                author_seq_start_num[author_chain],
                min([monomer.num for monomer in seq]),
            )

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_seq_to_structure_mappings (maps mmCIF idx into sequence to author ResidueAtPosition).
        # -mmcif_entity_to_author_chain_mappings (maps entity number to author chain IDs).
        mmcif_seq_to_structure_mappings = {}
        mmcif_entity_to_author_chain_mappings = defaultdict(dict)
        for atom in atom_site_list:
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if is_water(atom.residue_name):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                mmcif_current = mmcif_seq_to_structure_mappings.get(atom.mmcif_chain_id, {})
                if _is_set(atom.mmcif_seq_num):
                    seq_idx = int(atom.mmcif_seq_num) - mmcif_seq_start_num[atom.mmcif_chain_id]
                else:
                    seq_idx = int(atom.author_seq_num) - author_seq_start_num[atom.author_chain_id]
                mmcif_current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                mmcif_seq_to_structure_mappings[atom.mmcif_chain_id] = mmcif_current
                mmcif_entity_to_author_chain_mappings[atom.mmcif_entity_num][
                    atom.author_chain_id
                ] = True

        # Add missing residue information to mmcif_seq_to_structure_mappings.
        for chain_id, (seq_info, chem_comp_info) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            mmcif_current_mapping = mmcif_seq_to_structure_mappings[chain_id]
            for idx, monomer in enumerate(seq_info):
                seq_idx = idx + (mmcif_seq_start_num[chain_id] - 1)
                # NOTE: Water residues are often not labeled consecutively by authors
                # (e.g., see PDB 100d), so we avoid marking them as missing in this scenario.
                if seq_idx not in mmcif_current_mapping and not is_water(chem_comp_info[idx].id):
                    if not is_polymer(chem_comp_info[idx].type):
                        hetflag = "H_" + chem_comp_info[idx].id
                    else:
                        hetflag = " "
                    position = ResiduePosition(
                        chain_id=chain_id,
                        residue_number=seq_idx + 1,
                        insertion_code=" ",
                    )
                    mmcif_current_mapping[seq_idx] = ResidueAtPosition(
                        position=position,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=hetflag,
                    )
            mmcif_seq_to_structure_mappings[chain_id] = dict(
                sorted(mmcif_current_mapping.items(), key=lambda x: x[0])
            )

        # Extract all sequence and chemical component details, and
        # populate seq_to_structure_mappings using author chain IDs.
        author_chain_to_sequence = defaultdict(str)
        all_chem_comp_details = defaultdict(list)
        seq_to_structure_mappings = defaultdict(dict)
        for chain_id, (seq_info, chem_comp_info) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            mmcif_current_mapping = mmcif_seq_to_structure_mappings[chain_id]
            seq = []
            all_chem_comp_info = []
            for monomer_index, monomer in enumerate(seq_info):
                all_chem_comp_info.append(chem_comp_info[monomer_index])
                if "peptide" in chem_comp_info[monomer_index].type.lower():
                    code = PDBData.protein_letters_3to1.get(f"{monomer.id: <3}", "X")
                elif (
                    "dna" in chem_comp_info[monomer_index].type.lower()
                    or "rna" in chem_comp_info[monomer_index].type.lower()
                ):
                    code = PDBData.nucleic_letters_3to1.get(f"{monomer.id: <3}", "X")
                else:
                    code = "X"
                seq.append(code if len(code) == 1 else "X")
            author_chain_to_sequence[author_chain] += "".join(seq)
            all_chem_comp_details[author_chain].extend(all_chem_comp_info)
            if current_mapping:
                start_index = len(current_mapping)
                current_mapping.update(
                    {
                        start_index + value_index: value
                        for value_index, value in enumerate(mmcif_current_mapping.values())
                    }
                )
            else:
                current_mapping.update(mmcif_current_mapping)

        # NOTE: All three of the following variables need to be perfectly matching
        # in terms of sequence contents to guarantee correctness for downstream code.
        for author_chain in author_chain_to_sequence:
            # Ensure the `seq_to_structure_mappings` is zero-indexed and does not contain index gaps.
            seq_to_structure_mappings[author_chain] = {
                i: value
                for i, value in enumerate(seq_to_structure_mappings[author_chain].values())
            }
            assert (
                len(author_chain_to_sequence[author_chain])
                == len(all_chem_comp_details[author_chain])
                == len(seq_to_structure_mappings[author_chain])
            ), (
                f"In parse(), encountered a sequence length mismatch for chain {author_chain} of file {file_id} regarding `author_chain_to_sequence`, `all_chem_comp_details`, and `seq_to_structure_mapping`: "
                f"{len(author_chain_to_sequence[author_chain])} != "
                f"{len(all_chem_comp_details[author_chain])} != "
                f"{len(seq_to_structure_mappings[author_chain])}"
            )

        # Identify only chemical component details that are present in the structure.
        chem_comp_details = {
            chain_id: [
                chem_comp_info
                for res_index, chem_comp_info in enumerate(all_chem_comp_details[chain_id])
                if not seq_to_structure_mappings[chain_id][res_index].is_missing
            ]
            for chain_id in all_chem_comp_details
        }

        # Simplify entity-to-chain mapping.
        entity_to_chain = {
            entity_id: list(chains.keys())
            for entity_id, chains in mmcif_entity_to_author_chain_mappings.items()
        }

        # Identify all bonds.
        bonds = _get_bond_list(parsed_info)

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chem_comp_details=chem_comp_details,
            all_chem_comp_details=all_chem_comp_details,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            entity_to_chain=entity_to_chain,
            mmcif_to_author_chain=mmcif_to_author_chain_id,
            bonds=bonds,
            raw_string=parsed_info,
            atoms_to_remove=set(),
            residues_to_remove=set(),
            chains_to_remove=set(),
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info["_pdbx_audit_revision_history.revision_date"]
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if "_pdbx_audit_revision_history.revision_date" in parsed_info:
        header["release_date"] = get_release_date(parsed_info)
    else:
        logging.warning("Could not determine release_date for entry: %s", parsed_info["_entry.id"])

    header["resolution"] = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header["resolution"] = float(raw_resolution)
                break
            except ValueError:
                logging.debug("Invalid resolution format: %s", parsed_info[res_key])

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_entity_id"],
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"],
        )
    ]


def _get_bond_list(parsed_info: MmCIFDict) -> Sequence[Bond]:
    """Returns list of bonds present in the structure."""
    return [
        # Collect unique (partner) atom metadata required for each bond
        # per https://mmcif.wwpdb.org/docs/sw-examples/python/html/connections3.html.
        Bond(*conn)
        for conn in zip(  # pylint:disable=g-complex-comprehension
            # Partner 1
            parsed_info.get("_struct_conn.ptnr1_auth_seq_id", []),
            parsed_info.get("_struct_conn.ptnr1_auth_comp_id", []),
            parsed_info.get("_struct_conn.ptnr1_auth_asym_id", []),
            parsed_info.get("_struct_conn.ptnr1_label_atom_id", []),
            parsed_info.get("_struct_conn.pdbx_ptnr1_label_alt_id", []),
            # Partner 2
            parsed_info.get("_struct_conn.ptnr2_auth_seq_id", []),
            parsed_info.get("_struct_conn.ptnr2_auth_comp_id", []),
            parsed_info.get("_struct_conn.ptnr2_auth_asym_id", []),
            parsed_info.get("_struct_conn.ptnr2_label_atom_id", []),
            parsed_info.get("_struct_conn.pdbx_ptnr2_label_alt_id", []),
            # Connection metadata
            parsed_info.get("_struct_conn.pdbx_leaving_atom_flag", []),
            parsed_info.get("_struct_conn.pdbx_dist_value", []),
            parsed_info.get("_struct_conn.pdbx_role", []),
            parsed_info.get("_struct_conn.conn_type_id", []),
        )
        if len(conn[-1]) > 0
    ]


def _get_complex_chains(
    *, parsed_info: Mapping[str, Any]
) -> Mapping[ChainId, Tuple[Sequence[Monomer], Sequence[ChemComp]]]:
    """Extracts polymer information for complex chains.

    :param parsed_info: _mmcif_dict produced by the Biopython parser.

    :return: A dict mapping mmcif chain id to a tuple of a list of Monomers and a list of ChemComps.
    """
    # Get (non-)polymer information for each entity in the structure.
    poly_scheme = mmcif_loop_to_list("_pdbx_poly_seq_scheme.", parsed_info)
    branch_scheme = mmcif_loop_to_list("_pdbx_branch_scheme.", parsed_info)
    nonpoly_scheme = mmcif_loop_to_list("_pdbx_nonpoly_scheme.", parsed_info)
    # Get chemical compositions. Will allow us to identify which of these polymers
    # are protein, DNA, RNA, or ligand molecules.
    chem_comps = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    polymers = defaultdict(list)
    for scheme in poly_scheme:
        polymers[scheme["_pdbx_poly_seq_scheme.asym_id"]].append(
            Monomer(
                id=scheme["_pdbx_poly_seq_scheme.mon_id"],
                num=int(scheme["_pdbx_poly_seq_scheme.seq_id"]),
            )
        )

    non_polymers = defaultdict(list)
    for scheme in branch_scheme:
        non_polymers[scheme["_pdbx_branch_scheme.asym_id"]].append(
            Monomer(
                id=scheme["_pdbx_branch_scheme.pdb_mon_id"],
                num=int(scheme["_pdbx_branch_scheme.pdb_seq_num"]),
            )
        )
    for scheme in nonpoly_scheme:
        non_polymers[scheme["_pdbx_nonpoly_scheme.asym_id"]].append(
            Monomer(
                id=scheme["_pdbx_nonpoly_scheme.pdb_mon_id"],
                num=int(scheme["_pdbx_nonpoly_scheme.pdb_seq_num"]),
            )
        )

    # Identify and return all complex chains.
    valid_chains = defaultdict(lambda: ([], []))
    for chain_id, seq_info in polymers.items():
        valid_chains[chain_id][0].extend(seq_info)
        valid_chains[chain_id][1].extend(
            [
                ChemComp(
                    id=chem_comps[monomer.id]["_chem_comp.id"],
                    formula=chem_comps[monomer.id]["_chem_comp.formula"],
                    formula_weight=chem_comps[monomer.id]["_chem_comp.formula_weight"],
                    mon_nstd_flag=chem_comps[monomer.id]["_chem_comp.mon_nstd_flag"],
                    name=chem_comps[monomer.id]["_chem_comp.name"],
                    type=chem_comps[monomer.id]["_chem_comp.type"],
                )
                for monomer in seq_info
            ]
        )

    # Insert non-polymer chains in-place into the valid_chains dict.
    for chain_id, seq_info in non_polymers.items():
        valid_chain = {
            monomer.num: (monomer, chem_comp)
            for monomer, chem_comp in zip(valid_chains[chain_id][0], valid_chains[chain_id][1])
        }
        new_valid_chain = {
            monomer.num: (monomer, chem_comp)
            for monomer, chem_comp in zip(
                seq_info,
                [
                    ChemComp(
                        id=chem_comps[monomer.id]["_chem_comp.id"],
                        formula=chem_comps[monomer.id]["_chem_comp.formula"],
                        formula_weight=chem_comps[monomer.id]["_chem_comp.formula_weight"],
                        mon_nstd_flag=chem_comps[monomer.id]["_chem_comp.mon_nstd_flag"],
                        name=chem_comps[monomer.id]["_chem_comp.name"],
                        type=chem_comps[monomer.id]["_chem_comp.type"],
                    )
                    for monomer in seq_info
                ],
            )
        }
        merged_valid_chain = {**valid_chain, **new_valid_chain}
        valid_chains[chain_id] = (
            [chain[0] for chain in merged_valid_chain.values()],
            [chain[1] for chain in merged_valid_chain.values()],
        )

    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")


def parse_mmcif_object(
    filepath: str, file_id: str, auth_chains: bool = True, auth_residues: bool = True
) -> MmcifObject:
    """Parse an mmCIF file into an `MmcifObject` containing a BioPython `Structure` object as well as associated metadata."""
    with open(filepath, "r") as f:
        mmcif_string = f.read()

    parsing_result = parse(
        file_id=file_id,
        mmcif_string=mmcif_string,
        auth_chains=auth_chains,
        auth_residues=auth_residues,
    )

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with beforehand (e.g., at the alignment stage).
    if parsing_result.mmcif_object is None:
        raise list(parsing_result.errors.values())[0]

    return parsing_result.mmcif_object


def filter_mmcif(mmcif_object: MmcifObject) -> MmcifObject:
    """Filter an `MmcifObject` based on collected (atom/residue/chain) removal sets."""
    model = mmcif_object.structure

    # Filter out specified chains
    chains_to_remove = set()

    for chain in model:
        # Filter out specified residues
        residues_to_remove = set()
        assert len(chain) == len(mmcif_object.chem_comp_details[chain.id]), (
            f"Number of residues in chain {chain.id} does not match "
            f"number of chemical component details for this chain: {len(chain)} vs. "
            f"{len(mmcif_object.chem_comp_details[chain.id])}."
        )
        for res_index, residue in enumerate(chain):
            # Filter out specified atoms
            atoms_to_remove = set()
            for atom in residue:
                if atom.get_full_id() in mmcif_object.atoms_to_remove:
                    atoms_to_remove.add(atom)
            if len(atoms_to_remove) == len(residue):
                residues_to_remove.add((res_index, residue))
            for atom in atoms_to_remove:
                residue.detach_child(atom.id)
            if residue.get_full_id() in mmcif_object.residues_to_remove:
                residues_to_remove.add((res_index, residue))
        if len(residues_to_remove) == len(chain):
            chains_to_remove.add(chain)
        for res_index, residue in sorted(residues_to_remove, key=itemgetter(0), reverse=True):
            del mmcif_object.chem_comp_details[chain.id][res_index]
            chain.detach_child(residue.id)
        if chain.get_full_id() in mmcif_object.chains_to_remove:
            chains_to_remove.add(chain)

    for chain in chains_to_remove:
        model.detach_child(chain.id)
        mmcif_object.chem_comp_details.pop(chain.id)

    mmcif_object.atoms_to_remove.clear()
    mmcif_object.residues_to_remove.clear()
    mmcif_object.chains_to_remove.clear()

    return mmcif_object
