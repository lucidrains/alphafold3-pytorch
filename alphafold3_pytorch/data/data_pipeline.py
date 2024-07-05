"""General-purpose data pipeline."""

from typing import MutableMapping

import numpy as np

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object
from alphafold3_pytorch.data import mmcif_parsing

FeatureDict = MutableMapping[str, np.ndarray]


def make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=object)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=object)
    return features


def make_mmcif_features(mmcif_object: mmcif_parsing.MmcifObject) -> FeatureDict:
    """Make features from an mmCIF object."""
    input_sequence = "".join(mmcif_object.chain_to_seqres[chain_id] for chain_id in mmcif_object.chain_to_seqres)
    description = mmcif_object.file_id
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    biomol = _from_mmcif_object(mmcif_object)

    # TODO: Expand the first bioassembly/model sequence and structure, to obtain a biologically relevant complex (AF3 Supplement, Section 2.1).
    # Reference: https://github.com/biotite-dev/biotite/blob/1045f43f80c77a0dc00865e924442385ce8f83ab/src/biotite/structure/io/pdbx/convert.py#L1441

    mmcif_feats["all_atom_positions"] = biomol.atom_positions
    mmcif_feats["all_atom_mask"] = biomol.atom_mask
    mmcif_feats["b_factors"] = biomol.b_factors
    mmcif_feats["chain_index"] = biomol.chain_index
    mmcif_feats["chemid"] = biomol.chemid
    mmcif_feats["chemtype"] = biomol.chemtype
    mmcif_feats["residue_index"] = biomol.residue_index
    mmcif_feats["restype"] = biomol.restype

    mmcif_feats["resolution"] = np.array([mmcif_object.header["resolution"]], dtype=np.float32)

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    )

    mmcif_feats["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_feats


if __name__ == "__main__":
    mmcif_object = mmcif_parsing.parse_mmcif_object(
        # Load an example mmCIF file that includes
        # protein, nucleic acid, and ligand residues.
        filepath="data/pdb_data/mmcifs/16/316d.cif",
        file_id="316d",
    )
    mmcif_feats = make_mmcif_features(mmcif_object)
