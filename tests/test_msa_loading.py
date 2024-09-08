import os

import pytest

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.trainer import pdb_inputs_to_batched_atom_input
from alphafold3_pytorch.utils.utils import exists


def test_msa_loading():
    """Test an MSA-featurized PDBDataset constructed using a WeightedPDBSampler."""
    data_test = os.path.join("data", "test")
    data_test_mmcif_dir = os.path.join(data_test, "mmcif")
    data_test_clusterings_dir = os.path.join(data_test, "data_caches", "clusterings")
    data_test_msa_dir = os.path.join(data_test, "data_caches", "msa", "msas")

    if not os.path.exists(data_test_mmcif_dir):
        pytest.skip(f"The directory `{data_test_mmcif_dir}` is not populated yet.")

    interface_mapping_path = os.path.join(data_test_clusterings_dir, "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        os.path.join(data_test_clusterings_dir, "ligand_chain_cluster_mapping.csv"),
        os.path.join(
            data_test_clusterings_dir,
            "nucleic_acid_chain_cluster_mapping.csv",
        ),
        os.path.join(data_test_clusterings_dir, "peptide_chain_cluster_mapping.csv"),
        os.path.join(data_test_clusterings_dir, "protein_chain_cluster_mapping.csv"),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    pdb_input = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        crop_size=128,
        msa_dir=data_test_msa_dir,
        training=False,
    )

    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)
    assert exists(batched_atom_input)
