import os

import pytest

from alphafold3_pytorch.inputs import PDBDataset
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.trainer import pdb_inputs_to_batched_atom_input
from alphafold3_pytorch.utils.utils import exists


def test_template_loading():
    """Test a template-featurized PDBDataset constructed using a WeightedPDBSampler."""
    data_test = os.path.join("data", "test")
    if not os.path.exists(os.path.join("data", "test", "mmcif")):
        pytest.skip("The directory `data/test/mmcif` is not populated yet.")

    interface_mapping_path = os.path.join(data_test, "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        os.path.join(data_test, "ligand_chain_cluster_mapping.csv"),
        os.path.join(
            data_test,
            "nucleic_acid_chain_cluster_mapping.csv",
        ),
        os.path.join(data_test, "peptide_chain_cluster_mapping.csv"),
        os.path.join(data_test, "protein_chain_cluster_mapping.csv"),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    pdb_input = PDBDataset(
        folder=os.path.join("data", "test", "mmcif"),
        sampler=sampler,
        sample_type="default",
        crop_size=128,
        templates_dir=os.path.join("data", "test", "template"),
        training=False,
    )

    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)
    assert exists(batched_atom_input)
