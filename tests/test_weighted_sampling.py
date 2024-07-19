import os

import pytest

from torch.utils.data import Sampler

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler


@pytest.fixture
def sampler():
    """Return a `WeightedPDBSampler` object."""
    interface_mapping_path = os.path.join("data", "test", "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        os.path.join("data", "test", "ligand_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "nucleic_acid_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "peptide_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "protein_chain_cluster_mapping.csv"),
    ]
    return WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=4,
    )


def test_sample(sampler: Sampler):
    """Test the sampling method of the `WeightedSamplerPDB` class."""
    assert len(sampler.sample(4)) == 4, "The sampled batch size does not match the expected size."


def test_cluster_based_sample(sampler: Sampler):
    """Test the cluster-based sampling method of the `WeightedSamplerPDB` class."""
    assert (
        len(sampler.cluster_based_sample(4)) == 4
    ), "The cluster-based sampled batch size does not match the expected size."
