import pytest
from alphafold3_pytorch.data.weighted_pdb_dataset import WeightedSamplerPDB

@pytest.fixture
def dataset():
    interface_mapping_path = "data/test/interface_cluster_mapping.csv"
    chain_mapping_paths = [
        "data/test/ligand_chain_cluster_mapping.csv",
        "data/test/nucleic_acid_chain_cluster_mapping.csv",
        "data/test/peptide_chain_cluster_mapping.csv",
        "data/test/protein_chain_cluster_mapping.csv",
    ]
    return WeightedSamplerPDB(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=4,
    )

def test_sample(dataset):
    assert len(dataset.sample(4)) == 4, "The sampled batch size does not match the expected size."