import pytest
from alphafold3_pytorch.data.weighted_pdb_dataset import WeightedSamplerPDB

@pytest.fixture
def dataset():
    interface_mapping_path = "data/pdb_data/data_caches/clusterings/interface_cluster_mapping.csv"
    chain_mapping_paths = [
        "data/pdb_data/data_caches/clusterings/ligand_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/nucleic_acid_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/peptide_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/protein_chain_cluster_mapping.csv",
    ]
    return WeightedSamplerPDB(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

def test_sample(dataset):
    assert len(dataset.sample(64)) == 64, "The sampled batch size does not match the expected size."