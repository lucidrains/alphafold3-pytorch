import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from torch.utils.data import Sampler

from alphafold3_pytorch.data.weighted_pdb_sampler import (
    WeightedPDBSampler
)

from alphafold3_pytorch import (
    create_trainer_from_yaml
)

TEST_FOLDER = Path('./data/test')

INTERFACE_MAPPING_PATH = str(TEST_FOLDER / "interface_cluster_mapping.csv")

CHAIN_MAPPING_PATHS = [
    str(TEST_FOLDER / "ligand_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "nucleic_acid_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "peptide_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "protein_chain_cluster_mapping.csv"),
]

@pytest.fixture
def sampler():
    """Return a `WeightedPDBSampler` object."""
    return WeightedPDBSampler(
        chain_mapping_paths=CHAIN_MAPPING_PATHS,
        interface_mapping_path=INTERFACE_MAPPING_PATH,
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

# testing end to end with weighted pdb sampler

# testing trainer with pdb inputs

@pytest.fixture()
def populate_mock_pdb_and_remove_test_folders():
    proj_root = Path('.')
    working_cif_file = proj_root / 'data' / 'test' / '7a4d-assembly1.cif'

    pytest_root_folder = Path('./test-folder')
    data_folder = pytest_root_folder / 'data'
    train_folder = data_folder / 'train'
    train_folder.mkdir(exist_ok = True, parents = True)

    pdb_ids = []

    for path in [*CHAIN_MAPPING_PATHS, INTERFACE_MAPPING_PATH]:
        dataset = pl.read_csv(path)
        pdb_ids.extend(list(dataset.get_column('pdb_id')))

    for pdb_id in {*pdb_ids}:
        shutil.copy2(str(working_cif_file), str(train_folder / f'{pdb_id}.cif'))

    yield

    shutil.rmtree('./test-folder')

def test_weighted_sampling_from_trainer_config(populate_mock_pdb_and_remove_test_folders):

    trainer = create_trainer_from_yaml('./tests/configs/trainer_with_pdb_dataset_and_weighted_sampling.yaml')
    trainer()
