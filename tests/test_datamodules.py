"""This file prepares unit tests for datamodules."""

import os

import pytest
import torch

from alphafold3_pytorch.data.atom_datamodule import AtomDataModule

os.environ["TYPECHECK"] = "True"


@pytest.mark.parametrize("batch_size", [32, 128])
def test_atom_datamodule(batch_size: int) -> None:
    """Tests `AtomDataModule` to verify that the necessary attributes were
    created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = AtomDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = (
        len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    )  # 2 + 2 + 2 for this example
    assert num_datapoints == 6

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
