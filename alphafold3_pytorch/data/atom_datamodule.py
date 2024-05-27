from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from alphafold3_pytorch.models.alphafold3_module import AlphaFold3Input


class AtomDataset(Dataset):
    """
    A dummy dataset for atomic data.

    :param seq_len: The length of the protein sequence.
    :param atoms_per_window: The number of atoms per window.
    :param num_examples: The number of examples in the dataset.
    """

    def __init__(
        self,
        seq_len=16,
        atoms_per_window=27,
        num_examples=2,
    ):
        self.seq_len = seq_len
        self.atom_seq_len = seq_len * atoms_per_window
        self.num_examples = num_examples

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_examples

    def __getitem__(self, idx) -> AlphaFold3Input:
        """
        Return a random `AlphaFold3Input` sample from the dataset.

        :param idx: The index of the sample.
        :return: A random `AlphaFold3Input` sample from the dataset.
        """
        seq_len = self.seq_len
        atom_seq_len = self.atom_seq_len

        atom_inputs = torch.randn(atom_seq_len, 77)
        residue_atom_lens = torch.randint(0, 27, (seq_len,))
        atompair_feats = torch.randn(atom_seq_len, atom_seq_len, 16)
        additional_residue_feats = torch.randn(seq_len, 10)

        templates = torch.randn(2, seq_len, seq_len, 44)
        template_mask = torch.ones((2,)).bool()

        msa = torch.randn(7, seq_len, 64)
        msa_mask = torch.ones((7,)).bool()

        # required for training, but omitted on inference

        atom_pos = torch.randn(atom_seq_len, 3)
        residue_atom_indices = torch.randint(0, 27, (seq_len,))

        distance_labels = torch.randint(0, 37, (seq_len, seq_len))
        pae_labels = torch.randint(0, 64, (seq_len, seq_len))
        pde_labels = torch.randint(0, 64, (seq_len, seq_len))
        plddt_labels = torch.randint(0, 50, (seq_len,))
        resolved_labels = torch.randint(0, 2, (seq_len,))

        return AlphaFold3Input(
            atom_inputs=atom_inputs,
            residue_atom_lens=residue_atom_lens,
            atompair_feats=atompair_feats,
            additional_residue_feats=additional_residue_feats,
            templates=templates,
            template_mask=template_mask,
            msa=msa,
            msa_mask=msa_mask,
            atom_pos=atom_pos,
            residue_atom_indices=residue_atom_indices,
            distance_labels=distance_labels,
            pae_labels=pae_labels,
            pde_labels=pde_labels,
            plddt_labels=plddt_labels,
            resolved_labels=resolved_labels,
        )


class AtomDataModule(LightningDataModule):
    """`LightningDataModule` for a dummy atomic dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (2, 2, 2),
        sequence_crop_size: int = 384,
        sampling_weight_for_disorder_pdb_distillation: float = 0.02,
        train_on_transcription_factor_distillation_sets: bool = False,
        pdb_distillation: Optional[bool] = None,
        max_number_of_chains: int = 20,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load dataset splits only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = AtomDataset(num_examples=self.hparams.train_val_test_split[0])
            self.data_val = AtomDataset(num_examples=self.hparams.train_val_test_split[1])
            self.data_test = AtomDataset(num_examples=self.hparams.train_val_test_split[2])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = AtomDataModule()
