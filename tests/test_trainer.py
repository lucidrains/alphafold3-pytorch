import os
os.environ['TYPECHECK'] = 'True'

from pathlib import Path
from random import randrange, random

import pytest
import torch
from torch.utils.data import Dataset

from alphafold3_pytorch import (
    Alphafold3,
    AtomInput,
    DataLoader,
    Trainer,
    ConductorConfig,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml,
    create_alphafold3_from_yaml
)

def exists(v):
    return v is not None

# mock dataset

class MockAtomDataset(Dataset):
    def __init__(
        self,
        data_length,
        max_seq_len = 16,
        atoms_per_window = 4
    ):
        self.data_length = data_length
        self.max_seq_len = max_seq_len
        self.atoms_per_window = atoms_per_window

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        seq_len = randrange(1, self.max_seq_len)
        atom_seq_len = self.atoms_per_window * seq_len

        atom_inputs = torch.randn(atom_seq_len, 77)
        atompair_inputs = torch.randn(atom_seq_len, atom_seq_len, 5)

        molecule_atom_lens = torch.randint(1, self.atoms_per_window, (seq_len,))
        additional_molecule_feats = torch.randn(seq_len, 9)
        molecule_ids = torch.randint(0, 32, (seq_len,))

        templates = torch.randn(2, seq_len, seq_len, 44)
        template_mask = torch.ones((2,)).bool()

        msa = torch.randn(7, seq_len, 64)

        msa_mask = None
        if random() > 0.5:
            msa_mask = torch.ones((7,)).bool()

        # required for training, but omitted on inference

        atom_pos = torch.randn(atom_seq_len, 3)
        molecule_atom_indices = molecule_atom_lens - 1

        distance_labels = torch.randint(0, 37, (seq_len, seq_len))
        pae_labels = torch.randint(0, 64, (seq_len, seq_len))
        pde_labels = torch.randint(0, 64, (seq_len, seq_len))
        plddt_labels = torch.randint(0, 50, (seq_len,))
        resolved_labels = torch.randint(0, 2, (seq_len,))

        return AtomInput(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            molecule_ids = molecule_ids,
            molecule_atom_lens = molecule_atom_lens,
            additional_molecule_feats = additional_molecule_feats,
            templates = templates,
            template_mask = template_mask,
            msa = msa,
            msa_mask = msa_mask,
            atom_pos = atom_pos,
            molecule_atom_indices = molecule_atom_indices,
            distance_labels = distance_labels,
            pae_labels = pae_labels,
            pde_labels = pde_labels,
            plddt_labels = plddt_labels,
            resolved_labels = resolved_labels
        )

def test_trainer():
    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 44,
        num_dist_bins = 38,
        confidence_head_kwargs = dict(
            pairformer_depth = 1
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1
        ),
        msa_module_kwargs = dict(
            depth = 1
        ),
        pairformer_stack = dict(
            depth = 1
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    dataset = MockAtomDataset(100)
    valid_dataset = MockAtomDataset(4)
    test_dataset = MockAtomDataset(2)

    # test saving and loading from Alphafold3, independent of lightning

    dataloader = DataLoader(dataset, batch_size = 2)
    inputs = next(iter(dataloader))

    alphafold3.eval()
    _, breakdown = alphafold3(**inputs, return_loss_breakdown = True)
    before_distogram = breakdown.distogram

    path = './some/nested/folder/af3'
    alphafold3.save(path, overwrite = True)

    # load from scratch, along with saved hyperparameters

    alphafold3 = Alphafold3.init_and_load(path)

    alphafold3.eval()
    _, breakdown = alphafold3(**inputs, return_loss_breakdown = True)
    after_distogram = breakdown.distogram

    assert torch.allclose(before_distogram, after_distogram)

    # test training + validation

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        accelerator = 'cpu',
        num_train_steps = 2,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 2,
        checkpoint_every = 1,
        overwrite_checkpoints = True,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    trainer()

    # assert checkpoints created

    assert Path(f'./checkpoints/({trainer.train_id})_af3.ckpt.1.pt').exists()

    # assert can load latest checkpoint by loading from a directory

    trainer.load('./checkpoints')

    assert exists(trainer.model_loaded_from_path)

    # saving and loading from trainer

    trainer.save('./some/nested/folder2/training.pt', overwrite = True)
    trainer.load('./some/nested/folder2/training.pt')

    # allow for only loading model, needed for fine-tuning logic

    trainer.load('./some/nested/folder2/training.pt', only_model = True)

    # also allow for loading Alphafold3 directly from training ckpt

    alphafold3 = Alphafold3.init_and_load('./some/nested/folder2/training.pt')

# test creating trainer + alphafold3 from config

def test_trainer_config():
    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer.yaml'

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# test creating trainer without model, given when creating instance

def test_trainer_config_without_model():
    curr_dir = Path(__file__).parents[0]

    af3_yaml_path = curr_dir / 'configs/alphafold3.yaml'
    trainer_yaml_path = curr_dir / 'configs/trainer_without_model.yaml'

    alphafold3 = create_alphafold3_from_yaml(af3_yaml_path)

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        model = alphafold3,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

# test creating trainer from training config yaml

def test_conductor_config():
    curr_dir = Path(__file__).parents[0]
    training_yaml_path = curr_dir / 'configs/training.yaml'

    trainer = create_trainer_from_conductor_yaml(
        training_yaml_path,
        trainer_name = 'main',
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    assert str(trainer.checkpoint_folder) == 'main-and-finetuning/main'
    assert str(trainer.checkpoint_prefix) == 'af3.main.ckpt.'
