import os
os.environ['TYPECHECK'] = 'True'

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    Trainer
)

# mock dataset

class MockAtomDataset(Dataset):
    def __init__(
        self,
        data_length,
        seq_len = 16,
        atoms_per_window = 27
    ):
        self.data_length = data_length
        self.seq_len = seq_len
        self.atom_seq_len = seq_len * atoms_per_window

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        seq_len = self.seq_len
        atom_seq_len = self.atom_seq_len

        atom_inputs = torch.randn(atom_seq_len, 77)
        atompair_inputs = torch.randn(atom_seq_len, atom_seq_len, 5)

        residue_atom_lens = torch.randint(0, 27, (seq_len,))
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

        return Alphafold3Input(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            residue_atom_lens = residue_atom_lens,
            additional_residue_feats = additional_residue_feats,
            templates = templates,
            template_mask = template_mask,
            msa = msa,
            msa_mask = msa_mask,
            atom_pos = atom_pos,
            residue_atom_indices = residue_atom_indices,
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
            depth = 2
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
        overwrite_checkpoints = True
    )

    trainer()

    assert Path('./checkpoints/af3.ckpt.1.pt').exists()

    # saving and loading from trainer

    trainer.save('./some/nested/folder2/training', overwrite = True)
    trainer.load('./some/nested/folder2/training')

    # also allow for loading Alphafold3 directly from training ckpt

    alphafold3 = Alphafold3.init_and_load('./some/nested/folder2/training')
