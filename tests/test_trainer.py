import os
os.environ['TYPECHECK'] = 'True'

import pytest
import torch
from torch.utils.data import Dataset

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    Trainer
)

# mock dataset

class AtomDataset(Dataset):
    def __init__(
        self,
        seq_len = 16,
        atoms_per_window = 27
    ):
        self.seq_len = seq_len
        self.atom_seq_len = seq_len * atoms_per_window

    def __len__(self):
        return 100

    def __getitem__(self, idx):
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

        return Alphafold3Input(
            atom_inputs = atom_inputs,
            residue_atom_lens = residue_atom_lens,
            atompair_feats = atompair_feats,
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

    dataset = AtomDataset()

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        accelerator = 'cpu',
        num_train_steps = 2,
        batch_size = 1,
        grad_accum_every = 2
    )

    trainer()
