from random import randrange, random
from dataclasses import asdict

import torch
from torch.utils.data import Dataset
from alphafold3_pytorch import AtomInput
from alphafold3_pytorch.inputs import IS_MOLECULE_TYPES

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
        additional_molecule_feats = torch.randint(0, 2, (seq_len, 5))
        additional_token_feats = torch.randn(seq_len, 2)
        is_molecule_types = torch.randint(0, 2, (seq_len, IS_MOLECULE_TYPES)).bool()
        molecule_ids = torch.randint(0, 32, (seq_len,))
        token_bonds = torch.randint(0, 2, (seq_len, seq_len)).bool()

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
            token_bonds = token_bonds,
            molecule_atom_lens = molecule_atom_lens,
            additional_molecule_feats = additional_molecule_feats,
            additional_token_feats = additional_token_feats,
            is_molecule_types = is_molecule_types,
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
