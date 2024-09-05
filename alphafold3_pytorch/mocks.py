import random

import torch
from torch.utils.data import Dataset
from alphafold3_pytorch import AtomInput

from alphafold3_pytorch.inputs import (
    IS_MOLECULE_TYPES,
    DEFAULT_NUM_MOLECULE_MODS
)

from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

# mock dataset

class MockAtomDataset(Dataset):
    def __init__(
        self,
        data_length,
        max_seq_len = 16,
        atoms_per_window = 4,
        dim_atom_inputs = 77,
        has_molecule_mods = True
    ):
        self.data_length = data_length
        self.max_seq_len = max_seq_len
        self.atoms_per_window = atoms_per_window
        self.dim_atom_inputs = dim_atom_inputs
        self.has_molecule_mods = has_molecule_mods

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        seq_len = random.randrange(1, self.max_seq_len)
        atom_seq_len = self.atoms_per_window * seq_len

        atom_inputs = torch.randn(atom_seq_len, self.dim_atom_inputs)
        atompair_inputs = torch.randn(atom_seq_len, atom_seq_len, 5)

        molecule_atom_lens = torch.randint(1, self.atoms_per_window, (seq_len,))
        atom_offsets = exclusive_cumsum(molecule_atom_lens)

        additional_molecule_feats = torch.randint(0, 2, (seq_len, 5))
        additional_token_feats = torch.randn(seq_len, 33)
        is_molecule_types = torch.randint(0, 2, (seq_len, IS_MOLECULE_TYPES)).bool()

        # ensure the molecule-atom length mappings match the randomly-sampled atom sequence length

        if molecule_atom_lens.sum() < atom_seq_len:
            molecule_atom_lens[-1] = atom_seq_len - molecule_atom_lens[:-1].sum()

        # ensure each unique asymmetric ID has at least one molecule type associated with it

        asym_id = additional_molecule_feats[:, 2]
        unique_asym_id = asym_id.unique()
        for asym in unique_asym_id:
            if any(not row.any() for row in is_molecule_types[asym_id == asym]):
                rand_molecule_type_idx = random.randint(0, IS_MOLECULE_TYPES - 1)  # nosec
                is_molecule_types[asym_id == asym, rand_molecule_type_idx] = True

        is_molecule_mod = None
        if self.has_molecule_mods:
            is_molecule_mod = torch.rand((seq_len, DEFAULT_NUM_MOLECULE_MODS)) < 0.05

        molecule_ids = torch.randint(0, 32, (seq_len,))
        token_bonds = torch.randint(0, 2, (seq_len, seq_len)).bool()

        templates = torch.randn(2, seq_len, seq_len, 108)
        template_mask = torch.ones((2,)).bool()

        msa = torch.randn(7, seq_len, 32)

        msa_mask = None
        if random.random() > 0.5:
            msa_mask = torch.ones((7,)).bool()

        additional_msa_feats = torch.randn(7, seq_len, 2)

        # required for training, but omitted on inference

        atom_pos = torch.randn(atom_seq_len, 3)
        molecule_atom_indices = molecule_atom_lens - 1
        distogram_atom_indices = molecule_atom_lens - 1

        molecule_atom_indices += atom_offsets
        distogram_atom_indices += atom_offsets

        distance_labels = torch.randint(0, 64, (seq_len, seq_len))
        resolved_labels = torch.randint(0, 2, (atom_seq_len,))

        majority_asym_id = asym_id.mode().values.item()
        chains = torch.tensor([majority_asym_id, -1]).long()

        return AtomInput(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            molecule_ids = molecule_ids,
            token_bonds = token_bonds,
            molecule_atom_lens = molecule_atom_lens,
            additional_molecule_feats = additional_molecule_feats,
            additional_msa_feats = additional_msa_feats,
            additional_token_feats = additional_token_feats,
            is_molecule_types = is_molecule_types,
            is_molecule_mod = is_molecule_mod,
            templates = templates,
            template_mask = template_mask,
            msa = msa,
            msa_mask = msa_mask,
            atom_pos = atom_pos,
            molecule_atom_indices = molecule_atom_indices,
            distogram_atom_indices = distogram_atom_indices,
            distance_labels = distance_labels,
            resolved_labels = resolved_labels,
            chains = chains
        )
