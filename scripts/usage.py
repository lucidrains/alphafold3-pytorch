import rootutils
import torch

from alphafold3_pytorch import AlphaFold3

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


alphafold3 = AlphaFold3(dim_atom_inputs=77, dim_template_feats=44)

# Mock inputs

seq_len = 16
atom_seq_len = seq_len * 27

atom_inputs = torch.randn(2, atom_seq_len, 77)
atom_lens = torch.randint(0, 27, (2, seq_len))
atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
additional_residue_feats = torch.randn(2, seq_len, 10)

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()

# Required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
residue_atom_indices = torch.randint(0, 27, (2, seq_len))

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

# Train

loss = alphafold3(
    num_recycling_steps=2,
    atom_inputs=atom_inputs,
    residue_atom_lens=atom_lens,
    atompair_feats=atompair_feats,
    additional_residue_feats=additional_residue_feats,
    msa=msa,
    msa_mask=msa_mask,
    templates=template_feats,
    template_mask=template_mask,
    atom_pos=atom_pos,
    residue_atom_indices=residue_atom_indices,
    distance_labels=distance_labels,
    pae_labels=pae_labels,
    pde_labels=pde_labels,
    plddt_labels=plddt_labels,
    resolved_labels=resolved_labels,
)

loss.backward()

# After much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps=4,
    num_sample_steps=16,
    atom_inputs=atom_inputs,
    residue_atom_lens=atom_lens,
    atompair_feats=atompair_feats,
    additional_residue_feats=additional_residue_feats,
    msa=msa,
    msa_mask=msa_mask,
    templates=template_feats,
    template_mask=template_mask,
)

sampled_atom_pos.shape  # (2, 16 * 27, 3)
