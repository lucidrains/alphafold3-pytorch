import torch
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
).to(device)

# mock inputs

seq_len = 16

molecule_atom_indices = torch.randint(0, 2, (2, seq_len), device=device).long()
molecule_atom_lens = torch.full((2, seq_len), 2, device=device).long()

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = torch.randn(2, atom_seq_len, 77, device=device)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5, device=device)

additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5), device=device)
additional_token_feats = torch.randn(2, seq_len, 33, device=device)
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5), device=device).bool()
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4), device=device).bool()
molecule_ids = torch.randint(0, 32, (2, seq_len), device=device)

template_feats = torch.randn(2, 2, seq_len, seq_len, 108, device=device)
template_mask = torch.ones((2, 2), device=device).bool()

msa = torch.randn(2, 7, seq_len, 32, device=device)
msa_mask = torch.ones((2, 7), device=device).bool()

additional_msa_feats = torch.randn(2, 7, seq_len, 2, device=device)

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3, device=device)

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len), device=device)
resolved_labels = torch.randint(0, 2, (2, atom_seq_len), device=device)

# offset indices correctly

distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# train


loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    distogram_atom_indices = distogram_atom_indices,
    molecule_atom_indices = molecule_atom_indices,
    distance_labels = distance_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# after much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

print(sampled_atom_pos.shape) # (2, <atom_seqlen>, 3)
print(sampled_atom_pos)