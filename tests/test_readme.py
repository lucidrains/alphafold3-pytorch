
def test_readme1():
    import torch
    from alphafold3_pytorch import Alphafold3
    from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 108
    )

    # mock inputs

    seq_len = 16

    molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long()
    molecule_atom_lens = torch.full((2, seq_len), 2).long()

    atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
    atom_offsets = exclusive_cumsum(molecule_atom_lens)

    atom_inputs = torch.randn(2, atom_seq_len, 77)
    atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

    additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5))
    additional_token_feats = torch.randn(2, seq_len, 33)
    is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool()
    is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool()
    molecule_ids = torch.randint(0, 32, (2, seq_len))

    template_feats = torch.randn(2, 2, seq_len, seq_len, 108)
    template_mask = torch.ones((2, 2)).bool()

    msa = torch.randn(2, 7, seq_len, 32)
    msa_mask = torch.ones((2, 7)).bool()

    additional_msa_feats = torch.randn(2, 7, seq_len, 2)

    # required for training, but omitted on inference

    atom_pos = torch.randn(2, atom_seq_len, 3)

    distogram_atom_indices = molecule_atom_lens - 1

    distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
    resolved_labels = torch.randint(0, 2, (2, atom_seq_len))

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

def test_readme2():
    import torch
    from alphafold3_pytorch import Alphafold3, Alphafold3Input

    contrived_protein = 'AG'

    mock_atompos = [
        torch.randn(5, 3),   # alanine has 5 non-hydrogen atoms
        torch.randn(4, 3)    # glycine has 4 non-hydrogen atoms
    ]

    train_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein],
        atom_pos = mock_atompos
    )

    eval_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein]
    )

    # training

    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_template_feats = 108,
        num_molecule_mods = 0,
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
        )
    )

    loss = alphafold3.forward_with_alphafold3_inputs([train_alphafold3_input])
    loss.backward()

    # sampling

    alphafold3.eval()
    sampled_atom_pos = alphafold3.forward_with_alphafold3_inputs(eval_alphafold3_input)

    assert sampled_atom_pos.shape == (1, (5 + 4), 3)
