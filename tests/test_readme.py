import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch import (
    PairformerStack,
    MSAModule,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    Attention
)

def test_pairformer():
    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    mask = torch.randint(0, 2, (2, 16)).bool()

    pairformer = PairformerStack(
        depth = 4
    )

    single_out, pairwise_out = pairformer(
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert single.shape == single_out.shape
    assert pairwise.shape == pairwise_out.shape

def test_msa_module():

    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    msa = torch.randn(2, 7, 16, 64)
    mask = torch.randint(0, 2, (2, 16)).bool()

    msa_module = MSAModule()

    pairwise_out = msa_module(
        msa = msa,
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert pairwise.shape == pairwise_out.shape


def test_diffusion_transformer():

    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    mask = torch.randint(0, 2, (2, 16)).bool()

    diffusion_transformer = DiffusionTransformer(
        depth = 2,
        heads = 16
    )

    single_out = diffusion_transformer(
        single,
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert single.shape == single_out.shape

def test_sequence_local_attn():
    atoms = torch.randn(2, 17, 32)
    attn_bias = torch.randn(2, 17, 17)

    attn = Attention(
        dim = 32,
        dim_head = 16,
        heads = 8,
        window_size = 5
    )

    out = attn(atoms, attn_bias = attn_bias)
    assert out.shape == atoms.shape

def test_diffusion_module():

    seq_len = 16
    atom_seq_len = 27 * 16

    noised_atom_pos = torch.randn(2, atom_seq_len, 3)
    atom_feats = torch.randn(2, atom_seq_len, 128)
    atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
    atom_mask = torch.ones((2, atom_seq_len)).bool()

    times = torch.randn(2,)
    mask = torch.ones(2, seq_len).bool()
    single_trunk_repr = torch.randn(2, seq_len, 128)
    single_inputs_repr = torch.randn(2, seq_len, 256)

    pairwise_trunk = torch.randn(2, seq_len, seq_len, 128)
    pairwise_rel_pos_feats = torch.randn(2, seq_len, seq_len, 12)

    diffusion_module = DiffusionModule(
        atoms_per_window = 27,
        dim_pairwise_trunk = 128,
        dim_pairwise_rel_pos_feats = 12,
        atom_encoder_depth = 1,
        atom_decoder_depth = 1,
        token_transformer_depth = 1
    )

    atom_pos_update = diffusion_module(
        noised_atom_pos,
        times = times,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        atom_mask = atom_mask,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats
    )

    assert noised_atom_pos.shape == atom_pos_update.shape

    edm = ElucidatedAtomDiffusion(
        diffusion_module,
        num_sample_steps = 2
    )

    loss = edm(
        noised_atom_pos,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        atom_mask = atom_mask,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats
    )

    assert loss.numel() == 1

    sampled_atom_pos = edm.sample(
        atom_mask = atom_mask,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats
    )

    assert sampled_atom_pos.shape == noised_atom_pos.shape