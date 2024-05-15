import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch import (
    PairformerStack,
    MSAModule,
    DiffusionTransformer
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
