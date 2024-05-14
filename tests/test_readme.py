import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch import (
    PairformerStack,
    MSAModule
)

def test_pairformer():
    single = torch.randn(2, 16, 512)
    pairwise = torch.randn(2, 16, 16, 256)
    mask = torch.randint(0, 2, (2, 16)).bool()

    pairformer = PairformerStack(
        depth = 4,
        dim_single = 512,
        dim_pairwise = 256
    )

    single_out, pairwise_out = pairformer(
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert single.shape == single_out.shape
    assert pairwise.shape == pairwise_out.shape

def test_msa_module():

    single = torch.randn(2, 16, 512)
    msa = torch.randn(2, 7, 16, 64)
    pairwise = torch.randn(2, 16, 16, 128)
    mask = torch.randint(0, 2, (2, 16)).bool()

    msa_module = MSAModule(
        dim_single = 512,
        dim_pairwise = 128,
        dim_msa = 64
    )

    pairwise_out = msa_module(
        single_repr = single,
        msa = msa,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert pairwise.shape == pairwise_out.shape
