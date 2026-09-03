import pytest
import torch
from torch import nn

from alphafold3_pytorch.alphafold3 import DiffusionModule, ElucidatedAtomDiffusion


class ZeroDenoiser(DiffusionModule):
    def __init__(self):
        # The diffusion wrapper only calls ``forward`` in these tests.  Avoid
        # constructing the full network while retaining the annotated type.
        nn.Module.__init__(self)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, noised_atom_pos, **kwargs):
        return torch.zeros_like(noised_atom_pos)


def test_single_step_schedule_is_finite_and_reaches_zero():
    diffusion = ElucidatedAtomDiffusion(
        ZeroDenoiser(),
        num_sample_steps = 1,
        sigma_max = 80.,
        sigma_data = 0.5
    )

    sigmas = diffusion.sample_schedule(1, device = torch.device('cpu'))

    assert sigmas.shape == (2,)
    assert torch.isfinite(sigmas).all()
    assert torch.allclose(sigmas, torch.tensor([40., 0.]))


def test_single_step_sampling_produces_finite_coordinates():
    diffusion = ElucidatedAtomDiffusion(
        ZeroDenoiser(),
        num_sample_steps = 1,
        sigma_max = 80.,
        sigma_data = 0.5,
        augment_during_sampling = False
    )

    atom_mask = torch.ones((2, 4), dtype = torch.bool)
    sampled = diffusion.sample(
        atom_mask = atom_mask,
        num_sample_steps = 1,
        use_tqdm_pbar = False
    )

    assert sampled.shape == (2, 4, 3)
    assert torch.isfinite(sampled).all()


def test_invalid_sample_step_count_raises():
    diffusion = ElucidatedAtomDiffusion(ZeroDenoiser())

    with pytest.raises(ValueError, match = 'greater than zero'):
        diffusion.sample_schedule(0, device = torch.device('cpu'))
