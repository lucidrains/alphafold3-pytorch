import click
from pathlib import Path

import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    alphafold3_inputs_to_batched_atom_input
)

# simple cli using click

@click.command()
@click.option('-ckpt', '--checkpoint', type = str, help = 'path to alphafold3 checkpoint')
@click.option('-p', '--protein', type = str, help = 'one protein sequence')
@click.option('-o', '--output', type = str, help = 'output path', default = 'atompos.pt')
def cli(
    checkpoint: str,
    protein: str,
    output: str
):

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f'alphafold3 checkpoint must exist at {str(checkpoint_path)}'

    alphafold3_input = Alphafold3Input(
        proteins = [protein],
    )

    alphafold3 = Alphafold3.init_and_load(checkpoint_path)

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input, atoms_per_window = alphafold3.atoms_per_window)

    alphafold3.eval()
    sampled_atom_pos = alphafold3(**batched_atom_input.model_forward_dict())

    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok = True, parents = True)

    torch.save(sampled_atom_pos, str(output_path))

    print(f'atomic positions saved to {str(output_path)}')
