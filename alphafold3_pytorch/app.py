import click
from pathlib import Path

import gradio as gr

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    alphafold3_inputs_to_batched_atom_input
)

# constants

model = None

# main fold functoin

def fold(protein):
    alphafold3_input = Alphafold3Input(
        proteins = [protein]
    )

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input, atoms_per_window = model.atoms_per_window)

    model.eval()
    atom_pos, = model(**batched_atom_input.model_forward_dict())

    return str(atom_pos.tolist())

# gradio

gradio_app = gr.Interface(
    fn = fold,
    inputs = [
        "text"
    ],
    outputs = [
        "text"
    ],
)

# cli

@click.command()
@click.option('-ckpt', '--checkpoint', type = str, help = 'path to alphafold3 checkpoint', required = True)
def app(checkpoint: str):
    path = Path(checkpoint)
    assert path.exists(), 'checkpoint does not exist at path'

    global model
    model = Alphafold3.init_and_load(str(path))

    gradio_app.launch()
