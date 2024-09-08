import os
os.environ['TYPECHECK'] = 'True'
os.environ['DEBUG'] = 'True'
from shutil import rmtree
from pathlib import Path

import torch

from alphafold3_pytorch.cli import cli

from alphafold3_pytorch.alphafold3 import (
    Alphafold3
)

def test_cli():
    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_template_feats = 44,
        num_molecule_mods = 0
    )

    checkpoint_path = './test-folder/test-cli-alphafold3.pt'
    alphafold3.save(checkpoint_path, overwrite = True)

    cli([
        '--checkpoint', checkpoint_path,
        '-prot', 'AG',
        '-prot', 'TC',
        '--output',
        './test-folder/output.mmcif'
    ], standalone_mode = False)

    assert Path('./test-folder/output.mmcif').exists()

    rmtree('./test-folder')
