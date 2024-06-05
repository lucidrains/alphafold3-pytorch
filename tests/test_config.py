import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest
from pathlib import Path

from alphafold3_pytorch.alphafold3 import Alphafold3

from alphafold3_pytorch.configs import (
    Alphafold3Config
)

# constants

curr_dir = Path(__file__).parents[0]

# tests

def test_alphafold3_config():
    af3_yaml = curr_dir / 'alphafold3.yaml'

    alphafold3 = Alphafold3Config.create_instance_from_yaml_file(af3_yaml)
    assert isinstance(alphafold3, Alphafold3)
