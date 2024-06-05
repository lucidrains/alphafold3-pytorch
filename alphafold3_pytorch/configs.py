from __future__ import annotations

from alphafold3_pytorch.typing import typecheck
from alphafold3_pytorch.alphafold3 import Alphafold3

import yaml
from pathlib import Path

from pydantic import BaseModel

# functions

def exists(v):
    return v is not None

@typecheck
def yaml_config_path_to_dict(
    path: str | Path
) -> dict | None:

    if isinstance(path, str):
        path = Path(path)

    assert path.is_file()

    with open(str(path), 'r') as f:
        maybe_config_dict = yaml.safe_load(f)

    assert exists(maybe_config_dict), f'unable to parse yaml config at {str(path)}'
    assert isinstance(maybe_config_dict, dict), f'yaml config file is not a dictionary'

    return maybe_config_dict

# base pydantic classes for constructing alphafold3 and trainer from config files

class BaseModelWithExtra(BaseModel):
    class Config:
        extra = 'allow'
        use_enum_values = True

class Alphafold3Config(BaseModelWithExtra):
    dim_atom_inputs: int
    dim_template_feats: int
    dim_template_model: int
    atoms_per_window: int
    dim_atom: int
    dim_atompair_inputs: int
    dim_atompair: int
    dim_input_embedder_token: int
    dim_single: int
    dim_pairwise: int
    dim_token: int
    ignore_index: int = -1
    num_dist_bins: int | None
    num_plddt_bins: int
    num_pde_bins: int
    num_pae_bins: int
    sigma_data: int | float
    diffusion_num_augmentations: int
    loss_confidence_weight: int | float
    loss_distogram_weight: int | float
    loss_diffusion_weight: int | float

    @staticmethod
    @typecheck
    def from_yaml_file(path: str | Path):
        config_dict = yaml_config_path_to_dict(path)
        return Alphafold3Config(**config_dict)

    def create_instance(self) -> Alphafold3:
        alphafold3 = Alphafold3(**self.dict())
        return alphafold3

    def create_instance_from_yaml_file(path: str | Path) -> Alphafold3:
        af3_config = Alphafold3Config.from_yaml_file(path)
        return af3_config.create_instance()
