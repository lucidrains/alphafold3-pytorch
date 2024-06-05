from __future__ import annotations

from alphafold3_pytorch.typing import typecheck
from typing import Callable, List

from alphafold3_pytorch.alphafold3 import Alphafold3

from alphafold3_pytorch.trainer import (
    Trainer,
    Dataset,
    Fabric,
    Optimizer,
    LRScheduler
)

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
    assert isinstance(maybe_config_dict, dict), 'yaml config file is not a dictionary'

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
        alphafold3 = Alphafold3(**self.model_dump())
        return alphafold3

    def create_instance_from_yaml_file(path: str | Path) -> Alphafold3:
        af3_config = Alphafold3Config.from_yaml_file(path)
        return af3_config.create_instance()

class TrainerConfig(BaseModelWithExtra):
    model: Alphafold3Config
    num_train_steps: int
    batch_size: int
    grad_accum_every: int
    valid_every: int
    ema_decay: float
    lr: float
    clip_grad_norm: int | float
    accelerator: str 
    checkpoint_prefix: str
    checkpoint_every: int
    checkpoint_folder: str
    overwrite_checkpoints: bool

    @staticmethod
    @typecheck
    def from_yaml_file(path: str | Path):
        config_dict = yaml_config_path_to_dict(path)
        return TrainerConfig(**config_dict)

    def create_instance(
        self,
        dataset: Dataset,
        fabric: Fabric | None = None,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        valid_dataset: Dataset | None = None,
        map_dataset_input_fn: Callable | None = None,
    ) -> Trainer:

        trainer_kwargs = self.model_dump()

        alphafold3 = self.model.create_instance()

        trainer_kwargs.update(dict(
            model = alphafold3,
            dataset = dataset,
            fabric = fabric,
            test_dataset = test_dataset,
            optimizer = optimizer,
            scheduler = scheduler,
            valid_dataset = valid_dataset,
            map_dataset_input_fn = map_dataset_input_fn
        ))

        trainer = Trainer(**trainer_kwargs)
        return trainer

    def create_instance_from_yaml_file(
        path: str | Path,
        **kwargs
    ) -> Trainer:

        trainer_config = TrainerConfig.from_yaml_file(path)
        return trainer_config.create_instance(**kwargs)

# convenience functions

create_alphafold3_from_yaml = Alphafold3Config.create_instance_from_yaml_file
create_trainer_from_yaml = TrainerConfig.create_instance_from_yaml_file
