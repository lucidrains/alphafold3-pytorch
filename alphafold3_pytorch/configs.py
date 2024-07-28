from __future__ import annotations

from alphafold3_pytorch.tensor_typing import typecheck
from typing import Callable, List, Dict, Literal

from alphafold3_pytorch.alphafold3 import Alphafold3

from alphafold3_pytorch.inputs import (
    AtomDataset,
    PDBDataset,
    pdb_dataset_to_atom_inputs
)

from alphafold3_pytorch.trainer import (
    Trainer,
    Dataset,
    Fabric,
    Optimizer,
    LRScheduler
)

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

import yaml
from pathlib import Path

from pydantic import BaseModel, model_validator

from pydantic.types import (
    FilePath,
    DirectoryPath
)

# functions

def exists(v):
    return v is not None

@typecheck
def safe_deep_get(
    d: dict,
    dotpath: str | List[str],  # dotpath notation, so accessing {'a': {'b'': {'c': 1}}} would be "a.b.c"
    default = None
):
    if isinstance(dotpath, str):
        dotpath = dotpath.split('.')

    for key in dotpath:
        if (
            not isinstance(d, dict) or \
            key not in d
        ):
            return default

        d = d[key]

    return d

@typecheck
def yaml_config_path_to_dict(
    path: str | Path
) -> dict:

    if isinstance(path, str):
        path = Path(path)

    assert path.is_file(), f'cannot find {str(path)}'

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

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(self) -> Alphafold3:
        alphafold3 = Alphafold3(**self.model_dump())
        return alphafold3

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ) -> Alphafold3:

        af3_config = cls.from_yaml_file(path, dotpath)
        return af3_config.create_instance()

class WeightedPDBSamplerConfig(BaseModelWithExtra):
    chain_mapping_paths: List[FilePath]
    interface_mapping_path: FilePath

    def create_instance(self, batch_size: int):
        return WeightedPDBSampler(**{
            'batch_size': batch_size,
            **self.model_dump()
        })

class DatasetConfig(BaseModelWithExtra):
    dataset_type: Literal['pdb', 'atom'] = 'pdb'
    train_folder: DirectoryPath
    valid_folder: DirectoryPath | None = None
    test_folder: DirectoryPath | None = None
    convert_pdb_to_atom: bool = False
    pdb_to_atom_kwargs: dict = dict()
    train_weighted_sampler: WeightedPDBSamplerConfig | None = None
    kwargs: dict = dict()

class TrainerConfig(BaseModelWithExtra):
    model: Alphafold3Config | None = None
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
    dataset_config: DatasetConfig | None = None

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(
        self,
        dataset: Dataset | None = None,
        model: Alphafold3 | None = None,
        fabric: Fabric | None = None,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        valid_dataset: Dataset | None = None,
        map_dataset_input_fn: Callable | None = None,
    ) -> Trainer:

        trainer_kwargs = self.model_dump(
            exclude = {'dataset_config'}
        )

        assert exists(self.model) ^ exists(model), 'either model is available on the trainer config, or passed in when creating the instance, but not both or neither'

        # handle model

        if exists(self.model):
            alphafold3 = self.model.create_instance()
        else:
            alphafold3 = model

        # handle dataset

        if exists(dataset):
            trainer_kwargs.update(dataset = dataset)

        if exists(valid_dataset):
            trainer_kwargs.update(valid_dataset = dataset)

        if exists(test_dataset):
            trainer_kwargs.update(test_dataset = dataset)

        if exists(self.dataset_config):
            dataset_config = self.dataset_config

            dataset_type = dataset_config.dataset_type
            dataset_kwargs = dataset_config.kwargs

            convert_pdb_to_atom = dataset_config.convert_pdb_to_atom
            pdb_to_atom_kwargs = dataset_config.pdb_to_atom_kwargs

            if convert_pdb_to_atom:
                assert dataset_type == 'pdb', 'must be `pdb` dataset_type if `convert_pdb_to_atom` is set to True'

            if dataset_type == 'pdb':
                dataset_klass = PDBDataset
            elif dataset_type == 'atom':
                dataset_klass = AtomDataset
            else:
                raise ValueError(f'unhandled dataset_type {dataset_type}')

            # create dataset for train, valid, and test

            for trainer_kwarg_key, config_key in (('dataset', 'train_folder'), ('valid_dataset', 'valid_folder'), ('test_dataset', 'test_folder')):
                folder = getattr(dataset_config, config_key, None)

                if not exists(folder):
                    continue

                assert trainer_kwarg_key not in trainer_kwargs

                dataset = dataset_klass(folder, **dataset_kwargs)

                if convert_pdb_to_atom:
                    dataset = pdb_dataset_to_atom_inputs(dataset, return_atom_dataset = True, **pdb_to_atom_kwargs)

                trainer_kwargs.update(**{trainer_kwarg_key: dataset})

            # handle weighted pdb sampling

            if exists(dataset_config.train_weighted_sampler):
                sampler = dataset_config.train_weighted_sampler.create_instance(batch_size = self.batch_size)

                trainer_kwargs.update(train_sampler = sampler)

        assert 'dataset' in trainer_kwargs, 'dataset is absent - dataset_type must be specified along with train folders (pdb for now), or the Dataset instance must be passed in'

        # handle rest

        trainer_kwargs.update(dict(
            model = alphafold3,
            fabric = fabric,
            test_dataset = test_dataset,
            optimizer = optimizer,
            scheduler = scheduler,
            valid_dataset = valid_dataset,
            map_dataset_input_fn = map_dataset_input_fn
        ))

        trainer = Trainer(**trainer_kwargs)
        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = [],
        **kwargs
    ) -> Trainer:

        trainer_config = cls.from_yaml_file(path, dotpath)
        return trainer_config.create_instance(**kwargs)

# conductor config
# which contains multiple trainer configs for the main and various finetuning stages

class ConductorConfig(BaseModelWithExtra):
    model: Alphafold3Config | None = None
    checkpoint_folder: str
    checkpoint_prefix: str
    training_order: List[str]
    training: Dict[str, TrainerConfig]

    @model_validator(mode = 'after')
    def check_valid_conductor_order(self) -> 'ConductorConfig':
        training_order = set(self.training_order)
        trainer_names = set(self.training.keys())

        if training_order != trainer_names:
            raise ValueError('`training_order` needs to contain all the keys (trainer name) under the `training` field')

        return self

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(
        self,
        trainer_name: str,
        **kwargs
    ) -> Trainer:

        assert trainer_name in self.training, f'{trainer_name} not found among available trainers {tuple(self.training.keys())}'

        model = self.model.create_instance()

        trainer_config = self.training[trainer_name]

        # nest the checkpoint_folder of the trainer within the main checkpoint_folder

        nested_checkpoint_folder = str(Path(self.checkpoint_folder) / Path(trainer_config.checkpoint_folder))

        trainer_config.checkpoint_folder = nested_checkpoint_folder

        # prepend the main training checkpoint_prefix

        nested_checkpoint_prefix = self.checkpoint_prefix + trainer_config.checkpoint_prefix

        trainer_config.checkpoint_prefix = nested_checkpoint_prefix

        # create the Trainer, accounting for root level config

        trainer = trainer_config.create_instance(
            model = model,
            **kwargs
        )

        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = [],
        **kwargs
    ) -> Trainer:

        training_config = cls.from_yaml_file(path, dotpath)
        return training_config.create_instance(**kwargs)

# convenience functions

create_alphafold3_from_yaml = Alphafold3Config.create_instance_from_yaml_file
create_trainer_from_yaml = TrainerConfig.create_instance_from_yaml_file
create_trainer_from_conductor_yaml = ConductorConfig.create_instance_from_yaml_file
