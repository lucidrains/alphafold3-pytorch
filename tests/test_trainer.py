import os
os.environ['TYPECHECK'] = 'True'

import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from alphafold3_pytorch import (
    Alphafold3,
    PDBDataset,
    AtomInput,
    atom_input_to_file,
    DataLoader,
    Trainer,
    ConductorConfig,
    collate_inputs_to_batched_atom_input,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml,
    create_alphafold3_from_yaml
)

from alphafold3_pytorch.mocks import MockAtomDataset

DATA_TEST_PDB_ID = '209d'

def exists(v):
    return v is not None

@pytest.fixture()
def remove_test_folders():
    yield
    shutil.rmtree('./test-folder')

def test_trainer_with_mock_atom_input(remove_test_folders):

    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 108,
        num_dist_bins = 64,
        confidence_head_kwargs = dict(
            pairformer_depth = 1
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1
        ),
        msa_module_kwargs = dict(
            depth = 1
        ),
        pairformer_stack = dict(
            depth = 1
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    dataset = MockAtomDataset(100)
    valid_dataset = MockAtomDataset(4)
    test_dataset = MockAtomDataset(2)

    # test saving and loading from Alphafold3, independent of lightning

    dataloader = DataLoader(dataset, batch_size = 2)
    inputs = next(iter(dataloader))

    alphafold3.eval()
    _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)
    before_distogram = breakdown.distogram

    path = './test-folder/nested/folder/af3'
    alphafold3.save(path, overwrite = True)

    # load from scratch, along with saved hyperparameters

    alphafold3 = Alphafold3.init_and_load(path)

    alphafold3.eval()
    _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)
    after_distogram = breakdown.distogram

    assert torch.allclose(before_distogram, after_distogram)

    # test training + validation

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        accelerator = 'cpu',
        num_train_steps = 2,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 2,
        checkpoint_every = 1,
        checkpoint_folder = './test-folder/checkpoints',
        overwrite_checkpoints = True,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    trainer()

    # assert checkpoints created

    assert Path(f'./test-folder/checkpoints/({trainer.train_id})_af3.ckpt.1.pt').exists()

    # assert can load latest checkpoint by loading from a directory

    trainer.load('./test-folder/checkpoints', strict = False)

    assert exists(trainer.model_loaded_from_path)

    # saving and loading from trainer

    trainer.save('./test-folder/nested/folder2/training.pt', overwrite = True)
    trainer.load('./test-folder/nested/folder2/training.pt', strict = False)

    # allow for only loading model, needed for fine-tuning logic

    trainer.load('./test-folder/nested/folder2/training.pt', only_model = True, strict = False)

    # also allow for loading Alphafold3 directly from training ckpt

    alphafold3 = Alphafold3.init_and_load('./test-folder/nested/folder2/training.pt')

# testing trainer with pdb inputs

@pytest.fixture()
def populate_mock_pdb_and_remove_test_folders():
    proj_root = Path('.')
    working_cif_file = proj_root / 'data' / 'test' / 'pdb_data' / 'mmcifs' / DATA_TEST_PDB_ID[1:3] / f'{DATA_TEST_PDB_ID}-assembly1.cif'

    pytest_root_folder = Path('./test-folder')
    data_folder = pytest_root_folder / 'data'

    train_folder = data_folder / 'train'
    valid_folder = data_folder / 'valid'
    test_folder = data_folder / 'test'

    train_folder.mkdir(exist_ok = True, parents = True)
    valid_folder.mkdir(exist_ok = True, parents = True)
    test_folder.mkdir(exist_ok = True, parents = True)

    for i in range(10):
        shutil.copy2(str(working_cif_file), str(train_folder / f'{i}.cif'))

    for i in range(1):
        shutil.copy2(str(working_cif_file), str(valid_folder / f'{i}.cif'))

    for i in range(1):
        shutil.copy2(str(working_cif_file), str(test_folder / f'{i}.cif'))

    yield

    shutil.rmtree('./test-folder')

def test_trainer_with_pdb_input(populate_mock_pdb_and_remove_test_folders):

    alphafold3 = Alphafold3(
        dim_atom=4,
        dim_atompair=4,
        dim_input_embedder_token=4,
        dim_single=4,
        dim_pairwise=4,
        dim_token=4,
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=27,
        dim_template_feats=108,
        num_dist_bins=64,
        confidence_head_kwargs=dict(
            pairformer_depth=1,
        ),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(
            depth=1,
            pair_bias_attn_dim_head = 4,
            pair_bias_attn_heads = 2,
        ),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
            atom_decoder_kwargs = dict(
                attn_pair_bias_kwargs = dict(
                    dim_head = 4
                )
            ),
            atom_encoder_kwargs = dict(
                attn_pair_bias_kwargs = dict(
                    dim_head = 4
                )
            )
        ),
    )

    dataset = PDBDataset('./test-folder/data/train')
    valid_dataset = PDBDataset('./test-folder/data/valid')
    test_dataset = PDBDataset('./test-folder/data/test')

    # test saving and loading from Alphafold3, independent of lightning

    dataloader = DataLoader(dataset, batch_size = 1)
    inputs = next(iter(dataloader))

    alphafold3.eval()
    with torch.no_grad():
        _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)

    before_distogram = breakdown.distogram

    path = './test-folder/nested/folder/af3'
    alphafold3.save(path, overwrite = True)

    # load from scratch, along with saved hyperparameters

    alphafold3 = Alphafold3.init_and_load(path)

    alphafold3.eval()
    with torch.no_grad():
        _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)

    after_distogram = breakdown.distogram

    assert torch.allclose(before_distogram, after_distogram)

    # test training + validation

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        accelerator = 'cpu',
        num_train_steps = 2,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 1,
        checkpoint_every = 1,
        checkpoint_folder = './test-folder/checkpoints',
        overwrite_checkpoints = True,
        use_ema = False,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    trainer()

    # assert checkpoints created

    assert Path(f'./test-folder/checkpoints/({trainer.train_id})_af3.ckpt.1.pt').exists()

    # assert can load latest checkpoint by loading from a directory

    trainer.load('./test-folder/checkpoints', strict = False)

    assert exists(trainer.model_loaded_from_path)

    # saving and loading from trainer

    trainer.save('./test-folder/nested/folder2/training.pt', overwrite = True)
    trainer.load('./test-folder/nested/folder2/training.pt', strict = False)

    # allow for only loading model, needed for fine-tuning logic

    trainer.load('./test-folder/nested/folder2/training.pt', only_model = True, strict = False)

    # also allow for loading Alphafold3 directly from training ckpt

    alphafold3 = Alphafold3.init_and_load('./test-folder/nested/folder2/training.pt')

# test use of collation fn outside of trainer

def test_collate_fn():
    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 108,
        num_dist_bins = 64,
        confidence_head_kwargs = dict(
            pairformer_depth = 1
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1
        ),
        msa_module_kwargs = dict(
            depth = 1
        ),
        pairformer_stack = dict(
            depth = 1
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    dataset = MockAtomDataset(5)

    batched_atom_inputs = collate_inputs_to_batched_atom_input([dataset[i] for i in range(3)])

    _, breakdown = alphafold3(**batched_atom_inputs.model_forward_dict(), return_loss_breakdown = True)

# test creating trainer + alphafold3 from config

def test_trainer_config(remove_test_folders):
    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer.yaml'

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# test creating trainer + alphafold3 along with pdb dataset from config

def test_trainer_config_with_pdb_dataset(populate_mock_pdb_and_remove_test_folders):
    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer_with_pdb_dataset.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# test creating trainer + alphafold3 along with atom dataset from config

def test_trainer_config_with_atom_dataset(remove_test_folders):

    curr_dir = Path(__file__).parents[0]

    # setup atom dataset

    atom_folder = './test-folder/test-atom-folder'
    Path(atom_folder).mkdir(exist_ok = True, parents = True)

    mock_atom_dataset = MockAtomDataset(10)

    for i in range(10):
        atom_input = mock_atom_dataset[i]
        atom_input_to_file(atom_input, f'{atom_folder}/train/{i}.pt', overwrite = True)

    # path to config

    trainer_yaml_path = curr_dir / 'configs/trainer_with_atom_dataset.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# test creating trainer + alphafold3 with atom dataset that is precomputed from a pdb dataset

def test_trainer_config_with_atom_dataset_from_pdb_dataset(populate_mock_pdb_and_remove_test_folders):

    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer_with_atom_dataset_created_from_pdb.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# test creating trainer without model, given when creating instance

def test_trainer_config_without_model(remove_test_folders):
    curr_dir = Path(__file__).parents[0]

    af3_yaml_path = curr_dir / 'configs/alphafold3.yaml'
    trainer_yaml_path = curr_dir / 'configs/trainer_without_model.yaml'

    alphafold3 = create_alphafold3_from_yaml(af3_yaml_path)

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        model = alphafold3,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

# test creating trainer from training config yaml

def test_conductor_config():
    curr_dir = Path(__file__).parents[0]
    training_yaml_path = curr_dir / 'configs/training.yaml'

    trainer = create_trainer_from_conductor_yaml(
        training_yaml_path,
        trainer_name = 'main',
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    assert str(trainer.checkpoint_folder) == 'test-folder/main-and-finetuning/main'
    assert str(trainer.checkpoint_prefix) == 'af3.main.ckpt.'

# test creating trainer from training config yaml + pdb datasets

def test_conductor_config_with_pdb_datasets(populate_mock_pdb_and_remove_test_folders):
    curr_dir = Path(__file__).parents[0]
    training_yaml_path = curr_dir / 'configs/training_with_pdb_dataset.yaml'

    trainer = create_trainer_from_conductor_yaml(
        training_yaml_path,
        trainer_name = 'main'
    )

    assert isinstance(trainer, Trainer)

    assert str(trainer.checkpoint_folder) == 'test-folder/main-and-finetuning/main'
    assert str(trainer.checkpoint_prefix) == 'af3.main.ckpt.'
