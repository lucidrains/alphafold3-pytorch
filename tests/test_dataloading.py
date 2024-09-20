from pathlib import Path

import pytest
import torch

from alphafold3_pytorch import collate_inputs_to_batched_atom_input
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import (
    PDBDataset,
    PDBDistillationDataset,
    molecule_to_atom_input,
    pdb_input_to_molecule_input,
)
from alphafold3_pytorch.utils.utils import exists

DATA_TEST_PDB_ID = "721p"


def test_data_input():
    """Test a PDBDataset constructed using a WeightedPDBSampler along with a
    PDBDistillationDataset."""
    data_test = Path("data", "test", "pdb_data")
    data_test_mmcif_dir = data_test / "mmcifs"
    data_test_clusterings_dir = data_test / "data_caches" / "clusterings"

    distillation_data_test = Path("data", "test", "afdb_data")
    distillation_data_test_mmcif_dir = distillation_data_test / "mmcifs"
    distillation_uniprot_to_pdb_id_mapping_filepath = (
        distillation_data_test / "data_caches" / "uniprot_to_pdb_id_mapping.dat"
    )

    if not data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{data_test_mmcif_dir}` is not populated yet.")

    if not distillation_data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{distillation_data_test_mmcif_dir}` is not populated yet.")

    interface_mapping_path = str(data_test_clusterings_dir / "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        str(data_test_clusterings_dir / "ligand_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "nucleic_acid_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "peptide_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "protein_chain_cluster_mapping.csv"),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        pdb_ids_to_keep=[f"{DATA_TEST_PDB_ID}-assembly1"],
        batch_size=2,
    )

    sampler_pdb_ids = set(sampler.mappings.get_column("pdb_id").to_list())
    test_ids = set(
        filepath.stem
        for filepath in data_test_mmcif_dir.glob("**/*.cif")
        if filepath.stem in sampler_pdb_ids
    )

    dataset = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        sample_only_pdb_ids=test_ids,
        crop_size=4,
    )

    distillation_test_ids = {r[0] for r in sampler.mappings.select("pdb_id").rows()}
    distillation_test_ids = (
        distillation_test_ids.intersection(test_ids) if exists(test_ids) else distillation_test_ids
    )

    distillation_dataset = PDBDistillationDataset(
        folder=distillation_data_test_mmcif_dir,
        sample_only_pdb_ids=distillation_test_ids,
        crop_size=4,
        distillation=True,
        distillation_template_mmcif_dir=data_test_mmcif_dir,
        uniprot_to_pdb_id_mapping_filepath=distillation_uniprot_to_pdb_id_mapping_filepath,
    )

    combined_dataset = torch.utils.data.ConcatDataset([dataset, distillation_dataset])

    mol_input = pdb_input_to_molecule_input(pdb_input=combined_dataset[0])
    atom_input = molecule_to_atom_input(mol_input)
    batched_atom_input = collate_inputs_to_batched_atom_input([atom_input], atoms_per_window=4)

    alphafold3 = Alphafold3(
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=4,
        dim_template_feats=108,
        num_dist_bins=64,
        confidence_head_kwargs=dict(pairformer_depth=1),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(depth=1),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
        ),
    )

    loss = alphafold3(**batched_atom_input.model_forward_dict())
    loss.backward()
