import pytest
from pathlib import Path

from alphafold3_pytorch.inputs import PDBDataset, pdb_inputs_to_batched_atom_input
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.utils.utils import exists


def test_template_loading():
    """Test a template-featurized PDBDataset constructed using a WeightedPDBSampler."""
    data_test = Path("data", "test")
    data_test_mmcif_dir = data_test / "mmcifs"
    data_test_clusterings_dir = data_test / "data_caches" / "clusterings"
    data_test_template_dir = data_test / "data_caches" / "template" / "templates"

    if not data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{data_test_mmcif_dir}` is not populated yet.")

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
        batch_size=64,
    )

    sampler_pdb_ids = set(sampler.mappings.get_column("pdb_id").to_list())
    test_ids = set(
        filepath.stem
        for filepath in data_test_mmcif_dir.glob("**/*.cif")
        if filepath.stem in sampler_pdb_ids
    )

    pdb_input = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        crop_size=128,
        templates_dir=str(data_test_template_dir),
        sample_only_pdb_ids=test_ids,
        training=False,
    )

    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)
    assert exists(batched_atom_input)
