import os
import pytest
import shutil
import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    AtomInput,
    AtomDataset,
    PDBInput,
    maybe_transform_to_atom_input,
    collate_inputs_to_batched_atom_input,
    alphafold3_inputs_to_batched_atom_input,
    alphafold3_input_to_biomolecule,
    pdb_inputs_to_batched_atom_input,
    atom_input_to_file,
    file_to_atom_input
)

from alphafold3_pytorch.data.data_pipeline import *
from alphafold3_pytorch.data.data_pipeline import make_mmcif_features

from alphafold3_pytorch.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
    to_inference_mmcif,
    to_mmcif
)

from alphafold3_pytorch.tensor_typing import IS_GITHUB_CI

from alphafold3_pytorch.data import mmcif_writing, mmcif_parsing

from alphafold3_pytorch.life import (
    ATOMS,
    reverse_complement,
    reverse_complement_tensor
)

from alphafold3_pytorch.mocks import MockAtomDataset

DATA_TEST_PDB_ID = '7a4d'

# reverse complements

def test_string_reverse_complement():
    assert reverse_complement('ATCG') == 'CGAT'
    assert reverse_complement('AUCG', 'rna') == 'CGAU'

def test_tensor_reverse_complement():
    seq = torch.randint(0, 5, (100,))
    rc = reverse_complement_tensor(seq)
    assert torch.allclose(reverse_complement_tensor(rc), seq)

# atom input

def test_atom_input_to_file_and_from():
    mock_atom_dataset = MockAtomDataset(64)
    atom_input = mock_atom_dataset[0]

    file = atom_input_to_file(atom_input, './test-atom-input.pt', overwrite = True)
    atom_input_reconstituted = file_to_atom_input(str(file))
    assert torch.allclose(atom_input.atom_inputs, atom_input_reconstituted.atom_inputs)

def test_atom_dataset():
    num_atom_inputs = 10
    test_folder = './test_atom_folder'

    mock_atom_dataset = MockAtomDataset(num_atom_inputs)

    for i in range(num_atom_inputs):
        atom_input = mock_atom_dataset[i]
        atom_input_to_file(atom_input, f'{test_folder}/{i}.pt', overwrite = True)

    atom_dataset_from_disk = AtomDataset(test_folder)
    assert len(atom_dataset_from_disk) == num_atom_inputs

    shutil.rmtree(test_folder, ignore_errors = True)

# alphafold3 input

@pytest.mark.parametrize('directed_bonds', (False, True))
def test_alphafold3_input(
    directed_bonds
):

    CUSTOM_ATOMS = list({*ATOMS, 'Na', 'Fe', 'Si', 'F', 'K'})

    alphafold3_input = Alphafold3Input(
        proteins = ['MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDF', 'MGKCRGLRTARKLRSHRRDQKWHDKQYKKAHLGTALKANPFGGASHAKGIVLEKVGVEAKQPNSAIRKCVRVQLIKNGKKITAFVPNDGCLNFIEENDEVLVAGFGRKGHAVGDIPGVRFKVVKVANVSLLALYKGKKERPRS'],
        ds_dna = ['ACGTT'],
        ds_rna = ['GCCAU', 'CCAGU'],
        ss_dna = ['GCCTA'],
        ss_rna = ['CGCAUA'],
        metal_ions = ['Na', 'Na', 'Fe'],
        misc_molecule_ids = ['Phospholipid'],
        ligands = ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'],
        add_atom_ids = True,
        add_atompair_ids = True,
        directed_bonds = directed_bonds,
        custom_atoms = CUSTOM_ATOMS
    )

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input)

    # feed it into alphafold3

    num_atom_bond_types = (6 * (2 if directed_bonds else 1))

    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        num_atom_embeds = len(CUSTOM_ATOMS),
        num_atompair_embeds = num_atom_bond_types + 1, # 0 is for no bond
        atoms_per_window = 27,
        dim_template_feats = 108,
        num_dist_bins = 64,
        num_molecule_mods = 0,
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
            depth = 2
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        )
    )

    alphafold3(**batched_atom_input.model_forward_dict(), num_sample_steps = 1)


def test_alphafold3_input_to_biomolecule():

    alphafold3_input = Alphafold3Input(
        proteins = ['MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDF', 'MGKCRGLRTARKLRSHRRDQKWHDKQYKKAHLGTALKANPFGGASHAKGIVLEKVGVEAKQPNSAIRKCVRVQLIKNGKKITAFVPNDGCLNFIEENDEVLVAGFGRKGHAVGDIPGVRFKVVKVANVSLLALYKGKKERPRS'],
        ds_dna = ['ACGTT'],
        ds_rna = ['GCCAU', 'CCAGU'],
        ss_dna = ['GCCTA'],
        ss_rna = ['CGCAUA'],
        metal_ions = ['Na', 'Na', 'Fe'],
        misc_molecule_ids = ['Phospholipid'],
        ligands = ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'],
        add_atom_ids = True,
        add_atompair_ids = True,
        directed_bonds = True
    )

    test_biomol = alphafold3_input_to_biomolecule(alphafold3_input, atom_positions=torch.randn(261,47,3).numpy())
    # Ensure that the residues got loaded correctly
    assert test_biomol.chemid.tolist() == ['MET', 'LEU', 'GLU', 'ILE', 'CYS', 'LEU', 'LYS', 'LEU', 'VAL',
       'GLY', 'CYS', 'LYS', 'SER', 'LYS', 'LYS', 'GLY', 'LEU', 'SER',
       'SER', 'SER', 'SER', 'SER', 'CYS', 'TYR', 'LEU', 'GLU', 'GLU',
       'ALA', 'LEU', 'GLN', 'ARG', 'PRO', 'VAL', 'ALA', 'SER', 'ASP',
       'PHE', 'MET', 'GLY', 'LYS', 'CYS', 'ARG', 'GLY', 'LEU', 'ARG',
       'THR', 'ALA', 'ARG', 'LYS', 'LEU', 'ARG', 'SER', 'HIS', 'ARG',
       'ARG', 'ASP', 'GLN', 'LYS', 'TRP', 'HIS', 'ASP', 'LYS', 'GLN',
       'TYR', 'LYS', 'LYS', 'ALA', 'HIS', 'LEU', 'GLY', 'THR', 'ALA',
       'LEU', 'LYS', 'ALA', 'ASN', 'PRO', 'PHE', 'GLY', 'GLY', 'ALA',
       'SER', 'HIS', 'ALA', 'LYS', 'GLY', 'ILE', 'VAL', 'LEU', 'GLU',
       'LYS', 'VAL', 'GLY', 'VAL', 'GLU', 'ALA', 'LYS', 'GLN', 'PRO',
       'ASN', 'SER', 'ALA', 'ILE', 'ARG', 'LYS', 'CYS', 'VAL', 'ARG',
       'VAL', 'GLN', 'LEU', 'ILE', 'LYS', 'ASN', 'GLY', 'LYS', 'LYS',
       'ILE', 'THR', 'ALA', 'PHE', 'VAL', 'PRO', 'ASN', 'ASP', 'GLY',
       'CYS', 'LEU', 'ASN', 'PHE', 'ILE', 'GLU', 'GLU', 'ASN', 'ASP',
       'GLU', 'VAL', 'LEU', 'VAL', 'ALA', 'GLY', 'PHE', 'GLY', 'ARG',
       'LYS', 'GLY', 'HIS', 'ALA', 'VAL', 'GLY', 'ASP', 'ILE', 'PRO',
       'GLY', 'VAL', 'ARG', 'PHE', 'LYS', 'VAL', 'VAL', 'LYS', 'VAL',
       'ALA', 'ASN', 'VAL', 'SER', 'LEU', 'LEU', 'ALA', 'LEU', 'TYR',
       'LYS', 'GLY', 'LYS', 'LYS', 'GLU', 'ARG', 'PRO', 'ARG', 'SER', 'C',
       'G', 'C', 'A', 'U', 'A', 'G', 'C', 'C', 'A', 'U', 'A', 'U', 'G',
       'G', 'C', 'C', 'C', 'A', 'G', 'U', 'A', 'C', 'U', 'G', 'G', 'DG',
       'DC', 'DC', 'DT', 'DA', 'DA', 'DC', 'DG', 'DT', 'DT', 'DA', 'DA',
       'DC', 'DG', 'DT', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL',
       'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL',
       'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL',
       'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL', 'UNL',
       'UNL', 'UNL', 'UNL', 'UNK', 'UNK', 'UNK']


def test_alphafold3_input_to_mmcif(tmp_path):
    """Test the Inference I/O Pipeline. This codifies the data_pipeline.py file used for training."""
    
    alphafold3_input = Alphafold3Input(
        proteins = ['MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDF', 'MGKCRGLRTARKLRSHRRDQKWHDKQYKKAHLGTALKANPFGGASHAKGIVLEKVGVEAKQPNSAIRKCVRVQLIKNGKKITAFVPNDGCLNFIEENDEVLVAGFGRKGHAVGDIPGVRFKVVKVANVSLLALYKGKKERPRS'],
        ds_dna = ['ACGTT'],
        ds_rna = ['GCCAU', 'CCAGU'],
        ss_dna = ['GCCTA'],
        ss_rna = ['CGCAUA'],
        metal_ions = ['Na', 'Na', 'Fe'],
        misc_molecule_ids = ['Phospholipid'],
        ligands = ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'],
        add_atom_ids = True,
        add_atompair_ids = True,
        directed_bonds = True
    )
    
    test_biomol = alphafold3_input_to_biomolecule(alphafold3_input, atom_positions=torch.randn(261, 47, 3).numpy())
    mmcif_string = to_inference_mmcif(
                                        test_biomol,
                                        "test.cif",
                                        gapless_poly_seq=True,
                                        insert_alphafold_mmcif_metadata=True
                                    )
    # Use Pytest tempfile
    filepath = os.path.join(tmp_path, "test.cif")
    file_id = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'w') as f:
        f.write(mmcif_string)
    
    mmcif_object = mmcif_parsing.parse_mmcif_object(
        filepath=filepath,
        file_id=file_id,
    )
    mmcif_feats, assembly = make_mmcif_features(mmcif_object)

    mmcif_string = to_mmcif(
        assembly,
        file_id=file_id,
        gapless_poly_seq=True,
        insert_alphafold_mmcif_metadata=True,
        unique_res_atom_names=assembly.unique_res_atom_names,
    )

    # Use tempfile here as well
    with open(os.path.basename(filepath).replace(".cif", "_reconstructed.cif"), "w") as f:
        f.write(mmcif_string)

    assert True


def test_return_bio_pdb_structures():

    alphafold3_input = Alphafold3Input(
        proteins = ['MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDF', 'MGKCRGLRTARKLRSHRRDQKWHDKQYKKAHLGTALKANPFGGASHAKGIVLEKVGVEAKQPNSAIRKCVRVQLIKNGKKITAFVPNDGCLNFIEENDEVLVAGFGRKGHAVGDIPGVRFKVVKVANVSLLALYKGKKERPRS'],
    )

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input)

    # feed it into alphafold3

    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        num_atom_embeds = 0,
        num_atompair_embeds = 0,
        atoms_per_window = 27,
        dim_template_feats = 108,
        num_dist_bins = 64,
        num_molecule_mods = 0,
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
            depth = 2
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        )
    )

    alphafold3(**batched_atom_input.model_forward_dict(), num_sample_steps = 1, return_bio_pdb_structures = True)

def test_atompos_input():

    contrived_protein = 'AG'
    contrived_nucleic_acid = 'C'

    mock_atompos = [
        torch.randn(5, 3),   # alanine has 5 non-hydrogen atoms
        torch.randn(4, 3),   # glycine has 4 non-hydrogen atoms
        torch.randn(21, 3),  # cytosine has 21 atoms
        torch.randn(3, 3)    # ligand has 3 carbons
    ]

    train_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein],
        ss_rna = [contrived_nucleic_acid],
        missing_atom_indices = [[1, 2], None, [0, 1], None, None],
        ligands = ['CCC'],
        atom_pos = mock_atompos
    )

    eval_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein],
        ss_rna = [contrived_nucleic_acid],
        ligands = ['CCC'],
    )

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(train_alphafold3_input, atoms_per_window = 27)

    # training

    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_template_feats = 108,
        num_dist_bins = 64,
        num_molecule_mods = 0,
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
            depth = 2
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    loss = alphafold3(**batched_atom_input.model_forward_dict())
    loss.backward()

    # sampling

    batched_eval_atom_input = alphafold3_inputs_to_batched_atom_input(eval_alphafold3_input, atoms_per_window = 27)

    alphafold3.eval()
    sampled_atom_pos = alphafold3(**batched_eval_atom_input.model_forward_dict(), return_loss=False)

    assert sampled_atom_pos.shape == (1, (5 + 4 + 21 + 3), 3)

def test_pdbinput_input():

    """Test the PDBInput class, particularly its input transformations for mmCIF files."""
    filepath = os.path.join(
        "data",
        "test",
        "pdb_data",
        "mmcifs",
        DATA_TEST_PDB_ID[1:3],
        f"{DATA_TEST_PDB_ID}-assembly1.cif",
    )
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    assert os.path.exists(filepath), f"File {filepath} does not exist."

    train_pdb_input = PDBInput(
        filepath,
        chains=("A", "B"),
        cropping_config={
            "contiguous_weight": 0.2,
            "spatial_weight": 0.4,
            "spatial_interface_weight": 0.4,
            "n_res": 4,
        },
        training=True,
    )

    eval_pdb_input = PDBInput(filepath)

    batched_atom_input = pdb_inputs_to_batched_atom_input(train_pdb_input, atoms_per_window=4)

    # training

    alphafold3 = Alphafold3(
        dim_atom=2,
        dim_atompair=2,
        dim_input_embedder_token=8,
        dim_single=2,
        dim_pairwise=2,
        dim_token=2,
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=4,
        dim_template_feats=108,
        num_molecule_mods=4,
        num_dist_bins=64,
        num_rollout_steps=1,
        diffusion_num_augmentations=1,
        confidence_head_kwargs=dict(pairformer_depth=1),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1, dim_msa=2),
        pairformer_stack=dict(
            depth=1,
            pair_bias_attn_dim_head=1,
            pair_bias_attn_heads=2,
        ),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
            atom_decoder_kwargs=dict(attn_pair_bias_kwargs=dict(dim_head=1)),
            atom_encoder_kwargs=dict(attn_pair_bias_kwargs=dict(dim_head=1)),
        ),
    )

    loss = alphafold3(**batched_atom_input.model_forward_dict())
    loss.backward()

    # sampling is too much for github ci for now

    if IS_GITHUB_CI:
        return

    # sampling

    batched_eval_atom_input = pdb_inputs_to_batched_atom_input(eval_pdb_input, atoms_per_window=4)

    alphafold3.eval()

    batch_dict = batched_eval_atom_input.model_forward_dict()
    sampled_atom_pos = alphafold3(
        **batch_dict,
        return_loss=False,
    )

    batched_atom_mask = ~batch_dict["missing_atom_mask"]
    sampled_atom_positions = sampled_atom_pos[batched_atom_mask].cpu().numpy()

    assert sampled_atom_positions.shape == (4155, 3)

    # visualizing

    mmcif_writing.write_mmcif_from_filepath_and_id(
        input_filepath=filepath,
        output_filepath=filepath.replace(".cif", "-sampled.cif"),
        file_id=file_id,
        gapless_poly_seq=True,
        insert_orig_atom_names=True,
        insert_alphafold_mmcif_metadata=True,
        sampled_atom_positions=sampled_atom_positions,
    )
    assert os.path.exists(filepath.replace(".cif", "-sampled.cif"))
