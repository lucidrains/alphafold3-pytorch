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
    pdb_inputs_to_batched_atom_input,
    atom_input_to_file,
    file_to_atom_input
)

from alphafold3_pytorch.data import mmcif_writing

from alphafold3_pytorch.life import (
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
def test_alphafold3_input(directed_bonds):

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
        directed_bonds = directed_bonds
    )

    batched_atom_input = alphafold3_inputs_to_batched_atom_input(alphafold3_input)

    # feed it into alphafold3

    num_atom_bond_types = (6 * (2 if directed_bonds else 1))

    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        num_atom_embeds = 47,
        num_atompair_embeds = num_atom_bond_types + 1, # 0 is for no bond
        atoms_per_window = 27,
        dim_template_feats = 44,
        num_dist_bins = 38,
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
        dim_template_feats = 44,
        num_dist_bins = 38,
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

    loss = alphafold3(**batched_atom_input.model_forward_dict())
    loss.backward()

    # sampling

    batched_eval_atom_input = alphafold3_inputs_to_batched_atom_input(eval_alphafold3_input, atoms_per_window = 27)

    alphafold3.eval()
    sampled_atom_pos = alphafold3(**batched_eval_atom_input.model_forward_dict(), return_loss=False)

    assert sampled_atom_pos.shape == (1, (5 + 4 + 21 + 3), 3)

def test_pdbinput_input():
    """Test the PDBInput class, particularly its input transformations for mmCIF files."""
    filepath = os.path.join("data", "test", f"{DATA_TEST_PDB_ID}-assembly1.cif")
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    assert os.path.exists(filepath), f"File {filepath} does not exist."

    train_pdb_input = PDBInput(
        filepath,
        chains=("A", "B"),
        cropping_config={
            "contiguous_weight": 0.2,
            "spatial_weight": 0.4,
            "spatial_interface_weight": 0.4,
            "n_res": 64,
        },
        training=True,
    )

    eval_pdb_input = PDBInput(filepath)

    batched_atom_input = pdb_inputs_to_batched_atom_input(train_pdb_input, atoms_per_window=27)

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
        atoms_per_window=27,
        dim_template_feats=44,
        num_molecule_mods=4,
        num_dist_bins=38,
        num_rollout_steps=2,
        diffusion_num_augmentations=2,
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

    # sampling

    batched_eval_atom_input = pdb_inputs_to_batched_atom_input(eval_pdb_input, atoms_per_window=27)

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
