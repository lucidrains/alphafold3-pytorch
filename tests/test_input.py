import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input,
    AtomInput,
    maybe_transform_to_atom_input,
    collate_inputs_to_batched_atom_input
)

from alphafold3_pytorch.life import (
    reverse_complement,
    reverse_complement_tensor
)

def test_string_reverse_complement():
    assert reverse_complement('ATCG') == 'CGAT'
    assert reverse_complement('AUCG', 'rna') == 'CGAU'

def test_tensor_reverse_complement():
    seq = torch.randint(0, 5, (100,))
    rc = reverse_complement_tensor(seq)
    assert torch.allclose(reverse_complement_tensor(rc), seq)

def test_alphafold3_input():

    alphafold3_input = Alphafold3Input(
        proteins = ['MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDF', 'MGKCRGLRTARKLRSHRRDQKWHDKQYKKAHLGTALKANPFGGASHAKGIVLEKVGVEAKQPNSAIRKCVRVQLIKNGKKITAFVPNDGCLNFIEENDEVLVAGFGRKGHAVGDIPGVRFKVVKVANVSLLALYKGKKERPRS'],
        ds_dna = ['ACGTT'],
        ds_rna = ['GCCAU'],
        ss_dna = ['GCCTA'],
        ss_rna = ['CGCAUA'],
        metal_ions = ['Na', 'Na', 'Fe'],
        misc_molecule_ids = ['Phospholipid'],
        ligands = ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'],
        add_atom_ids = True,
        add_atompair_ids = True
    )

    atom_input = maybe_transform_to_atom_input(alphafold3_input)

    assert isinstance(atom_input, AtomInput)

    # feed it into alphafold3

    batched_atom_input = collate_inputs_to_batched_atom_input([atom_input], atoms_per_window = 27)

    alphafold3 = Alphafold3(
        dim_atom_inputs = 1,
        dim_atompair_inputs = 1,
        num_atom_embeds = 47,
        num_atompair_embeds = 6,
        atoms_per_window = 27,
        dim_template_feats = 44,
        num_dist_bins = 38,
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

    alphafold3(**batched_atom_input.dict(), num_sample_steps = 1)
