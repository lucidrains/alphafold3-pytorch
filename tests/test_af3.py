import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch import (
    SmoothLDDTLoss,
    WeightedRigidAlign,
    ExpressCoordinatesInFrame,
    ComputeAlignmentError,
    CentreRandomAugmentation,
    PairformerStack,
    MSAModule,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    RelativePositionEncoding,
    TemplateEmbedder,
    Attention,
    InputFeatureEmbedder,
    ConfidenceHead,
    DistogramHead,
    Alphafold3,
)

from alphafold3_pytorch.alphafold3 import (
    calc_smooth_lddt_loss
)

def test_calc_smooth_lddt_loss():
    denoised = torch.randn(8, 100, 3)
    ground_truth = torch.randn(8, 100, 3)
    is_rna_per_atom = torch.randint(0, 2, (8, 100)).float()
    is_dna_per_atom = torch.randint(0, 2, (8, 100)).float()
    
    loss = calc_smooth_lddt_loss(
        denoised, 
        ground_truth, 
        is_rna_per_atom, 
        is_dna_per_atom
    )

    assert torch.all(loss <= 1) and torch.all(loss >= 0)

# ToDo tests

def test_smooth_lddt_loss():
    pred_coords = torch.randn(2, 100, 3)
    true_coords = torch.randn(2, 100, 3)
    is_dna = torch.randint(0, 2, (2, 100)).bool()
    is_rna = torch.randint(0, 2, (2, 100)).bool()

    loss_fn = SmoothLDDTLoss()
    loss = loss_fn(pred_coords, true_coords, is_dna, is_rna)

    assert loss.numel() == 1

def test_weighted_rigid_align():
    pred_coords = torch.randn(2, 100, 3)
    true_coords = torch.randn(2, 100, 3)
    weights = torch.rand(2, 100)

    align_fn = WeightedRigidAlign()
    aligned_coords = align_fn(pred_coords, true_coords, weights)

    assert aligned_coords.shape == pred_coords.shape

def test_express_coordinates_in_frame():
    batch_size = 2
    num_coords = 100
    coords = torch.randn(batch_size, num_coords, 3)
    frame = torch.randn(batch_size, num_coords, 3, 3)

    express_fn = ExpressCoordinatesInFrame()
    transformed_coords = express_fn(coords, frame)

    assert transformed_coords.shape == (batch_size, num_coords, 3)

    broadcastable_seq_frame = torch.randn(batch_size, 3, 3)
    transformed_coords = express_fn(coords, broadcastable_seq_frame)

    assert transformed_coords.shape == (batch_size, num_coords, 3)

    broadcastable_batch_and_seq_frame = torch.randn(3, 3)
    transformed_coords = express_fn(coords, broadcastable_batch_and_seq_frame)

    assert transformed_coords.shape == (batch_size, num_coords, 3)

def test_compute_alignment_error():
    pred_coords = torch.randn(2, 100, 3)
    true_coords = torch.randn(2, 100, 3)
    pred_frames = torch.randn(2, 100, 3, 3)
    true_frames = torch.randn(2, 100, 3, 3)

    error_fn = ComputeAlignmentError()
    alignment_errors = error_fn(pred_coords, true_coords, pred_frames, true_frames)

    assert alignment_errors.shape == (2, 100)

def test_centre_random_augmentation():
    coords = torch.randn(2, 100, 3)

    augmentation_fn = CentreRandomAugmentation()
    augmented_coords = augmentation_fn(coords)

    assert augmented_coords.shape == coords.shape


def test_pairformer():
    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    mask = torch.randint(0, 2, (2, 16)).bool()

    pairformer = PairformerStack(
        depth = 4
    )

    single_out, pairwise_out = pairformer(
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert single.shape == single_out.shape
    assert pairwise.shape == pairwise_out.shape

def test_msa_module():

    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    msa = torch.randn(2, 7, 16, 64)
    mask = torch.randint(0, 2, (2, 16)).bool()

    msa_module = MSAModule(
        max_num_msa = 3 # will randomly select 3 out of the MSAs, accounting for mask, using sample without replacement
    )

    pairwise_out = msa_module(
        msa = msa,
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert pairwise.shape == pairwise_out.shape


def test_diffusion_transformer():

    single = torch.randn(2, 16, 384)
    pairwise = torch.randn(2, 16, 16, 128)
    mask = torch.randint(0, 2, (2, 16)).bool()

    diffusion_transformer = DiffusionTransformer(
        depth = 2,
        heads = 16
    )

    single_out = diffusion_transformer(
        single,
        single_repr = single,
        pairwise_repr = pairwise,
        mask = mask
    )

    assert single.shape == single_out.shape

def test_sequence_local_attn():
    atoms = torch.randn(2, 17, 32)
    attn_bias = torch.randn(2, 17, 17)

    attn = Attention(
        dim = 32,
        dim_head = 16,
        heads = 8,
        window_size = 5
    )

    out = attn(atoms, attn_bias = attn_bias)
    assert out.shape == atoms.shape

def test_diffusion_module():

    seq_len = 16
    atom_seq_len = 27 * 16

    noised_atom_pos = torch.randn(2, atom_seq_len, 3)
    atom_feats = torch.randn(2, atom_seq_len, 128)
    atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
    atom_mask = torch.ones((2, atom_seq_len)).bool()

    times = torch.randn(2,)
    mask = torch.ones(2, seq_len).bool()
    single_trunk_repr = torch.randn(2, seq_len, 128)
    single_inputs_repr = torch.randn(2, seq_len, 256)

    pairwise_trunk = torch.randn(2, seq_len, seq_len, 128)
    pairwise_rel_pos_feats = torch.randn(2, seq_len, seq_len, 12)

    diffusion_module = DiffusionModule(
        atoms_per_window = 27,
        dim_pairwise_trunk = 128,
        dim_pairwise_rel_pos_feats = 12,
        atom_encoder_depth = 1,
        atom_decoder_depth = 1,
        token_transformer_depth = 1
    )

    atom_pos_update = diffusion_module(
        noised_atom_pos,
        times = times,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        atom_mask = atom_mask,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats
    )

    assert noised_atom_pos.shape == atom_pos_update.shape

    edm = ElucidatedAtomDiffusion(
        diffusion_module,
        num_sample_steps = 2
    )

    edm_return = edm(
        noised_atom_pos,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        atom_mask = atom_mask,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats,
        add_bond_loss = True
    )

    assert edm_return.loss.numel() == 1

    sampled_atom_pos = edm.sample(
        atom_mask = atom_mask,
        atom_feats = atom_feats,
        atompair_feats = atompair_feats,
        mask = mask,
        single_trunk_repr = single_trunk_repr,
        single_inputs_repr = single_inputs_repr,
        pairwise_trunk = pairwise_trunk,
        pairwise_rel_pos_feats = pairwise_rel_pos_feats
    )

    assert sampled_atom_pos.shape == noised_atom_pos.shape
    
def test_relative_position_encoding():
    additional_residue_feats = torch.randn(8, 100, 10)

    embedder = RelativePositionEncoding()

    rpe_embed = embedder(
        additional_residue_feats = additional_residue_feats
    )

def test_template_embed():
    template_feats = torch.randn(2, 2, 16, 16, 77)
    template_mask = torch.ones((2, 2)).bool()

    pairwise_repr = torch.randn(2, 16, 16, 128)
    mask = torch.ones((2, 16)).bool()

    embedder = TemplateEmbedder(
        dim_template_feats = 77
    )

    template_embed = embedder(
        templates = template_feats,
        template_mask = template_mask,
        pairwise_repr = pairwise_repr,
        mask = mask
    )


def test_confidence_head():
    single_inputs_repr = torch.randn(2, 16, 77)
    single_repr = torch.randn(2, 16, 384)
    pairwise_repr = torch.randn(2, 16, 16, 128)
    pred_atom_pos = torch.randn(2, 16, 3)
    mask = torch.ones((2, 16)).bool()

    confidence_head = ConfidenceHead(
        dim_single_inputs = 77,
        atompair_dist_bins = torch.linspace(3, 20, 37),
        dim_single = 384,
        dim_pairwise = 128,
    )

    confidence_head(
        single_inputs_repr = single_inputs_repr,
        single_repr = single_repr,
        pairwise_repr = pairwise_repr,
        pred_atom_pos = pred_atom_pos,
        mask = mask
    )

def test_input_embedder():

    atom_seq_len = 16 * 27
    atom_inputs = torch.randn(2, atom_seq_len, 77)
    atom_mask = torch.ones((2, atom_seq_len)).bool()
    atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
    additional_residue_feats = torch.randn(2, 16, 10)

    embedder = InputFeatureEmbedder(
        dim_atom_inputs = 77,
        dim_additional_residue_feats = 10
    )

    embedder(
        atom_inputs = atom_inputs,
        atom_mask = atom_mask,
        atompair_feats = atompair_feats,
        additional_residue_feats = additional_residue_feats
    )

def test_distogram_head():
    pairwise_repr = torch.randn(2, 16, 16, 128)

    distogram_head = DistogramHead(dim_pairwise = 128)

    logits = distogram_head(pairwise_repr)


def test_alphafold3():
    seq_len = 16
    atom_seq_len = seq_len * 27

    atom_inputs = torch.randn(2, atom_seq_len, 77)
    atom_mask = torch.ones((2, atom_seq_len)).bool()
    atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
    additional_residue_feats = torch.randn(2, seq_len, 10)

    template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
    template_mask = torch.ones((2, 2)).bool()

    msa = torch.randn(2, 7, seq_len, 64)

    atom_pos = torch.randn(2, atom_seq_len, 3)
    residue_atom_indices = torch.randint(0, 27, (2, seq_len))

    distance_labels = torch.randint(0, 38, (2, seq_len, seq_len))
    pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
    pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
    plddt_labels = torch.randint(0, 50, (2, seq_len))
    resolved_labels = torch.randint(0, 2, (2, seq_len))

    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_additional_residue_feats = 10,
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
        ),
    )

    loss, breakdown = alphafold3(
        num_recycling_steps = 2,
        atom_inputs = atom_inputs,
        atom_mask = atom_mask,
        atompair_feats = atompair_feats,
        additional_residue_feats = additional_residue_feats,
        msa = msa,
        templates = template_feats,
        template_mask = template_mask,
        atom_pos = atom_pos,
        residue_atom_indices = residue_atom_indices,
        distance_labels = distance_labels,
        pae_labels = pae_labels,
        pde_labels = pde_labels,
        plddt_labels = plddt_labels,
        resolved_labels = resolved_labels,
        return_loss_breakdown = True
    )

    loss.backward()

    sampled_atom_pos = alphafold3(
        num_sample_steps = 16,
        atom_inputs = atom_inputs,
        atom_mask = atom_mask,
        atompair_feats = atompair_feats,
        additional_residue_feats = additional_residue_feats,
        msa = msa,
        templates = template_feats,
        template_mask = template_mask,
    )

    assert sampled_atom_pos.ndim == 3


def test_token_bonds_embedding():
    batch_size = 2
    seq_len = 16
    atom_seq_len = seq_len * 27

    atom_inputs = torch.randn(batch_size, atom_seq_len, 77)
    atom_mask = torch.ones((batch_size, atom_seq_len)).bool()
    atompair_feats = torch.randn(batch_size, atom_seq_len, atom_seq_len, 16)
    additional_residue_feats = torch.randn(batch_size, seq_len, 10)

    # Test case 1: token_bonds is provided
    token_bonds = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()

    model = Alphafold3(
        dim_atom_inputs=77,
        dim_additional_residue_feats=10,
        dim_pairwise=128,
    )

    output = model(
        atom_inputs=atom_inputs,
        atom_mask=atom_mask,
        atompair_feats=atompair_feats,
        additional_residue_feats=additional_residue_feats,
        token_bonds=token_bonds,
    )

    assert output.shape == (batch_size, atom_seq_len, 3)

    # Test case 2: token_bonds is not provided (default to single chain)
    model = Alphafold3(
        dim_atom_inputs=77,
        dim_additional_residue_feats=10,
        dim_pairwise=128,
    )

    output = model(
        atom_inputs=atom_inputs,
        atom_mask=atom_mask,
        atompair_feats=atompair_feats,
        additional_residue_feats=additional_residue_feats,
    )

    assert output.shape == (batch_size, atom_seq_len, 3)