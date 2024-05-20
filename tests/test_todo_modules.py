import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch.todo_modules import (
    SmoothLDDTLoss,
    WeightedRigidAlign,
    ExpressCoordinatesInFrame,
    ComputeAlignmentError,
    CentreRandomAugmentation
)

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
    coords = torch.randn(2, 3)
    frame = torch.randn(2, 3, 3)

    express_fn = ExpressCoordinatesInFrame()
    transformed_coords = express_fn(coords, frame)

    assert transformed_coords.shape == coords.shape

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