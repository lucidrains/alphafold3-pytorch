import torch
import torch.nn.functional as F

class SmoothLDDTLoss(torch.nn.Module):
    """Alg 27"""
    def __init__(self, nucleic_acid_cutoff=30.0, other_cutoff=15.0):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

    def forward(self, pred_coords, true_coords, is_dna, is_rna):
        """
        pred_coords: predicted coordinates (b, n, 3)
        true_coords: true coordinates (b, n, 3)
        is_dna: boolean tensor indicating DNA atoms (b, n)
        is_rna: boolean tensor indicating RNA atoms (b, n)
        """
        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values
        eps = (
            F.sigmoid(0.5 - dist_diff) +
            F.sigmoid(1.0 - dist_diff) +
            F.sigmoid(2.0 - dist_diff) +
            F.sigmoid(4.0 - dist_diff)
        ) / 4.0

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = is_nucleotide.unsqueeze(-1) & is_nucleotide.unsqueeze(-2)
        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = torch.logical_and(inclusion_radius, torch.logical_not(torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)))
        lddt_sum = (eps * mask).sum(dim=(-1, -2))
        lddt_count = mask.sum(dim=(-1, -2))
        lddt = lddt_sum / lddt_count.clamp(min=1)

        return 1 - lddt.mean()
    

class WeightedRigidAlign(torch.nn.Module):
    """Alg 28"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_coords, true_coords, weights):
        """
        pred_coords: predicted coordinates (b, n, 3)
        true_coords: true coordinates (b, n, 3)
        weights: weights for each atom (b, n)
        """
        # Compute weighted centroids
        pred_centroid = (pred_coords * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        true_centroid = (true_coords * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)

        # Center the coordinates
        pred_coords_centered = pred_coords - pred_centroid.unsqueeze(1)
        true_coords_centered = true_coords - true_centroid.unsqueeze(1)

        # Compute the weighted covariance matrix
        cov_matrix = torch.einsum('bni,bnj->bij', true_coords_centered * weights.unsqueeze(-1), pred_coords_centered)

        # Compute the SVD of the covariance matrix
        U, _, V = torch.svd(cov_matrix)

        # Compute the rotation matrix
        rot_matrix = torch.einsum('bij,bjk->bik', U, V)

        # Ensure proper rotation matrix with determinant 1
        det = torch.det(rot_matrix)
        det_mask = det < 0
        V_fixed = V.clone()
        V_fixed[det_mask, :, -1] *= -1
        rot_matrix[det_mask] = torch.einsum('bij,bjk->bik', U[det_mask], V_fixed[det_mask])

        # Apply the rotation and translation
        aligned_coords = torch.einsum('bni,bij->bnj', pred_coords_centered, rot_matrix) + true_centroid.unsqueeze(1)

        return aligned_coords.detach()