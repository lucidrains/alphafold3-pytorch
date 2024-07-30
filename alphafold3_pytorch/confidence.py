from __future__ import annotations
from typing import List, Literal, NamedTuple
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import Module

from alphafold3_pytorch.alphafold3 import (
    ConfidenceHeadLogits,
    repeat_consecutive_with_lens
)


from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    typecheck
)

from alphafold3_pytorch.inputs import (
    IS_MOLECULE_TYPES,
    IS_PROTEIN_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
    IS_BIOMOLECULE_INDICES,
)

import einx
from einops import rearrange, repeat, einsum


# TODO: define index for DNA and RNA
IS_DNA_INDEX = 1
IS_RNA_INDEX = 2

IS_PROTEIN, IS_DNA, IS_RNA, IS_LIGAND, IS_METAL_ION = map(
    lambda x: IS_MOLECULE_TYPES - x if x < 0 else x, [
        IS_PROTEIN_INDEX, IS_DNA_INDEX, IS_RNA_INDEX, IS_LIGAND_INDEX, IS_METAL_ION_INDEX])

IS_BIOMOLECULE = (0, 1, 2)


def exists(v):
    return v is not None

def get_cid_molecule_type(
    cid: int,
    asym_id: Int['n'],
    is_molecule_types: Bool['n {IS_MOLECULE_TYPES}'],
    return_one_hot: bool = False,
    ) -> int | Bool[' {IS_MOLECULE_TYPES}']:
    """
    
    get the molecule type for where asym_id == cid
    """

    cid_is_molecule_types = is_molecule_types[asym_id == cid]
    valid = torch.all(
        einx.equal('b i, i -> b i', 
                   cid_is_molecule_types,
                   cid_is_molecule_types[0])
        )
    assert valid, f"Ambiguous molecule types for chain {cid}"

    if return_one_hot:
        molecule_type = cid_is_molecule_types[0]
    else: 
        molecule_type = cid_is_molecule_types[0].int().argmax().item()
    return molecule_type

class ConfidenceScore(NamedTuple):
    plddt: Float['b n']
    ptm: Float[' b']
    iptm: Float[' b'] | None

class ComputeConfidenceScore(Module):
    
    @typecheck
    def __init__(
        self,
        pae_breaks: Float[' pae_break'] = torch.arange(0, 31.5, 0.5),
        pde_breaks: Float[' pde_break'] = torch.arange(0, 31.5, 0.5),
        eps: float = 1e-8
    ):

        super().__init__()
        self.pae_breaks = pae_breaks
        self.pde_breaks = pde_breaks
        self.eps = eps

    @typecheck
    def _calculate_bin_centers(
        self,
        breaks,
    ):
        """

        Args:
            breaks: [num_bins -1] bin edges

        Returns:
            bin_centers: [num_bins] bin centers
        """

        step = breaks[1] - breaks[0]

        bin_centers = breaks + step / 2
        last_bin_center = breaks[-1] + step

        bin_centers = torch.concat(
            [bin_centers, last_bin_center.unsqueeze(0)]
        )

        return bin_centers

    @typecheck
    def forward(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        ptm_residue_weight: Float['b n'] | None = None,
        multimer_mode: bool=True,
    ):
        device = asym_id.device
        plddt = self.compute_plddt(confidence_head_logits.plddt)

        # Section 5.9.1 equation 17
        ptm = self.compute_ptm(confidence_head_logits.pae, self.pae_breaks.to(device),
                               asym_id, has_frame, ptm_residue_weight, interface=False)

        iptm = None

        if multimer_mode:
            # Section 5.9.2 equation 18
            iptm = self.compute_ptm(confidence_head_logits.pae, self.pae_breaks.to(device),
                                asym_id, has_frame, ptm_residue_weight, interface=True)

        confidence_score = ConfidenceScore(plddt=plddt, ptm=ptm, iptm=iptm)
        return confidence_score

    @typecheck
    def compute_plddt(
        self,
        logits: Float['b plddt m'],
    )->Float['b m']:
        
        logits = rearrange(logits, 'b plddt m -> b m plddt')
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=logits.device)
        probs = F.softmax(logits, dim=-1)

        predicted_lddt = einsum(probs, bin_centers, 'b m plddt, plddt -> b m')
        return predicted_lddt * 100

    @typecheck
    def compute_ptm(
        self,
        logits: Float['b pae n n '],
        breaks: Float[' d'],
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        residue_weights: Float['b n'] | None = None,
        interface: bool = False,
        compute_chain_wise_iptm: bool = False,  
    ): 

        device = logits.device

        if not exists(residue_weights):
            residue_weights = torch.ones_like(has_frame)

        residue_weights = residue_weights * has_frame
                
        num_batch = logits.shape[0]
        num_res = logits.shape[-1]
        logits = rearrange(logits, 'b c i j -> b i j c')
        
        bin_centers = self._calculate_bin_centers(breaks.to(device))
 
        num_frame = torch.sum(has_frame, dim=-1)
        # Clip num_frame to avoid negative/undefined d0.
        clipped_num_frame = torch.clamp(num_frame, min=19)

        # Compute d_0(num_frame) as defined by TM-score, eqn. (5) in Yang & Skolnick
        # "Scoring function for automated assessment of protein structure template
        # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
        d0 = 1.24 * (clipped_num_frame - 15) ** (1.0 / 3) - 1.8

        # TM-Score term for every bin. [num_batch, num_bins]
        tm_per_bin = 1.0 / (1 + torch.square(bin_centers[None, :]) / torch.square(d0[..., None]))

        # Convert logits to probs.
        probs = F.softmax(logits, dim=-1)

        # E_distances tm(distance).
        predicted_tm_term = einsum(probs, tm_per_bin, 'b i j pae, b pae -> b i j ')

        if compute_chain_wise_iptm:

            # chain_wise_iptm[b, i, j]: iptm of chain i and chain j in batch b

            # get the max num_chains across batch 
            unique_chains = [torch.unique(asym).tolist() for asym in asym_id]
            max_chains = max(len(chains) for chains in unique_chains)

            chain_wise_iptm = torch.zeros((num_batch, max_chains, max_chains), device=logits.device)
            chain_wise_iptm_mask = torch.zeros_like(chain_wise_iptm).bool()

            for b in range(num_batch):
                for i, chain_i in enumerate(unique_chains[b]):
                    for j, chain_j in enumerate(unique_chains[b]):
                        if chain_i != chain_j:
                            mask_i = (asym_id[b] == chain_i)[:, None]
                            mask_j = (asym_id[b] == chain_j)[None, :]
                            pair_mask = mask_i * mask_j
                            pair_residue_weights = pair_mask * (
                                residue_weights[b, None, :] * residue_weights[b, :, None])

                            if pair_residue_weights.sum() == 0:
                                # chain i or chain j doesnot have any valid frame
                                continue
                            else:
                                normed_residue_mask = pair_residue_weights / (self.eps + torch.sum(
                                    pair_residue_weights, dim=-1, keepdims=True))

                                masked_predicted_tm_term = predicted_tm_term[b] * pair_mask

                                per_alignment = torch.sum(masked_predicted_tm_term * normed_residue_mask, dim=-1)
                                weighted_argmax = (residue_weights[b] * per_alignment).argmax()
                                chain_wise_iptm[b, i, j] = per_alignment[weighted_argmax]
                                chain_wise_iptm_mask[b, i, j] = True

            return chain_wise_iptm, chain_wise_iptm_mask, unique_chains

        else:

            pair_mask = torch.ones(size=(num_batch, num_res, num_res), device=logits.device).bool()
            if interface:
                pair_mask *= asym_id[:, :, None] != asym_id[:, None, :]

            predicted_tm_term *= pair_mask

            pair_residue_weights = pair_mask * (
                residue_weights[:, None, :] * residue_weights[:, :, None])
            normed_residue_mask = pair_residue_weights / (self.eps + torch.sum(
                pair_residue_weights, dim=-1, keepdims=True))

            per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
            weighted_argmax = (residue_weights * per_alignment).argmax(dim=-1)
            return per_alignment[torch.arange(num_batch) , weighted_argmax]

    @typecheck
    def compute_pde(
        self,
        logits: Float['b pde n n'],
        breaks: Float[' pde_break'],
        tok_repr_atm_mask: Bool[' b n'],
    )-> Float[' b n n']:
        
        logits = rearrange(logits, 'b pde i j -> b i j pde')
        bin_centers = self._calculate_bin_centers(breaks.to(logits.device))
        probs = F.softmax(logits, dim=-1)

        pde = einsum(probs, bin_centers, 'b i j pde, pde -> b i j ')

        mask = einx.logical_and(
            'b i, b j -> b i j', tok_repr_atm_mask, tok_repr_atm_mask)
        
        pde = pde * mask
        return pde
    
class ComputeClash(Module):
    def __init__(
        self,
        atom_clash_dist=1.1,
        chain_clash_count=100,
        chain_clash_ratio=0.5
    ):

        super().__init__()
        self.atom_clash_dist = atom_clash_dist
        self.chain_clash_count = chain_clash_count
        self.chain_clash_ratio = chain_clash_ratio

    def compute_has_clash(
        self,
        atom_pos: Float['m 3'],
        asym_id: Int[' n'],
        indices: Int[' m'],
        valid_indices: Int[' m'],
    )-> Bool['']:

        # Section 5.9.2

        atom_pos = atom_pos[valid_indices]
        atom_asym_id = asym_id[indices][valid_indices]

        unique_chains = atom_asym_id.unique()
        for i in range(len(unique_chains)):
            for j in range(i + 1, len(unique_chains)):
                chain_i, chain_j = unique_chains[i], unique_chains[j]

                mask_i = atom_asym_id == chain_i
                mask_j = atom_asym_id == chain_j

                chain_i_len = mask_i.sum()
                chain_j_len = mask_j.sum()
                assert min(chain_i_len, chain_j_len) > 0

                chain_pair_dist  = torch.cdist(atom_pos[mask_i], atom_pos[mask_j])
                chain_pair_clash = chain_pair_dist < self.atom_clash_dist
                clashes = chain_pair_clash.sum()
                has_clash = (
                    (clashes > self.chain_clash_count) or 
                    ( clashes / min(chain_i_len, chain_j_len) > self.chain_clash_ratio )
                )

                if has_clash:
                    return torch.tensor(True, dtype=torch.bool, device=atom_pos.device)
        
        return torch.tensor(False, dtype=torch.bool, device=atom_pos.device)
                
    def forward(
        self,
        atom_pos: Float['b m 3'] | Float['m 3'],
        atom_mask: Bool['b m'] | Bool[' m'],
        molecule_atom_lens: Int['b n'] | Int[' n'],
        asym_id: Int['b n']| Int[' n'],
    )-> Bool:

        if atom_pos.ndim ==2:
            atom_pos = atom_pos.unsqueeze(0)
            molecule_atom_lens = molecule_atom_lens.unsqueeze(0)
            asym_id = asym_id.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)

        device = atom_pos.device
        batch_size, seq_len= asym_id.shape

        indices = torch.arange(seq_len, device = device)

        indices = repeat(indices, 'n -> b n', b = batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = repeat_consecutive_with_lens(indices, molecule_atom_lens)
        valid_indices = repeat_consecutive_with_lens(valid_indices, molecule_atom_lens)

        if atom_mask is not None:
            valid_indices = valid_indices * atom_mask
        
        has_clash = []
        for b in range(batch_size):
            has_clash.append(self.compute_has_clash(
                atom_pos[b], 
                asym_id[b],
                indices[b],
                valid_indices[b]
            ))

        has_clash = torch.stack(has_clash)
        return has_clash

class ComputeRankingScore(Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.compute_clash = ComputeClash()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)

    def compute_disorder(
        self,
        plddt: Float['b m'],
        atom_mask: Float['b m'],
        atom_is_molecule_types: Float['b m'],
    )-> Float[' b']:
        
        is_protein_mask = atom_is_molecule_types[..., IS_PROTEIN_INDEX]
        mask = atom_mask * is_protein_mask

        atom_rasa = 1 - plddt

        disorder = ( (atom_rasa > 0.581) * mask ).sum(dim=-1) / ( self.eps + mask.sum(dim=1)) 
        return disorder

    def compute_full_complex_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        molecule_atom_lens: Int['b n'],
        atom_pos: Float['b m 3'],
        atom_mask: Bool['b m'],
        is_molecule_types: Int[f'b n {IS_MOLECULE_TYPES}'],
    ):

        # Section 5.9.3.1
        
        device = atom_pos.device
        batch_size, seq_len= asym_id.shape

        indices = torch.arange(seq_len, device = device)

        indices = repeat(indices, 'n -> b n', b = batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = repeat_consecutive_with_lens(indices, molecule_atom_lens)
        valid_indices = repeat_consecutive_with_lens(valid_indices, molecule_atom_lens)

        expand_indices = indices.unsqueeze(-1).expand(-1, -1, is_molecule_types.shape[-1])
        # broadcast is_molecule_types to atom
        atom_is_molecule_types = torch.gather(is_molecule_types, 1, expand_indices) * valid_indices[..., None]

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=True
        )
        has_clash = self.compute_clash(
            atom_pos, atom_mask, molecule_atom_lens, asym_id, 
        )

        disorder = self.compute_disorder(confidence_score.plddt, atom_mask, atom_is_molecule_types)

        # Section 5.9.3 equation 19
        score = 0.8 * confidence_score.iptm + 0.2 * confidence_score.ptm + 0.5 * disorder - 100 * has_clash

        return score

    def compute_single_chain_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
    ):
        # Section 5.9.3.2
  
        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=False
        )

        score = confidence_score.ptm
        return score

    def compute_interface_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        interface_chains: List,
    ):

        # Section 5.9.3.3

        # interface_chains: List[chain_id_tuple]
        # chain_id_tuple: 
        #  - correspond to the asym_id of one or two chain
        #  - compute R(C) for one chain
        #  - compute 1/2 [R(A) + R(b)] for two chain

        chain_wise_iptm, chain_wise_iptm_mask, unique_chains = self.compute_confidence_score.compute_ptm(
            confidence_head_logits.pae, self.compute_confidence_score.pae_breaks, asym_id, has_frame, compute_chain_wise_iptm=True
        )

        # Section 5.9.3 equation 20
        num_batch = asym_id.shape[0]
        interface_metric = torch.zeros(num_batch).type_as(chain_wise_iptm)

        masked_chain_wise_iptm = chain_wise_iptm * chain_wise_iptm_mask

        iptm_sum = masked_chain_wise_iptm + rearrange(masked_chain_wise_iptm, 'b i j -> b j i')
        iptm_count = chain_wise_iptm_mask.int() + rearrange(chain_wise_iptm_mask.int(), 'b i j -> b j i')
        
        interface_metric = torch.zeros(num_batch).type_as(chain_wise_iptm)
        for b, chains in enumerate(interface_chains):
            num_chains = len(chains)
            for chain in chains:
                idx = unique_chains[b].index(chain)
                interface_metric[b] += iptm_sum[b, idx].sum() / iptm_count[b, idx].sum().clamp(min=1)

            interface_metric[b] /= num_chains

        return interface_metric
            
    def compute_modified_residue_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        atom_mask: Bool['b m'],
        atom_is_modified_residue: Int['b m'],
    ):

        # Section 5.9.3.4

        plddt = self.compute_confidence_score.compute_plddt(
            confidence_head_logits.plddt, 
        )

        mask = atom_is_modified_residue * atom_mask
        plddt_mean =  (plddt * mask).sum(dim=-1) / ( self.eps +  mask.sum(dim=-1)) 

        return plddt_mean

class ComputeModelSelectionScore(Module):

    def __init__(
        self,
        eps: float = 1e-8,
        dist_breaks: Float[' dist_break'] = torch.linspace(2.3125,21.6875,63,),
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0
    ):

        super().__init__()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.dist_breaks = dist_breaks
        self.eps = eps
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

    def compute_gpde(
        self,
        pde_logits: Float['b pde n n'],
        dist_logits: Float['b dist n n '],
        dist_breaks: Float[' dist_break'],
        tok_repr_atm_mask: Bool[' b n'],
    ):        
        """
        
        Section 5.7

        tok_repr_atm_mask: [b n] true if token representation atoms exists
        """
        
        pde = self.compute_confidence_score.compute_pde(
            pde_logits, self.compute_confidence_score.pde_breaks, tok_repr_atm_mask)
        
        dist_logits = rearrange(dist_logits, 'b dist i j -> b i j dist')
        dist_probs = F.softmax(dist_logits, dim=-1)
        contact_mask = dist_breaks < 8.0
        contact_mask = torch.cat([contact_mask, torch.zeros([1], device=dist_logits.device)]).bool()
        contact_prob = einx.where(
            ' dist, b i j dist, -> b i j dist',
            contact_mask, dist_probs, 0.
        ).sum(dim=-1)

        mask = einx.logical_and(
            'b i, b j -> b i j', tok_repr_atm_mask, tok_repr_atm_mask)
        contact_prob = contact_prob * mask

        # Section 5.7 equation 16
        gpde = einsum(contact_prob * pde, 'b i j -> b') / einsum(contact_prob, 'b i j -> b').clamp(min=1.)

        return gpde
    
    def compute_lddt(
        self,
        pred_coords: Float['b m 3'],
        true_coords: Float['b m 3'],
        is_dna: Bool['b m'],
        is_rna: Bool['b m'],
        pairwise_mask: Bool['b m m'],
        coords_mask: Bool['b m'] | None = None,
    ) -> Float['b']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms

        pairwise_mask: boolean tensor indicating atompair for which LDDT is computed
        """
        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        
        lddt = (
            ((0.5 - dist_diff) >=0).float() +
            ((1.0 - dist_diff) >=0).float() +
            ((2.0 - dist_diff) >=0).float() +
            ((4.0 - dist_diff) >=0).float()
        ) / 4.0

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = einx.logical_and('b i, b j -> b i j', is_nucleotide, is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = einx.logical_and('b i, b j -> b i j', coords_mask, coords_mask)
            mask = mask & paired_coords_mask

        mask = mask * pairwise_mask

        # Calculate masked averaging
        lddt_sum = (lddt * mask).sum(dim=(-1, -2))
        lddt_count = mask.sum(dim=(-1, -2))
        lddt_mean = lddt_sum / lddt_count.clamp(min=1)

        return lddt_mean

    def compute_chain_pair_lddt(
        self,
        asym_mask_a: Bool['b m'] | Bool [' m'],
        asym_mask_b: Bool['b m'] | Bool [' m'],
        pred_coords: Float['b m 3'] | Float['m 3'],
        true_coords: Float['b m 3'] | Float['m 3'], 
        is_molecule_types: Int['b m {IS_MOLECULE_TYPES}'] | Int['m {IS_MOLECULE_TYPES}'],
        coords_mask: Bool['b m'] | Bool [' m'] | None = None,
    ) -> Float['b']:
        
        if coords_mask is None:
            coords_mask = torch.ones_like(asym_mask_a)

        if asym_mask_a.ndim == 1:
            args = [asym_mask_a, asym_mask_b, pred_coords, true_coords, is_molecule_types, coords_mask ]
            args = list(
                map(lambda x: x.unsqueeze(0), args)
            )
            asym_mask_a, asym_mask_b, pred_coords, true_coords, is_molecule_types, coords_mask = args
        

        is_dna = is_molecule_types[..., IS_DNA_INDEX]
        is_rna = is_molecule_types[..., IS_RNA_INDEX]
        pairwise_mask = einx.logical_and(
             'b m, b n -> b m n', asym_mask_a, asym_mask_b,
        )

        lddt = self.compute_lddt(
            pred_coords, true_coords, is_dna, is_rna, pairwise_mask, coords_mask
        )

        return lddt
    
    def get_lddt_weight(
        self,
        type_chain_a,
        type_chain_b,
        lddt_type: Literal['interface', 'intra-chain', 'unresolved'],
        is_fine_tuning: bool = False,
    ):
        
        type_mapping = {
            IS_PROTEIN: 'protein',
            IS_DNA: 'DNA',
            IS_RNA: 'RNA',
            IS_LIGAND: 'ligand',
            IS_METAL_ION: 'metal_ion'
        }

        initial_training_dict = {
            'protein-protein': {'interface': 20, 'intra-chain': 20}, 
            'DNA-protein': {'interface': 10}, 
            'RNA-protein': {'interface': 10}, 

            'ligand-protein': {'interface': 10}, 
            'DNA-ligand': {'interface': 5}, 
            'RNA-ligand': {'interface': 5}, 

            'DNA-DNA': {'intra-chain': 4}, 
            'RNA-RNA': {'intra-chain': 16},
            'ligand-ligand': {'intra-chain': 20},
            'metal_ion-metal_ion': {'intra-chain': 10},

            'unresolved': {'unresolved': 10} 
        }

        fine_tuning_dict = {
            'protein-protein': {'interface': 20, 'intra-chain': 20}, 
            'DNA-protein': {'interface': 10},  
            'RNA-protein': {'interface': 2}, 

            'ligand-protein': {'interface': 10}, 
            'DNA-ligand': {'interface': 5}, 
            'RNA-ligand': {'interface': 2}, 

            'DNA-DNA': {'intra-chain': 4}, 
            'RNA-RNA': {'intra-chain': 16},
            'ligand-ligand': {'intra-chain': 20},
            'metal_ion-metal_ion': {'intra-chain': 0},

            'unresolved': {'unresolved': 10} 
        }
        
        weight_dict = fine_tuning_dict if is_fine_tuning else initial_training_dict

        if lddt_type == 'unresolved':
            weight =  weight_dict.get(lddt_type, None).get(lddt_type, None)
            assert weight
            return weight

        interface_type = sorted([type_mapping[type_chain_a], type_mapping[type_chain_b]])
        interface_type = '-'.join(interface_type)
        weight = weight_dict.get(interface_type, {}).get(lddt_type, None)
        assert weight, f"Weight not found for {interface_type} {lddt_type}"
        return weight

    def compute_weighted_lddt(
        self,
        # atom level input
        pred_coords: Float['b m 3'],
        true_coords: Float['b m 3'],
        atom_mask: Bool['b m'] | None,
        # token level input
        asym_id: Int['b n'],
        is_molecule_types: Bool['b n {IS_MOLECULE_TYPES}'],
        molecule_atom_lens: Int['b n'],
        # additional input
        chains_list: List[Tuple[int, int] | Tuple[int]],
        is_fine_tuning: bool = False,
    ):
        
        device = pred_coords.device
        batch_size = pred_coords.shape[0]

        atom_asym_id = repeat_consecutive_with_lens(asym_id, molecule_atom_lens, mask_value=-1)
        atom_is_molecule_types = repeat_consecutive_with_lens(is_molecule_types, molecule_atom_lens)

        weighted_lddt = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            chains = chains_list[b]
            if len(chains) == 2:
                asym_id_a = chains[0]
                asym_id_b = chains[0]
                lddt_type = 'interface'
            elif len(chains) == 1:
                asym_id_a =  asym_id_b = chains[0]
                lddt_type = 'intra-chain'
            else:
                raise Exception(f"Invalid chain list {chains}")

            type_chain_a = get_cid_molecule_type(
                asym_id_a, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )
            type_chain_b = get_cid_molecule_type(
                asym_id_b, atom_asym_id[b], atom_is_molecule_types[b], return_one_hot=False
            )

            lddt_weight = self.get_lddt_weight(
                type_chain_a, type_chain_b, lddt_type, is_fine_tuning
            )

            asym_mask_a = atom_asym_id[b] == asym_id_a
            asym_mask_b = atom_asym_id[b] == asym_id_b

            lddt = self.compute_chain_pair_lddt(
                asym_mask_a, asym_mask_b, 
                pred_coords[b], true_coords[b], 
                atom_is_molecule_types[b], atom_mask[b],
            )

            weighted_lddt[b] = lddt_weight * lddt

        return weighted_lddt