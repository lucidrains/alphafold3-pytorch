from __future__ import annotations
from typing import List, NamedTuple
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
)

from einops import rearrange, repeat

def exists(v):
    return v is not None

class ConfidenceScore(NamedTuple):
    plddt: Float['b n']
    ptm: Float[' b']
    iptm: Float[' b'] | None

class ComputeConfidenceScore(Module):
    
    @typecheck
    def __init__(
        self,
        pae_breaks: Float[' nbreak'] = torch.arange(0, 31.5, 0.5),
        eps: float = 1e-8
    ):

        super().__init__()
        self.pae_breaks = pae_breaks
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
        logits: Float['b c m'],
    )->Float['b m']:
        
        logits = rearrange(logits, 'b c m -> b m c')
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=logits.device)
        probs = F.softmax(logits, dim=-1)

        predicted_lddt = torch.sum(probs * bin_centers[None, None, : ], dim=-1)
        return predicted_lddt * 100

    @typecheck
    def compute_ptm(
        self,
        logits: Float['b c n n '],
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
        predicted_tm_term = torch.sum(probs * tm_per_bin[:, None, None, :], dim=-1)


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
        for b, chains in enumerate(interface_chains):
            num_chains = len(chains)
            for chain in chains:
                idx = unique_chains[b].index(chain)
                if chain_wise_iptm_mask[idx].sum() == 0:
                    continue
                else:
                    interface_metric[b] += (chain_wise_iptm[idx] * chain_wise_iptm_mask[idx]).sum() / chain_wise_iptm_mask[idx].sum()

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
