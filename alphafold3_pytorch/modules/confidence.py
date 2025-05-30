from __future__ import annotations

from math import pi, sqrt
from functools import wraps

import torch
from torch import nn
from torch import tensor

from torch.nn import (
    Module,
)

from beartype.typing import (
    List,
    Literal,
    NamedTuple,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    typecheck,
    checkpoint,
)

from alphafold3_pytorch.modules.diffusion import *
from alphafold3_pytorch.modules.pairformer import (
    PairformerStack,
)

from alphafold3_pytorch.inputs import (
    IS_PROTEIN_INDEX,
    IS_DNA_INDEX,
    IS_RNA_INDEX,
    IS_PROTEIN,
    IS_DNA,
    IS_RNA,
    IS_LIGAND,
    IS_METAL_ION,
    BatchedAtomInput,
)

from alphafold3_pytorch.utils.helpers import *
from alphafold3_pytorch.modules.basic_models import *

from alphafold3_pytorch.common.biomolecule import (
    get_residue_constants,
)

from alphafold3_pytorch.utils.model_utils import distance_to_dgram

# other external libs

from loguru import logger

from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder

# einstein notation related

import einx
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# distogram head

class DistogramHead(Module):

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        num_dist_bins = 64,
        dim_atom = 128,
        atom_resolution = False,
        checkpoint = False,
    ):
        super().__init__()

        self.to_distogram_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_dist_bins),
            Rearrange('b ... l -> b l ...')
        )

        # atom resolution
        # for now, just embed per atom distances, sum to atom features, project to pairwise dimension

        self.atom_resolution = atom_resolution

        if atom_resolution:
            self.atom_feats_to_pairwise = LinearNoBiasThenOuterSum(dim_atom, dim_pairwise)

        # checkpointing

        self.checkpoint = checkpoint

        # tensor typing

        self.da = dim_atom

    @typecheck
    def to_layers(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        if self.atom_resolution:
            assert exists(molecule_atom_lens)
            assert exists(atom_feats)

            pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

        logits = self.to_distogram_logits(symmetrize(pairwise_repr))

        return logits

    @typecheck
    def to_checkpointed_layers(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the checkpointed distogram logits.

        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The checkpointed distogram logits.
        """
        wrapped_layers = []
        inputs = (pairwise_repr, molecule_atom_lens, atom_feats)

        def atom_resolution_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs

                assert exists(molecule_atom_lens)
                assert exists(atom_feats)

                pairwise_repr = batch_repeat_interleave_pairwise(pairwise_repr, molecule_atom_lens)

                pairwise_repr = pairwise_repr + fn(atom_feats)
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        def distogram_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, molecule_atom_lens, atom_feats = inputs
                pairwise_repr = fn(symmetrize(pairwise_repr))
                return pairwise_repr, molecule_atom_lens, atom_feats

            return inner

        if self.atom_resolution:
            wrapped_layers.append(atom_resolution_wrapper(self.atom_feats_to_pairwise))
        wrapped_layers.append(distogram_wrapper(self.to_distogram_logits))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        logits, *_ = inputs
        return logits

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],  # type: ignore
        molecule_atom_lens: Int["b n"] | None = None,  # type: ignore
        atom_feats: Float["b m {self.da}"] | None = None,  # type: ignore
    ) -> Float["b l n n"] | Float["b l m m"]:  # type: ignore
        """Compute the distogram logits.
        
        :param pairwise_repr: The pairwise representation tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param atom_feats: The atom features tensor.
        :return: The distogram logits.
        """
        # going through the layers

        if should_checkpoint(self, pairwise_repr):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        logits = to_layers_fn(
            pairwise_repr=pairwise_repr,
            molecule_atom_lens=molecule_atom_lens,
            atom_feats=atom_feats,
        )

        return logits

# confidence head

class ConfidenceHeadLogits(NamedTuple):
    pae: Float['b pae n n'] |  None
    pde: Float['b pde n n']
    plddt: Float['b plddt m']
    resolved: Float['b 2 m']

class Alphafold3Logits(NamedTuple):
    pae: Float['b pae n n'] |  None
    pde: Float['b pde n n']
    plddt: Float['b plddt m']
    resolved: Float['b 2 m']
    distance: Float['b dist m m'] | Float['b dist n n'] | None

class ConfidenceHead(Module):
    """ Algorithm 31 """

    @typecheck
    def __init__(
        self,
        *,
        dim_single_inputs,
        dim_atom = 128,
        atompair_dist_bins: List[float],
        dim_single = 384,
        dim_pairwise = 128,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        pairformer_depth = 4,
        pairformer_kwargs: dict = dict(),
        checkpoint = False
    ):
        super().__init__()

        atompair_dist_bins = tensor(atompair_dist_bins)

        self.register_buffer('atompair_dist_bins', atompair_dist_bins)

        num_dist_bins = atompair_dist_bins.shape[-1]
        self.num_dist_bins = num_dist_bins

        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, dim_pairwise)
        self.single_inputs_to_pairwise = LinearNoBiasThenOuterSum(dim_single_inputs, dim_pairwise)

        # pairformer stack

        self.pairformer_stack = PairformerStack(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            depth = pairformer_depth,
            checkpoint = checkpoint,
            **pairformer_kwargs
        )

        # to predictions

        self.to_pae_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pae_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_pde_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_pde_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_plddt_logits = nn.Sequential(
            LinearNoBias(dim_single, num_plddt_bins),
            Rearrange('b ... l -> b l ...')
        )

        self.to_resolved_logits = nn.Sequential(
            LinearNoBias(dim_single, 2),
            Rearrange('b ... l -> b l ...')
        )

        # atom resolution

        self.atom_feats_to_single = LinearNoBias(dim_atom, dim_single)

        # tensor typing

        self.da = dim_atom

    @typecheck
    def forward(
        self,
        *,
        single_inputs_repr: Float["b n dsi"],
        single_repr: Float["b n ds"],
        pairwise_repr: Float["b n n dp"],
        pred_atom_pos: Float["b m 3"],
        atom_feats: Float["b m {self.da}"],
        molecule_atom_indices: Int["b n"],
        molecule_atom_lens: Int["b n"],
        mask: Bool["b n"] | None = None,
        return_pae_logits: bool = True,
    ) -> ConfidenceHeadLogits:
        """Compute the confidence head logits.

        :param single_inputs_repr: The single inputs representation tensor.
        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param pred_atom_pos: The predicted atom positions tensor.
        :param atom_feats: The atom features tensor.
        :param molecule_atom_indices: The molecule atom indices tensor.
        :param molecule_atom_lens: The molecule atom lengths tensor.
        :param mask: The mask tensor.
        :param return_pae_logits: Whether to return the predicted aligned error (PAE) logits.
        :return: The confidence head logits.
        """

        pairwise_repr = pairwise_repr + self.single_inputs_to_pairwise(single_inputs_repr)

        # pluck out the representative atoms for non-atomic resolution confidence head outputs

        # pred_molecule_pos = einx.get_at('b [m] c, b n -> b n c', pred_atom_pos, molecule_atom_indices)

        molecule_atom_indices = repeat(
            molecule_atom_indices, "b n -> b n c", c=pred_atom_pos.shape[-1]
        )
        pred_molecule_pos = pred_atom_pos.gather(1, molecule_atom_indices)

        # interatomic distances - embed and add to pairwise

        intermolecule_dist = torch.cdist(pred_molecule_pos, pred_molecule_pos, p=2)

        dist_bin_indices = distance_to_dgram(
            intermolecule_dist, self.atompair_dist_bins, return_labels=True
        )
        pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

        # pairformer stack

        single_repr, pairwise_repr = self.pairformer_stack(
            single_repr=single_repr, pairwise_repr=pairwise_repr, mask=mask
        )

        # handle atom level resolution

        atom_single_repr = batch_repeat_interleave(single_repr, molecule_atom_lens)

        atom_single_repr = atom_single_repr + self.atom_feats_to_single(atom_feats)

        # to logits

        pde_logits = self.to_pde_logits(symmetrize(pairwise_repr))

        plddt_logits = self.to_plddt_logits(atom_single_repr)
        resolved_logits = self.to_resolved_logits(atom_single_repr)

        # they only incorporate pae at some stage of training

        pae_logits = None

        if return_pae_logits:
            pae_logits = self.to_pae_logits(pairwise_repr)

        # return all logits

        return ConfidenceHeadLogits(pae_logits, pde_logits, plddt_logits, resolved_logits)

# more confidence / clash calculations

class ConfidenceScore(NamedTuple):
    """The ConfidenceScore class."""

    plddt: Float["b m"]
    ptm: Float[" b"]
    iptm: Float[" b"] | None


class ComputeConfidenceScore(Module):
    """Compute confidence score."""

    @typecheck
    def __init__(
        self,
        pae_breaks: Float[" pae_break"] = torch.arange(0, 31.5, 0.5),
        pde_breaks: Float[" pde_break"] = torch.arange(0, 31.5, 0.5),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer("pae_breaks", pae_breaks)
        self.register_buffer("pde_breaks", pde_breaks)

    @typecheck
    def _calculate_bin_centers(
        self,
        breaks: Float[" breaks"],
    ) -> Float[" breaks+1"]:
        """Calculate bin centers from bin edges.

        :param breaks: [num_bins -1] bin edges
        :return: bin_centers: [num_bins] bin centers
        """

        step = breaks[1] - breaks[0]

        bin_centers = breaks + step / 2
        last_bin_center = breaks[-1] + step

        bin_centers = torch.concat([bin_centers, last_bin_center.unsqueeze(0)])

        return bin_centers

    @typecheck
    def forward(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],
        has_frame: Bool["b n"],
        ptm_residue_weight: Float["b n"] | None = None,
        multimer_mode: bool = True,
    ) -> ConfidenceScore:
        """Main function to compute confidence score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param ptm_residue_weight: [b n] weight of each residue
        :param multimer_mode: bool
        :return: Confidence score
        """
        plddt = self.compute_plddt(confidence_head_logits.plddt)

        # Section 5.9.1 equation 17
        ptm = self.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=False,
        )

        iptm = None

        if multimer_mode:
            # Section 5.9.2 equation 18
            iptm = self.compute_ptm(
                confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=True,
            )

        confidence_score = ConfidenceScore(plddt=plddt, ptm=ptm, iptm=iptm)
        return confidence_score

    @typecheck
    def compute_plddt(
        self,
        logits: Float["b plddt m"],
    ) -> Float["b m"]:
        """Compute plDDT from logits.

        :param logits: [b c m] logits
        :return: [b m] plDDT
        """
        logits = rearrange(logits, "b plddt m -> b m plddt")
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 1.0, bin_width, dtype=logits.dtype, device=logits.device
        )
        probs = F.softmax(logits, dim=-1)

        predicted_lddt = einsum(probs, bin_centers, "b m plddt, plddt -> b m")
        return predicted_lddt * 100

    @typecheck
    def compute_ptm(
        self,
        pae_logits: Float["b pae n n"],
        asym_id: Int["b n"],
        has_frame: Bool["b n"],
        residue_weights: Float["b n"] | None = None,
        interface: bool = False,
        compute_chain_wise_iptm: bool = False,
    ) -> Float[" b"] | Tuple[Float["b chains chains"], Bool["b chains chains"], Int["b chains"]]:

        """Compute pTM from logits.

        :param logits: [b c n n] logits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param residue_weights: [b n] weight of each residue
        :param interface: bool
        :param compute_chain_wise_iptm: bool
        :return: pTM
        """
        if not exists(residue_weights):
            residue_weights = torch.ones_like(has_frame)

        residue_weights = residue_weights * has_frame

        num_batch, *_, num_res, device = *pae_logits.shape, pae_logits.device

        pae_logits = rearrange(pae_logits, "b c i j -> b i j c")

        bin_centers = self._calculate_bin_centers(self.pae_breaks)

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
        probs = F.softmax(pae_logits, dim=-1)

        # E_distances tm(distance).
        predicted_tm_term = einsum(probs, tm_per_bin, "b i j pae, b pae -> b i j ")

        if compute_chain_wise_iptm:
            # chain_wise_iptm[b, i, j]: iptm of chain i and chain j in batch b

            # get the max num_chains across batch
            unique_chains = [torch.unique(asym).tolist() for asym in asym_id]
            max_chains = max(len(chains) for chains in unique_chains)

            chain_wise_iptm = torch.zeros(
                (num_batch, max_chains, max_chains), device=device
            )
            chain_wise_iptm_mask = torch.zeros_like(chain_wise_iptm).bool()

            for b in range(num_batch):
                for i, chain_i in enumerate(unique_chains[b]):
                    for j, chain_j in enumerate(unique_chains[b]):
                        if chain_i != chain_j:
                            mask_i = (asym_id[b] == chain_i)
                            mask_j = (asym_id[b] == chain_j)

                            pair_mask = einx.multiply('i, j -> i j', mask_i, mask_j)

                            pair_residue_weights = pair_mask * einx.multiply(
                                "... i, ... j -> ... i j", residue_weights[b], residue_weights[b]
                            )

                            if pair_residue_weights.sum() == 0:
                                # chain i or chain j does not have any valid frame
                                continue

                            normed_residue_mask = pair_residue_weights / (
                                self.eps
                                + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
                            )

                            masked_predicted_tm_term = predicted_tm_term[b] * pair_mask

                            per_alignment = torch.sum(
                                masked_predicted_tm_term * normed_residue_mask, dim=-1
                            )
                            weighted_argmax = (residue_weights[b] * per_alignment).argmax()
                            chain_wise_iptm[b, i, j] = per_alignment[weighted_argmax]
                            chain_wise_iptm_mask[b, i, j] = True

            return chain_wise_iptm, chain_wise_iptm_mask, torch.tensor(unique_chains)

        else:
            pair_mask = torch.ones(size=(num_batch, num_res, num_res), device=device).bool()
            if interface:
                pair_mask *= einx.not_equal('b i, b j -> b i j', asym_id, asym_id)

            predicted_tm_term *= pair_mask

            pair_residue_weights = pair_mask * einx.multiply('b i, b j -> b i j', residue_weights, residue_weights)

            normed_residue_mask = pair_residue_weights / (
                self.eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
            )

            per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
            weighted_argmax = (residue_weights * per_alignment).argmax(dim=-1)
            return per_alignment[torch.arange(num_batch), weighted_argmax]

    @typecheck
    def compute_pde(
        self,
        pde_logits: Float["b pde n n"],
        tok_repr_atm_mask: Bool["b n"],
    ) -> Float["b n n"]:
        """Compute PDE from logits."""

        pde_logits = rearrange(pde_logits, "b pde i j -> b i j pde")
        bin_centers = self._calculate_bin_centers(self.pde_breaks)
        probs = F.softmax(pde_logits, dim=-1)

        pde = einsum(probs, bin_centers, "b i j pde, pde -> b i j")

        mask = to_pairwise_mask(tok_repr_atm_mask)

        pde = pde * mask
        return pde


class ComputeClash(Module):
    """Compute clash score."""

    def __init__(
        self,
        atom_clash_dist: float = 1.1,
        chain_clash_count: int = 100,
        chain_clash_ratio: float = 0.5,
    ):
        super().__init__()
        self.atom_clash_dist = atom_clash_dist
        self.chain_clash_count = chain_clash_count
        self.chain_clash_ratio = chain_clash_ratio

    @typecheck
    def compute_has_clash(
        self,
        atom_pos: Float["m 3"],
        asym_id: Int[" n"],
        indices: Int[" m"],
        valid_indices: Bool[" m"],
    ) -> Bool[""]:
        """Compute if there is a clash in the chain.

        :param atom_pos: [m 3] atom positions
        :param asym_id: [n] asym_id of each residue
        :param indices: [m] indices
        :param valid_indices: [m] valid indices
        :return: [1] has_clash
        """

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

                chain_pair_dist = torch.cdist(atom_pos[mask_i], atom_pos[mask_j])
                chain_pair_clash = chain_pair_dist < self.atom_clash_dist
                clashes = chain_pair_clash.sum()
                has_clash = (clashes > self.chain_clash_count) or (
                    clashes / min(chain_i_len, chain_j_len) > self.chain_clash_ratio
                )

                if has_clash:
                    return torch.tensor(True, dtype=torch.bool, device=atom_pos.device)

        return torch.tensor(False, dtype=torch.bool, device=atom_pos.device)

    @typecheck
    def forward(
        self,
        atom_pos: Float["b m 3"] | Float["m 3"],
        atom_mask: Bool["b m"] | Bool[" m"],
        molecule_atom_lens: Int["b n"] | Int[" n"],
        asym_id: Int["b n"] | Int[" n"],
    ) -> Bool[" b"]:

        """Compute if there is a clash in the chain.

        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param molecule_atom_lens: [b n] molecule atom lens
        :param asym_id: [b n] asym_id of each residue
        :return: [b] has_clash
        """

        if atom_pos.ndim == 2:
            atom_pos = atom_pos.unsqueeze(0)
            molecule_atom_lens = molecule_atom_lens.unsqueeze(0)
            asym_id = asym_id.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)

        device = atom_pos.device
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        if exists(atom_mask):
            valid_indices = valid_indices * atom_mask

        has_clash = []
        for b in range(batch_size):
            has_clash.append(
                self.compute_has_clash(atom_pos[b], asym_id[b], indices[b], valid_indices[b])
            )

        has_clash = torch.stack(has_clash)
        return has_clash


class ComputeRankingScore(Module):
    """Compute ranking score."""

    def __init__(
        self,
        eps: float = 1e-8,
        score_iptm_weight: float = 0.8,
        score_ptm_weight: float = 0.2,
        score_disorder_weight: float = 0.5,
    ):
        super().__init__()
        self.eps = eps
        self.compute_clash = ComputeClash()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)

        self.score_iptm_weight = score_iptm_weight
        self.score_ptm_weight = score_ptm_weight
        self.score_disorder_weight = score_disorder_weight

    @typecheck
    def compute_disorder(
        self,
        plddt: Float["b m"],
        atom_mask: Bool["b m"],
        atom_is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"],
    ) -> Float[" b"]:
        """Compute disorder score.

        :param plddt: [b m] plddt
        :param atom_mask: [b m] atom mask
        :param atom_is_molecule_types: [b m 2] atom is molecule types
        :return: [b] disorder
        """
        is_protein_mask = atom_is_molecule_types[..., IS_PROTEIN_INDEX]
        mask = atom_mask * is_protein_mask

        atom_rasa = 1.0 - plddt

        disorder = ((atom_rasa > 0.581) * mask).sum(dim=-1) / (self.eps + mask.sum(dim=1))
        return disorder

    @typecheck
    def compute_full_complex_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],
        has_frame: Bool["b n"],
        molecule_atom_lens: Int["b n"],
        atom_pos: Float["b m 3"],
        atom_mask: Bool["b m"],
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],
        return_confidence_score: bool = False,
    ) -> Float[" b"] | Tuple[Float[" b"], Tuple[ConfidenceScore, Bool[" b"]]]:

        """Compute full complex metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param molecule_atom_lens: [b n] molecule atom lens
        :param atom_pos: [b m 3] atom positions
        :param atom_mask: [b m] atom mask
        :param is_molecule_types: [b n 2] is_molecule_types
        :return: [b] score
        """

        # Section 5.9.3.1

        device = atom_pos.device
        batch_size, seq_len = asym_id.shape

        indices = torch.arange(seq_len, device=device)

        indices = repeat(indices, "n -> b n", b=batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        valid_indices = batch_repeat_interleave(valid_indices, molecule_atom_lens)

        # broadcast is_molecule_types to atom

        # einx.get_at('b [n] is_type, b m -> b m is_type', is_molecule_types, indices)

        indices = repeat(indices, "b m -> b m is_type", is_type=is_molecule_types.shape[-1])
        atom_is_molecule_types = is_molecule_types.gather(1, indices) * valid_indices[..., None]

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=True
        )
        has_clash = self.compute_clash(
            atom_pos,
            atom_mask,
            molecule_atom_lens,
            asym_id,
        )

        disorder = self.compute_disorder(confidence_score.plddt, atom_mask, atom_is_molecule_types)

        # Section 5.9.3 equation 19
        weighted_score = (
            confidence_score.iptm * self.score_iptm_weight
            + confidence_score.ptm * self.score_ptm_weight
            + disorder * self.score_disorder_weight
            - 100 * has_clash
        )

        if not return_confidence_score:
            return weighted_score

        return weighted_score, (confidence_score, has_clash)

    @typecheck
    def compute_single_chain_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],
        has_frame: Bool["b n"],
    ) -> Float[" b"]:

        """Compute single chain metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :return: [b] score
        """

        # Section 5.9.3.2

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=False
        )

        score = confidence_score.ptm
        return score

    @typecheck
    def compute_interface_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int["b n"],
        has_frame: Bool["b n"],
        interface_chains: List,
    ) -> Float[" b"]:
        """Compute interface metric.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param asym_id: [b n] asym_id of each residue
        :param has_frame: [b n] has_frame of each residue
        :param interface_chains: List
        :return: [b] score
        """

        batch = asym_id.shape[0]

        # Section 5.9.3.3

        # interface_chains: List[chain_id_tuple]
        # chain_id_tuple:
        #  - correspond to the asym_id of one or two chain
        #  - compute R(C) for one chain
        #  - compute 1/2 [R(A) + R(b)] for two chain

        (
            chain_wise_iptm,
            chain_wise_iptm_mask,
            unique_chains,
        ) = self.compute_confidence_score.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, compute_chain_wise_iptm=True
        )

        # Section 5.9.3 equation 20
        interface_metric = torch.zeros(batch).type_as(chain_wise_iptm)

        # R(c) = mean(Mij) restricted to i = c or j = c
        masked_chain_wise_iptm = chain_wise_iptm * chain_wise_iptm_mask
        iptm_sum = masked_chain_wise_iptm + rearrange(masked_chain_wise_iptm, "b i j -> b j i")
        iptm_count = chain_wise_iptm_mask.int() + rearrange(
            chain_wise_iptm_mask.int(), "b i j -> b j i"
        )

        for b, chains in enumerate(interface_chains):
            for chain in chains:
                idx = unique_chains[b].tolist().index(chain)
                interface_metric[b] += iptm_sum[b, idx].sum() / iptm_count[b, idx].sum().clamp(
                    min=1
                )
            interface_metric[b] /= len(chains)
        return interface_metric

    @typecheck
    def compute_modified_residue_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        atom_mask: Bool["b m"],
        atom_is_modified_residue: Int["b m"],
    ) -> Float[" b"]:
        """Compute modified residue score.

        :param confidence_head_logits: ConfidenceHeadLogits
        :param atom_mask: [b m] atom mask
        :param atom_is_modified_residue: [b m] atom is modified residue
        :return: [b] score
        """

        # Section 5.9.3.4

        plddt = self.compute_confidence_score.compute_plddt(
            confidence_head_logits.plddt,
        )

        mask = atom_is_modified_residue * atom_mask
        plddt_mean = masked_average(plddt, mask, dim=-1, eps=self.eps)

        return plddt_mean


# model selection

@typecheck
def get_cid_molecule_type(
    cid: int,
    asym_id: Int[" n"],
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],
    return_one_hot: bool = False,
) -> int | Bool[f" {IS_MOLECULE_TYPES}"]:
    """Get the (majority) molecule type for where `asym_id == cid`.

    NOTE: Several PDB chains contain multiple molecule types, so
    we must choose a single molecule type for the chain. We choose
    the molecule type that is most common (i.e., the mode) in the chain.

    :param cid: chain id
    :param asym_id: [n] asym_id of each residue
    :param is_molecule_types: [n 2] is_molecule_types
    :param return_one_hot: return one hot
    :return: molecule type
    """

    cid_is_molecule_types = is_molecule_types[asym_id == cid]

    molecule_types = cid_is_molecule_types.int().argmax(1)
    molecule_type_mode = molecule_types.mode()
    molecule_type = cid_is_molecule_types[molecule_type_mode.indices.item()]

    if not return_one_hot:
        molecule_type = molecule_type_mode.values.item()
    return molecule_type


@typecheck
def protein_structure_from_feature(
    asym_id: Int[" n"],
    molecule_ids: Int[" n"],
    molecule_atom_lens: Int[" n"],
    atom_pos: Float["m 3"],
    atom_mask: Bool[" m"],
) -> Structure:

    """Create structure for unresolved proteins.

    :param atom_mask: True for valid atoms, False for missing/padding atoms
    return: A Biopython Structure object
    """

    num_atom = atom_pos.shape[0]
    num_res = molecule_ids.shape[0]

    residue_constants = get_residue_constants(res_chem_index=IS_PROTEIN)

    molecule_atom_indices = exclusive_cumsum(molecule_atom_lens)

    builder = StructureBuilder()
    builder.init_structure("structure")
    builder.init_model(0)

    cur_cid = None
    cur_res_id = None

    for res_idx in range(num_res):
        num_atom = molecule_atom_lens[res_idx]
        cid = str(asym_id[res_idx].detach().cpu().item())

        if cid != cur_cid:
            builder.init_chain(cid)
            builder.init_seg(segid=" ")
            cur_cid = cid
            cur_res_id = 0

        restype = residue_constants.restypes[molecule_ids[res_idx]]
        resname = residue_constants.restype_1to3[restype]
        atom_names = residue_constants.restype_name_to_compact_atom_names[resname]
        atom_names = list(filter(lambda x: x, atom_names))
        # assume residues for unresolved protein are standard
        assert (
            len(atom_names) == num_atom
        ), f"Molecule atom lens {num_atom} doesn't match with residue constant {len(atom_names)}"

        # skip if all atom of the residue is missing
        atom_idx_offset = molecule_atom_indices[res_idx]
        if not torch.any(atom_mask[atom_idx_offset : atom_idx_offset + num_atom]):
            continue

        builder.init_residue(resname, " ", cur_res_id + 1, " ")
        cur_res_id += 1

        for atom_idx in range(num_atom):
            if not atom_mask[atom_idx]:
                continue

            atom_coord = atom_pos[atom_idx + atom_idx_offset].detach().cpu().numpy()
            atom_name = atom_names[atom_idx]
            builder.init_atom(
                name=atom_name,
                coord=atom_coord,
                b_factor=1.0,
                occupancy=1.0,
                fullname=atom_name,
                altloc=" ",
                # only N, C, O in restype_name_to_compact_atom_names for protein
                # so just take the first char
                element=atom_name[0],
            )

    return builder.get_structure()

Sample = Tuple[Float["b m 3"], Float["b pde n n"], Float["b m"], Float["b dist n n"]]
ScoredSample = Tuple[int, Float["b m 3"], Float["b m"], Float[" b"], Float[" b"]]

class ScoreDetails(NamedTuple):
    best_gpde_index: int
    best_lddt_index: int
    score: Float[' b']
    scored_samples: List[ScoredSample]

class ComputeModelSelectionScore(Module):
    """Compute model selection score."""

    INITIAL_TRAINING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 10},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 5},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "DNA-RNA": {"interface": 4, "intra-chain": 4},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "metal_ion-metal_ion": {"interface": 10, "intra-chain": 10},
        "unresolved": {"unresolved": 10},
    }

    FINETUNING_DICT = {
        "protein-protein": {"interface": 20, "intra-chain": 20},
        "DNA-protein": {"interface": 10},
        "RNA-protein": {"interface": 2},
        "ligand-protein": {"interface": 10},
        "DNA-ligand": {"interface": 5},
        "RNA-ligand": {"interface": 2},
        "DNA-DNA": {"interface": 4, "intra-chain": 4},
        "RNA-RNA": {"interface": 16, "intra-chain": 16},
        "DNA-RNA": {"interface": 4, "intra-chain": 4},
        "ligand-ligand": {"interface": 20, "intra-chain": 20},
        "metal_ion-metal_ion": {"interface": 0, "intra-chain": 0},
        "unresolved": {"unresolved": 10},
    }

    TYPE_MAPPING = {
        IS_PROTEIN: "protein",
        IS_DNA: "DNA",
        IS_RNA: "RNA",
        IS_LIGAND: "ligand",
        IS_METAL_ION: "metal_ion",
    }

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8,
        dist_breaks: Float[" dist_break"] = torch.linspace(2, 22, 63),
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0,
        contact_mask_threshold: float = 8.0,
        is_fine_tuning: bool = False,
        weight_dict_config: dict = None,
        fibonacci_sphere_n = 200, # more points equal better approximation at cost of compute
    ):
        super().__init__()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.eps = eps
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff
        self.contact_mask_threshold = contact_mask_threshold
        self.is_fine_tuning = is_fine_tuning
        self.weight_dict_config = weight_dict_config

        self.register_buffer("dist_breaks", dist_breaks)
        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

        # for rsa calculation

        atom_type_radii = tensor([
            1.65,    # 0 - nitrogen
            1.87,    # 1 - carbon alpha
            1.76,    # 2 - carbon
            1.4,     # 3 - oxygen
            1.8,     # 4 - side atoms
            1.4      # 5 - water
        ])

        self.atom_type_index = dict(
            N = 0,
            CA = 1,
            C = 2,
            O = 3
        ) # rest go to 4 (side chain atom)

        self.register_buffer('atom_radii', atom_type_radii, persistent = False)

        # constitute the fibonacci sphere

        num_surface_dots = fibonacci_sphere_n * 2 + 1
        golden_ratio = 1. + sqrt(5.) / 2
        weight = (4. * pi) / num_surface_dots

        arange = torch.arange(-fibonacci_sphere_n, fibonacci_sphere_n + 1, device = self.device) # for example, N = 3 -> [-3, -2, -1, 0, 1, 2, 3]

        lat = torch.asin((2. * arange) / num_surface_dots)
        lon = torch.fmod(arange, golden_ratio) * 2 * pi / golden_ratio

        # ein:
        # sd - surface dots
        # c - coordinate (3)
        # i, j - source and target atom

        unit_surface_dots: Float['sd 3'] = torch.stack((
            lon.sin() * lat.cos(),
            lon.cos() * lat.cos(),
            lat.sin()
        ), dim = -1)

        self.register_buffer('unit_surface_dots', unit_surface_dots)
        self.surface_weight = weight

    @property
    def device(self):
        return self.atom_radii.device

    @typecheck
    def compute_gpde(
        self,
        pde_logits: Float["b pde n n"],
        dist_logits: Float["b dist n n"],
        dist_breaks: Float[" dist_break"],
        tok_repr_atm_mask: Bool["b n"],
    ) -> Float[" b"]:
        """Compute global PDE following Section 5.7 of the AF3 supplement.

        :param pde_logits: [b pde n n] PDE logits
        :param dist_logits: [b dist n n] distance logits
        :param dist_breaks: [dist_break] distance breaks
        :param tok_repr_atm_mask: [b n] true if token representation atoms exists
        :return: [b] global PDE
        """

        dtype = pde_logits.dtype

        pde = self.compute_confidence_score.compute_pde(pde_logits, tok_repr_atm_mask)

        dist_logits = rearrange(dist_logits, "b dist i j -> b i j dist")
        dist_probs = F.softmax(dist_logits, dim=-1)

        # for distances greater than the last breaks
        dist_breaks = F.pad(dist_breaks.float(), (0, 1), value=1e6).type(dtype)
        contact_mask = dist_breaks < self.contact_mask_threshold

        contact_prob = einx.where(
            " dist, b i j dist, -> b i j dist", contact_mask, dist_probs, 0.0
        ).sum(dim=-1)

        mask = to_pairwise_mask(tok_repr_atm_mask)
        contact_prob = contact_prob * mask

        # Section 5.7 equation 16
        gpde = masked_average(pde, contact_prob, dim=(-1, -2))

        return gpde

    @typecheck
    def compute_lddt(
        self,
        pred_coords: Float["b m 3"],
        true_coords: Float["b m 3"],
        is_dna: Bool["b m"],
        is_rna: Bool["b m"],
        pairwise_mask: Bool["b m m"],
        coords_mask: Bool["b m"] | None = None,
    ) -> Float[" b"]:
        """Compute lDDT.

        :param pred_coords: predicted coordinates
        :param true_coords: true coordinates
        :param is_dna: boolean tensor indicating DNA atoms
        :param is_rna: boolean tensor indicating RNA atoms
        :param pairwise_mask: boolean tensor indicating atompair for which LDDT is computed
        :param coords_mask: boolean tensor indicating valid atoms
        :return: lDDT
        """

        dtype = pred_coords.dtype
        atom_seq_len, device = pred_coords.shape[1], pred_coords.device

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        lddt = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        lddt = (lddt >= 0).type(dtype).mean(dim=-1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff,
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(atom_seq_len, dtype=torch.bool, device=device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        mask = mask * pairwise_mask

        # Calculate masked averaging
        lddt_mean = masked_average(lddt, mask, dim=(-1, -2))

        return lddt_mean

    @typecheck
    def compute_chain_pair_lddt(
        self,
        asym_mask_a: Bool["b m"] | Bool[" m"],
        asym_mask_b: Bool["b m"] | Bool[" m"],
        pred_coords: Float["b m 3"] | Float["m 3"],
        true_coords: Float["b m 3"] | Float["m 3"],
        is_molecule_types: Bool[f"b m {IS_MOLECULE_TYPES}"] | Bool[f"m {IS_MOLECULE_TYPES}"],
        coords_mask: Bool["b m"] | Bool[" m"] | None = None,
    ) -> Float[" b"]:
        """Compute the plDDT between atoms marked by `asym_mask_a` and `asym_mask_b`.

        :param asym_mask_a: [b m] asym_mask_a
        :param asym_mask_b: [b m] asym_mask_b
        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param is_molecule_types: [b m 2] is_molecule_types
        :param coords_mask: [b m] coords_mask
        :return: [b] lddt
        """

        if not exists(coords_mask):
            coords_mask = torch.ones_like(asym_mask_a)

        if asym_mask_a.ndim == 1:
            (
                asym_mask_a,
                asym_mask_b,
                pred_coords,
                true_coords,
                is_molecule_types,
                coords_mask,
            ) = map(lambda t: rearrange(t, '... -> 1 ...'), (
                asym_mask_a,
                asym_mask_b,
                pred_coords,
                true_coords,
                is_molecule_types,
                coords_mask,
            ))

        is_dna = is_molecule_types[..., IS_DNA_INDEX]
        is_rna = is_molecule_types[..., IS_RNA_INDEX]
        pairwise_mask = to_pairwise_mask(asym_mask_a)

        lddt = self.compute_lddt(
            pred_coords, true_coords, is_dna, is_rna, pairwise_mask, coords_mask
        )

        return lddt

    @typecheck
    def get_lddt_weight(
        self,
        type_chain_a: int,
        type_chain_b: int,
        lddt_type: Literal["interface", "intra-chain", "unresolved"],
        is_fine_tuning: bool = None,
    ) -> int:
        """Get a specified lDDT weight.

        :param type_chain_a: type of chain a
        :param type_chain_b: type of chain b
        :param lddt_type: lDDT type
        :param is_fine_tuning: is fine tuning
        :return: lDDT weight
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        if lddt_type == "unresolved":
            weight = weight_dict.get(lddt_type, {}).get(lddt_type, None)
            assert weight
            return weight

        interface_type = sorted([self.TYPE_MAPPING[type_chain_a], self.TYPE_MAPPING[type_chain_b]])
        interface_type = "-".join(interface_type)
        weight = weight_dict.get(interface_type, {}).get(lddt_type, None)
        assert weight, f"Weight not found for {interface_type} {lddt_type}"
        return weight

    @typecheck
    def compute_weighted_lddt(
        self,
        # atom level input
        pred_coords: Float["b m 3"],
        true_coords: Float["b m 3"],
        atom_mask: Bool["b m"] | None,
        # token level input
        asym_id: Int["b n"],
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"],
        molecule_atom_lens: Int["b n"],
        # additional input
        chains_list: List[Tuple[int, int] | Tuple[int]],
        is_fine_tuning: bool = None,
        unweighted: bool = False,
        # RASA input
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,
        molecule_ids: Int["b n"] | None = None,
    ) -> Float[" b"]:
        """Compute the weighted lDDT.

        :param pred_coords: [b m 3] predicted coordinates
        :param true_coords: [b m 3] true coordinates
        :param atom_mask: [b m] atom mask
        :param asym_id: [b n] asym_id of each residue
        :param is_molecule_types: [b n 2] is_molecule_types
        :param molecule_atom_lens: [b n] molecule atom lens
        :param chains_list: List of chains
        :param is_fine_tuning: is fine tuning
        :param unweighted: unweighted lddt
        :param compute_rasa: compute RASA
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :return: [b] weighted lddt
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        device = pred_coords.device
        batch_size = pred_coords.shape[0]

        # broadcast asym_id and is_molecule_types to atom level
        atom_asym_id = batch_repeat_interleave(asym_id, molecule_atom_lens, output_padding_value=-1)
        atom_is_molecule_types = batch_repeat_interleave(is_molecule_types, molecule_atom_lens)

        weighted_lddt = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            chains = chains_list[b]
            if len(chains) == 2:
                asym_id_a = chains[0]
                asym_id_b = chains[1]
                lddt_type = "interface"
            elif len(chains) == 1:
                asym_id_a = asym_id_b = chains[0]
                lddt_type = "intra-chain"
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
                asym_mask_a,
                asym_mask_b,
                pred_coords[b],
                true_coords[b],
                atom_is_molecule_types[b],
                atom_mask[b],
            )

            weighted_lddt[b] = (1.0 if unweighted else lddt_weight) * lddt

        # Average the lDDT with the relative solvent accessible surface area (RASA) for unresolved proteins
        # NOTE: This differs from the AF3 Section 5.7 slightly, as here we compute the algebraic mean of the (batched) lDDT and RASA
        if compute_rasa:
            assert (
                exists(unresolved_cid) and exists(unresolved_residue_mask) and exists(molecule_ids)
            ), "RASA computation requires `unresolved_cid`, `unresolved_residue_mask`, and `molecule_ids` to be provided."
            weighted_rasa = self.compute_unresolved_rasa(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                true_coords,
                atom_mask,
                is_fine_tuning=is_fine_tuning,
            )
            weighted_lddt = (weighted_lddt + weighted_rasa) / 2

        return weighted_lddt

    @typecheck
    def calc_atom_access_surface_score_from_structure(
        self,
        structure: Structure,
        **kwargs
    ) -> Float['m']:

       # use the structure as source of truth, matching what xluo did

        structure_atom_pos = []
        structure_atom_type_for_radii = []
        side_atom_index = len(self.atom_type_index)

        for atom in structure.get_atoms():

            one_atom_pos = list(atom.get_vector())
            one_atom_type = self.atom_type_index.get(atom.name, side_atom_index)

            structure_atom_pos.append(one_atom_pos)
            structure_atom_type_for_radii.append(one_atom_type)

        structure_atom_pos: Float['m 3'] = tensor(structure_atom_pos, device = self.device)
        structure_atom_type_for_radii: Int['m'] = tensor(structure_atom_type_for_radii, device = self.device)

        structure_atoms_per_residue: Int['n'] = tensor([len([*residue.get_atoms()]) for residue in structure.get_residues()], device = self.device).long()

        return self.calc_atom_access_surface_score(
            atom_pos = structure_atom_pos,
            atom_type = structure_atom_type_for_radii,
            molecule_atom_lens = structure_atoms_per_residue,
            **kwargs
        )

    @typecheck
    def calc_atom_access_surface_score(
        self,
        atom_pos: Float['m 3'],
        atom_type: Int['m'],
        molecule_atom_lens: Int['n'] | None = None,
        atom_distance_min_thres = 1e-4
    ) -> Float['m'] | Float['n']:

        atom_radii: Float['m'] = self.atom_radii[atom_type]

        water_radii = self.atom_radii[-1]

        # atom radii is always summed with water radii

        atom_radii += water_radii
        atom_radii_sq = atom_radii.pow(2) # always use square of radii or distance for comparison - save on sqrt

        # write custom RSA function here

        # get atom relative positions + distance
        # for determining whether to include pairs of atom in calculation for the `free` adjective

        atom_rel_pos = einx.subtract('j c, i c -> i j c', atom_pos, atom_pos)
        atom_rel_dist_sq = atom_rel_pos.pow(2).sum(dim = -1)

        max_distance_include = einx.add('i, j -> i j', atom_radii, atom_radii).pow(2)

        include_in_free_calc = (
            (atom_rel_dist_sq < max_distance_include) &
            (atom_rel_dist_sq > atom_distance_min_thres)
        )

        # max included in calculation per row

        max_included = include_in_free_calc.long().sum(dim = -1).amax()

        include_in_free_calc, include_indices = include_in_free_calc.long().topk(max_included, dim = -1)

        # atom_rel_pos = einx.get_at('i [m] c, i j -> i j c', atom_rel_pos, include_indices)

        include_in_free_calc = include_in_free_calc.bool()
        atom_rel_pos = atom_rel_pos.gather(1, repeat(include_indices, 'i j -> i j c', c = 3))
        target_atom_radii_sq = atom_radii_sq[include_indices]

        # overall logic

        surface_dots = einx.multiply('m, sd c -> m sd c', atom_radii, self.unit_surface_dots)

        dist_from_surface_dots_sq = einx.subtract('i j c, i sd c -> i sd j c', atom_rel_pos, surface_dots).pow(2).sum(dim = -1)

        target_atom_close_to_surface_dots = einx.less('i j, i sd j -> i sd j', target_atom_radii_sq, dist_from_surface_dots_sq)

        target_atom_close_or_not_included = einx.logical_or('i sd j, i j -> i sd j', target_atom_close_to_surface_dots, ~include_in_free_calc)

        is_free = reduce(target_atom_close_or_not_included, 'i sd j -> i sd', 'all') # basically the most important line, calculating whether an atom is free by some distance measure

        score = reduce(is_free.float() * self.surface_weight, 'm sd -> m', 'sum')

        per_atom_access_surface_score = score * atom_radii_sq

        if not exists(molecule_atom_lens):
            return per_atom_access_surface_score

        # sum up all surface scores for atoms per residue
        # the final score seems to be the average of the rsa across all residues (selected by `chain_unresolved_residue_mask`)

        rasa, mask = sum_pool_with_lens(
            rearrange(per_atom_access_surface_score, '... -> 1 ... 1'),
            rearrange(molecule_atom_lens, '... -> 1 ...')
        )

        rasa = einx.where('b n, b n d, -> b n d', mask, rasa, 0.)

        rasa = rearrange(rasa, '1 n 1 -> n')

        return rasa

    @typecheck
    def _compute_unresolved_rasa(
        self,
        unresolved_cid: int,
        unresolved_residue_mask: Bool["n"],
        asym_id: Int["n"],
        molecule_ids: Int["n"],
        molecule_atom_lens: Int["n"],
        atom_pos: Float["m 3"],
        atom_mask: Bool["m"],
        **rsa_calc_kwargs
    ) -> Float[""]:
        """Compute the unresolved relative solvent accessible surface area (RASA) for proteins.
        using inhouse rebuilt RSA calculation

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: asym_id for each residue
        molecule_ids: molecule_ids for each residue
        molecule_atom_lens: number of atoms for each residue
        atom_pos: [m 3] atom positions
        atom_mask: True for valid atoms, False for missing/padding atoms
        :return: unresolved RASA
        """

        num_atom = atom_pos.shape[0]

        chain_mask = asym_id == unresolved_cid
        chain_unresolved_residue_mask = unresolved_residue_mask[chain_mask]
        chain_asym_id = asym_id[chain_mask]
        chain_molecule_ids = molecule_ids[chain_mask]
        chain_molecule_atom_lens = molecule_atom_lens[chain_mask]

        chain_mask_to_atom = torch.repeat_interleave(chain_mask, molecule_atom_lens)

        # if there's padding in num atom
        num_pad = num_atom - molecule_atom_lens.sum()
        if num_pad > 0:
            chain_mask_to_atom = F.pad(chain_mask_to_atom, (0, num_pad), value=False)

        chain_atom_pos = atom_pos[chain_mask_to_atom]
        chain_atom_mask = atom_mask[chain_mask_to_atom]

        structure = protein_structure_from_feature(
            chain_asym_id,
            chain_molecule_ids,
            chain_molecule_atom_lens,
            chain_atom_pos,
            chain_atom_mask,
        )

        # per atom rsa calculation

        rasa = self.calc_atom_access_surface_score_from_structure(
            structure,
            **rsa_calc_kwargs
        )

        unresolved_rasa = rasa[chain_unresolved_residue_mask]

        return unresolved_rasa.mean()

    @typecheck
    def compute_unresolved_rasa(
        self,
        unresolved_cid: List[int],
        unresolved_residue_mask: Bool["b n"],
        asym_id: Int["b n"],
        molecule_ids: Int["b n"],
        molecule_atom_lens: Int["b n"],
        atom_pos: Float["b m 3"],
        atom_mask: Bool["b m"],
        is_fine_tuning: bool = None,
    ) -> Float["b"]:
        """Compute the unresolved relative solvent accessible surface area (RASA) for (batched)
        proteins.

        unresolved_cid: asym_id for protein chains with unresolved residues
        unresolved_residue_mask: True for unresolved residues, False for resolved residues
        asym_id: [b n] asym_id of each residue
        molecule_ids: [b n] molecule_ids of each residue
        molecule_atom_lens: [b n] molecule atom lens
        atom_pos: [b m 3] atom positions
        atom_mask: [b m] atom mask
        :return: [b] unresolved RASA
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        weight_dict = default(
            self.weight_dict_config,
            self.FINETUNING_DICT if is_fine_tuning else self.INITIAL_TRAINING_DICT,
        )

        weight = weight_dict.get("unresolved", {}).get("unresolved", None)
        assert weight, "Weight not found for unresolved"

        unresolved_rasa = [
            self._compute_unresolved_rasa(*args)
            for args in zip(
                unresolved_cid,
                unresolved_residue_mask,
                asym_id,
                molecule_ids,
                molecule_atom_lens,
                atom_pos,
                atom_mask,
            )
        ]
        return torch.stack(unresolved_rasa) * weight

    @typecheck
    def compute_model_selection_score(
        self,
        batch: BatchedAtomInput,
        samples: List[Sample],
        is_fine_tuning: bool = None,
        return_details: bool = False,
        return_unweighted_scores: bool = False,
        compute_rasa: bool = False,
        unresolved_cid: List[int] | None = None,
        unresolved_residue_mask: Bool["b n"] | None = None,
        missing_chain_index: int = -1,
    ) -> Float[" b"] | ScoreDetails:
        """Compute the model selection score for an input batch and corresponding (sampled) atom
        positions.

        :param batch: A batch of `AtomInput` data.
        :param samples: A list of sampled atom positions along with their predicted distance errors and labels.
        :param is_fine_tuning: is fine tuning
        :param return_details: return the top model and its score
        :param return_unweighted_scores: return the unweighted scores (i.e., lDDT)
        :param compute_rasa: compute the relative solvent accessible surface area (RASA) for unresolved proteins
        :param unresolved_cid: unresolved chain ids
        :param unresolved_residue_mask: unresolved residue mask
        :param missing_chain_index: missing chain index
        :return: [b] model selection score and optionally the top model
        """
        is_fine_tuning = default(is_fine_tuning, self.is_fine_tuning)

        if compute_rasa:
            if not (exists(unresolved_cid) and exists(unresolved_residue_mask)):
                logger.warning(
                    "RASA computation requires `unresolved_cid` and `unresolved_residue_mask` to be provided. Skipping RASA computation."
                )
                compute_rasa = False

        # collect required features

        batch_dict = batch.dict()

        atom_pos_true = batch_dict["atom_pos"]
        atom_mask = ~batch_dict["missing_atom_mask"]

        asym_id = batch_dict["additional_molecule_feats"].unbind(dim=-1)[2]
        is_molecule_types = batch_dict["is_molecule_types"]

        chains = [
            tuple(chain for chain in chains_list if chain != missing_chain_index)
            for chains_list in batch_dict["chains"].tolist()
        ]
        molecule_atom_lens = batch_dict["molecule_atom_lens"]
        molecule_ids = batch_dict["molecule_ids"]

        valid_atom_len_mask = batch_dict["molecule_atom_lens"] >= 0
        tok_repr_atm_mask = batch_dict["distogram_atom_indices"] >= 0 & valid_atom_len_mask

        # score samples

        scored_samples: List[ScoredSample] = []

        for sample_idx, sample in enumerate(samples):
            atom_pos_pred, pde_logits, plddt, dist_logits = sample

            weighted_lddt = self.compute_weighted_lddt(
                atom_pos_pred,
                atom_pos_true,
                atom_mask,
                asym_id,
                is_molecule_types,
                molecule_atom_lens,
                chains_list=chains,
                is_fine_tuning=is_fine_tuning,
                compute_rasa=compute_rasa,
                unresolved_cid=unresolved_cid,
                unresolved_residue_mask=unresolved_residue_mask,
                molecule_ids=molecule_ids,
                unweighted=return_unweighted_scores,
            )

            gpde = self.compute_gpde(
                pde_logits,
                dist_logits,
                self.dist_breaks,
                tok_repr_atm_mask,
            )

            scored_samples.append((sample_idx, atom_pos_pred, plddt, weighted_lddt, gpde))

        # quick collate

        *_, all_weighted_lddt, all_gpde = zip(*scored_samples)

        # rank by batch-averaged minimum gPDE

        best_gpde_index = torch.stack(all_gpde).mean(dim=-1).argmin().item()

        # rank by batch-averaged maximum lDDT

        best_lddt_index = torch.stack(all_weighted_lddt).mean(dim=-1).argmax().item()

        # some weighted score

        model_selection_score = (
            scored_samples[best_gpde_index][-2] + scored_samples[best_lddt_index][-2]
        ) / 2

        if not return_details:
            return model_selection_score

        score_details = ScoreDetails(
            best_gpde_index=best_gpde_index,
            best_lddt_index=best_lddt_index,
            score=model_selection_score,
            scored_samples=scored_samples,
        )

        return score_details

    @typecheck
    def forward(
        self, alphafolds: Tuple[Alphafold3], batched_atom_inputs: BatchedAtomInput, **kwargs
    ) -> Float[" b"] | ScoreDetails:
        """Make model selections by computing the model selection score.

        NOTE: Give this function a tuple of `Alphafold3` modules and a batch of atomic inputs, and it will
        select the best module via the model selection score by returning the index of the corresponding tuple.

        :param alphafolds: Tuple of `Alphafold3` modules
        :param batched_atom_inputs: A batch of `AtomInput` data
        :param kwargs: Additional keyword arguments
        :return: Model selection score
        """

        samples = []

        with torch.no_grad():
            for alphafold in alphafolds:
                alphafold.eval()

                pred_atom_pos, logits = alphafold(
                    **batched_atom_inputs.model_forward_dict(),
                    return_loss=False,
                    return_confidence_head_logits=True,
                    return_distogram_head_logits=True,
                )
                plddt = self.compute_confidence_score.compute_plddt(logits.plddt)

                samples.append((pred_atom_pos, logits.pde, plddt, logits.distance))

        scores = self.compute_model_selection_score(batched_atom_inputs, samples=samples, **kwargs)

        return scores
