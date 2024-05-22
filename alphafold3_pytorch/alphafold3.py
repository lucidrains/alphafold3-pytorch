"""
global ein notation:

b - batch
h - heads
n - residue sequence length
i - residue sequence length (source)
j - residue sequence length (target)
m - atom sequence length
d - feature dimension
ds - feature dimension (single)
dp - feature dimension (pairwise)
dap - feature dimension (atompair)
da - feature dimension (atom)
t - templates
s - msa
"""

from __future__ import annotations

from math import pi, sqrt
from functools import partial
from collections import namedtuple

import torch
from torch import nn, sigmoid
from torch import Tensor
import torch.nn.functional as F

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from typing import Literal, Tuple, NamedTuple

from alphafold3_pytorch.typing import (
    Float,
    Int,
    Bool,
    typecheck
)

from alphafold3_pytorch.attention import Attention

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# Loss functions

@typecheck
def calc_smooth_lddt_loss(
    denoised: Float['b m 3'], 
    ground_truth: Float['b m 3'], 
    is_rna_per_atom: Float['b m'],
    is_dna_per_atom: Float['b m']
) -> Float[' b']:
    
    m, device = is_rna_per_atom.shape[-1], denoised.device
    
    dx_denoised = torch.cdist(denoised, denoised)
    dx_gt = torch.cdist(ground_truth, ground_truth)
    
    ddx = torch.abs(dx_gt - dx_denoised)
    eps = 0.25 * (
        sigmoid(0.5 - ddx) + sigmoid(1 - ddx) + sigmoid(2 - ddx) + sigmoid(4 - ddx)
    )
    
    is_nuc = is_rna_per_atom + is_dna_per_atom
    mask = einx.multiply('b i, b j -> b i j', is_nuc, is_nuc)
    c = (dx_gt < 30) * mask + (dx_gt < 15) * (1 - mask)
    
    eye = torch.eye(m, device = device)
    num = einx.sum('b [...]', c * eps * (1 - eye)) / (m**2 - m)
    den = einx.sum('b [...]', c * (1 - eye)) / (m**2 - m)

    return 1. - num/den

# linear and outer sum
# for single repr -> pairwise pattern throughout this architecture

class LinearNoBiasThenOuterSum(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = LinearNoBias(dim, dim_out * 2)

    @typecheck
    def forward(
        self,
        t: Float['b n ds']
    ) -> Float['b n n dp']:

        single_i, single_j = self.proj(t).chunk(2, dim = -1)
        out = einx.add('b i d, b j d -> b i j d', single_i, single_j)
        return out

# classic feedforward, SwiGLU variant
# they name this 'transition' in their paper
# Algorithm 11

class SwiGLU(Module):
    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float[' ... (d//2)']:

        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

class Transition(Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor = 4
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float['... d']:

        return self.ff(x)

# dropout
# they seem to be using structured dropout - row / col wise in triangle modules

class Dropout(Module):
    @typecheck
    def __init__(
        self,
        prob: float,
        *,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    @typecheck
    def forward(
        self,
        t: Tensor
    ) -> Tensor:

        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'tensor must be 4 dimensions for row / col structured dropout'

        if not exists(self.dropout_type):
            return self.dropout(t)

        if self.dropout_type == 'row':
            batch, row, _, _ = t.shape
            ones_shape = (batch, row, 1, 1)

        elif self.dropout_type == 'col':
            batch, _, col, _ = t.shape
            ones_shape = (batch, 1, col, 1)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped

# normalization
# both pre layernorm as well as adaptive layernorm wrappers

class PreLayerNorm(Module):
    @typecheck
    def __init__(
        self,
        fn: Attention | Transition | TriangleAttention | TriangleMultiplication | AttentionPairBias,
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    @typecheck
    def forward(
        self,
        x: Float['... n d'],
        **kwargs
    ) -> Float['... n d']:

        x = self.norm(x)
        return self.fn(x, **kwargs)

class AdaptiveLayerNorm(Module):
    """ Algorithm 26 """

    def __init__(
        self,
        *,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)
        self.norm_cond = nn.LayerNorm(dim_cond, bias = False)

        self.to_gamma = nn.Sequential(
            Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    @typecheck
    def forward(
        self,
        x: Float['b n d'],
        cond: Float['b n dc']
    ) -> Float['b n d']:

        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta

class ConditionWrapper(Module):
    """ Algorithm 25 """

    @typecheck
    def __init__(
        self,
        fn: Attention | Transition | TriangleAttention |  AttentionPairBias,
        *,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        self.fn = fn
        self.adaptive_norm = AdaptiveLayerNorm(dim = dim, dim_cond = dim_cond)

        adaln_zero_gamma_linear = Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    @typecheck
    def forward(
        self,
        x: Float['b n d'],
        *,
        cond: Float['b n dc'],
        **kwargs
    ) -> Float['b n d']:
        x = self.adaptive_norm(x, cond = cond)

        out = self.fn(x, **kwargs)

        gamma = self.to_adaln_zero_gamma(cond)
        return out * gamma

# triangle multiplicative module
# seems to be unchanged from alphafold2

class TriangleMultiplication(Module):

    @typecheck
    def __init__(
        self,
        *,
        dim,
        dim_hidden = None,
        mix: Literal["incoming", "outgoing"] = 'incoming',
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()

        dim_hidden = default(dim_hidden, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = Linear(dim, dim_hidden)
        self.right_proj = Linear(dim, dim_hidden)

        self.left_gate = Linear(dim, dim_hidden)
        self.right_gate = Linear(dim, dim_hidden)
        self.out_gate = Linear(dim, dim_hidden)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'incoming':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(dim_hidden)

        self.to_out = Sequential(
            Linear(dim_hidden, dim),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        x: Float['b n n d'],
        mask: Bool['b n'] | None = None
    ) -> Float['b n n d']:

        if exists(mask):
            mask = einx.logical_and('b i, b j -> b i j 1', mask, mask)

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

# there are two types of attention in this paper, triangle and attention-pair-bias
# they differ by how the attention bias is computed
# triangle is axial attention w/ itself projected for bias

class AttentionPairBias(Module):
    def __init__(
        self,
        *,
        heads,
        dim_pairwise,
        window_size = None,
        **attn_kwargs
    ):
        super().__init__()

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        to_attn_bias_linear = LinearNoBias(dim_pairwise, heads)
        nn.init.zeros_(to_attn_bias_linear.weight)

        self.to_attn_bias = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            to_attn_bias_linear,
            Rearrange('... i j h -> ... h i j')
        )

    @typecheck
    def forward(
        self,
        single_repr: Float['b n ds'],
        *,
        pairwise_repr: Float['b n n dp'],
        attn_bias: Float['b n n'] | None = None,
        **kwargs
    ) -> Float['b n ds']:

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'b i j -> b 1 i j')
        else:
            attn_bias = 0.

        attn_bias = self.to_attn_bias(pairwise_repr) + attn_bias

        out = self.attn(
            single_repr,
            attn_bias = attn_bias,
            **kwargs
        )

        return out

class TriangleAttention(Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        node_type: Literal['starting', 'ending'],
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None,
        **attn_kwargs
    ):
        super().__init__()
        self.need_transpose = node_type == 'ending'

        self.attn = Attention(dim = dim, heads = heads, **attn_kwargs)

        self.dropout = Dropout(dropout, dropout_type = dropout_type)

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        mask: Bool['b n'] | None = None,
        **kwargs
    ) -> Float['b n n d']:

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b r) ...', r = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b r) ...', r = batch_repeat)

        pairwise_repr, packed_shape = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            **kwargs
        )

        out = unpack_one(out, packed_shape, '* n d')

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)

# PairwiseBlock
# used in both MSAModule and Pairformer
# consists of all the "Triangle" modules + Transition

class PairwiseBlock(Module):
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout_row_prob = 0.25,
        dropout_col_prob = 0.25,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)

        tri_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        tri_attn_kwargs = dict(
            dim = dim_pairwise,
            heads = tri_attn_heads,
            dim_head = tri_attn_dim_head
        )

        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = 'outgoing', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = 'incoming', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_attn_starting = pre_ln(TriangleAttention(node_type = 'starting', dropout = dropout_row_prob, dropout_type = 'row', **tri_attn_kwargs))
        self.tri_attn_ending = pre_ln(TriangleAttention(node_type = 'ending', dropout = dropout_col_prob, dropout_type = 'col', **tri_attn_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    @typecheck
    def forward(
        self,
        *,
        pairwise_repr: Float['b n n d'],
        mask: Bool['b n'] | None = None
    ):
        pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_starting(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_ending(pairwise_repr, mask = mask) + pairwise_repr

        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr

# msa module

class OuterProductMean(Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_hidden = 32,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_pairwise_repr = nn.Linear(dim_hidden ** 2, dim_pairwise)

    @typecheck
    def forward(
        self,
        msa: Float['b s n d'],
        *,
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None
    ) -> Float['b n n dp']:

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)

        outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e s')

        # maybe masked mean for outer product

        if exists(msa_mask):
            msa_mask = rearrange(msa_mask, 'b s -> b 1 1 1 s')
            outer_product = outer_product * msa_mask

            num = reduce(outer_product, '... s -> ...', 'sum')
            den = reduce(msa_mask.float(), '... s -> ...', 'sum')

            outer_product_mean = num / den.clamp(min = self.eps)
        else:
            outer_product_mean = reduce(outer_product, '... s -> ...', 'mean')

        # flatten

        outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # masking for pairwise repr

        if exists(mask):
            mask = einx.logical_and('b i , b j -> b i j 1', mask, mask)
            outer_product_mean = outer_product_mean * mask

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        return pairwise_repr


class MSAPairWeightedAveraging(Module):
    """ Algorithm 10 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_head = 32,
        heads = 8,
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            nn.LayerNorm(dim_msa),
            LinearNoBias(dim_msa, dim_inner * 2),
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            Rearrange('b h s n d -> b s n (h d)'),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        *,
        msa: Float['b s n d'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b s n d']:

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # line 3

        b = self.pairwise_repr_to_attn(pairwise_repr)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5

        weights = b.softmax(dim = -1)

        # line 6

        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')

        out = out * gates

        # combine heads

        return self.to_out(out)

class MSAModule(Module):
    """ Algorithm 8 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 4,
        dim_msa = 64,
        dim_msa_input = None,
        outer_product_mean_dim_hidden = 32,
        msa_pwa_dropout_row_prob = 0.15,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 32,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa: int | None = None
    ):
        super().__init__()

        self.max_num_msa = default(max_num_msa, float('inf'))  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input, dim_msa) if exists(dim_msa_input) else nn.Identity()

        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = ModuleList([])

        for _ in range(depth):

            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = outer_product_mean_dim_hidden
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                heads = msa_pwa_heads,
                dim_head = msa_pwa_dim_head,
                dropout = msa_pwa_dropout_row_prob,
                dropout_type = 'row'
            )

            msa_transition = Transition(dim = dim_msa)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            layers.append(ModuleList([
                outer_product_mean,
                msa_pair_weighted_avg,
                msa_pre_ln(msa_transition),
                pairwise_block
            ]))

        self.layers = layers

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        batch, num_msa, device = *msa.shape[:2], msa.device

        # sample without replacement

        if num_msa > self.max_num_msa:
            rand = torch.randn((batch, num_msa), device = device)

            if exists(msa_mask):
                rand.masked_fill_(~msa_mask, max_neg_value(msa))

            indices = rand.topk(self.max_num_msa, dim = -1).indices

            msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)

            if exists(msa_mask):
                msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)

        # process msa

        msa = self.msa_init_proj(msa)

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in self.layers:

            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask = mask, msa_mask = msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
            msa = msa_transition(msa) + msa            

            # pairwise block

            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

        return pairwise_repr

# pairformer stack

class PairformerStack(Module):
    """ Algorithm 17 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 48,
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        pairwise_block_kwargs: dict = dict()
    ):
        super().__init__()
        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob
        )

        for _ in range(depth):

            single_pre_ln = partial(PreLayerNorm, dim = dim_single)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            pair_bias_attn = AttentionPairBias(**pair_bias_attn_kwargs)
            single_transition = Transition(dim = dim_single)

            layers.append(ModuleList([
                pairwise_block,
                single_pre_ln(pair_bias_attn),
                single_pre_ln(single_transition),
            ]))

        self.layers = layers

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        for (
            pairwise_block,
            pair_bias_attn,
            single_transition
        ) in self.layers:

            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

            single_repr = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask) + single_repr
            single_repr = single_transition(single_repr) + single_repr

        return single_repr, pairwise_repr

# embedding related

"""
additional_residue_feats: [*, rf]:
    0: residue_index
    1: token_index
    2: asym_id
    3: entity_id
    4: sym_id 
    5: restype (must be one hot encoded to 32)
    6: is_protein
    7: is_rna
    8: is_dna
    9: is_ligand
"""

class RelativePositionEncoding(Module):
    """ Algorithm 3 """
    
    def __init__(
        self,
        *,
        r_max = 32,
        s_max = 2,
        dim_out = 128
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        
        dim_input = (2*r_max+2) + (2*r_max+2) + 1 + (2*s_max+2)
        self.out_embedder = LinearNoBias(dim_input, dim_out)

    @typecheck
    def forward(
        self,
        *,
        additional_residue_feats: Float['b n 10']
    ) -> Float['b n n dp']:

        device = additional_residue_feats.device
        assert additional_residue_feats.shape[-1] >= 5

        res_idx, token_idx, asym_id, entity_id, sym_id = additional_residue_feats[..., :5].unbind(dim = -1)
        
        diff_res_idx = einx.subtract('b i, b j -> b i j', res_idx, res_idx)
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        diff_sym_id = einx.subtract('b i, b j -> b i j', sym_id, sym_id)

        mask_same_chain = einx.subtract('b i, b j -> b i j', asym_id, asym_id) == 0
        mask_same_res = diff_res_idx == 0
        mask_same_entity = einx.subtract('b i, b j -> b i j 1', entity_id, entity_id) == 0
        
        d_res = torch.where(
            mask_same_chain, 
            torch.clip(diff_res_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_token = torch.where(
            mask_same_chain * mask_same_res, 
            torch.clip(diff_token_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_chain = torch.where(
            ~mask_same_chain, 
            torch.clip(diff_sym_id + self.s_max, 0, 2*self.s_max),
            2*self.s_max + 1
        )
        
        def onehot(x, bins):
            x, packed_shape = pack_one(x, '*')
            dist_from_bins = einx.subtract('i, j -> i j', x, bins)
            indexes = dist_from_bins.abs().min(dim = 1, keepdim = True).indices
            indexes = rearrange(indexes.long(), 'i j -> (i j) 1')
            one_hots = torch.zeros(indexes.shape[0], len(bins)).scatter_(1, indexes, 1)
            return unpack_one(one_hots, packed_shape, '* d')

        r_arange = torch.arange(2*self.r_max + 2, device = device)
        s_arange = torch.arange(2*self.s_max + 2, device = device)

        a_rel_pos = onehot(d_res, r_arange)
        a_rel_token = onehot(d_token, r_arange)
        a_rel_chain = onehot(d_chain, s_arange)

        out, _ = pack((
            a_rel_pos,
            a_rel_token,
            mask_same_entity,
            a_rel_chain
        ), 'b i j *')

        return self.out_embedder(out)

class TemplateEmbedder(Module):
    """ Algorithm 16 """

    def __init__(
        self,
        *,
        dim_template_feats,
        dim = 64,
        dim_pairwise = 128,
        pairformer_stack_depth = 2,
        pairwise_block_kwargs: dict = dict(),
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps

        self.template_feats_to_embed_input = LinearNoBias(dim_template_feats, dim)

        self.pairwise_to_embed_input = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim)
        )

        layers = ModuleList([])
        for _ in range(pairformer_stack_depth):
            block = PairwiseBlock(
                dim_pairwise = dim,
                **pairwise_block_kwargs
            )

            layers.append(block)

        self.pairformer_stack = layers

        self.final_norm = nn.LayerNorm(dim)

        # final projection of mean pooled repr -> out

        self.to_out = nn.Sequential(
            LinearNoBias(dim, dim_pairwise),
            nn.ReLU()
        )

    @typecheck
    def forward(
        self,
        *,
        templates: Float['b t n n dt'],
        template_mask: Bool['b t'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b n n dp']:

        num_templates = templates.shape[1]

        pairwise_repr = self.pairwise_to_embed_input(pairwise_repr)
        pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b 1 i j d')

        v = self.template_feats_to_embed_input(templates) + pairwise_repr

        v, merged_batch_ps = pack_one(v, '* i j d')

        if exists(mask):
            mask = repeat(mask, 'b n -> (b t) n', t = num_templates)

        for block in self.pairformer_stack:
            v = block(
                pairwise_repr = v,
                mask = mask
            ) + v

        u = self.final_norm(v)

        u = unpack_one(u, merged_batch_ps, '* i jk d')

        # masked mean pool template repr

        u = u.masked_fill(
            ~rearrange(template_mask, 'b t -> b t 1 1 1'),
            0.
        )

        num = reduce(u, 'b t i j d -> b i j d', 'sum')
        den = reduce(template_mask.float(), 'b t -> b 1 1 1', 'sum')

        avg_template_repr = num / den.clamp(min = self.eps)

        return self.to_out(avg_template_repr)

# diffusion related
# both diffusion transformer as well as atom encoder / decoder

class FourierEmbedding(Module):
    """ Algorithm 22 """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    @typecheck
    def forward(
        self,
        times: Float[' b'],
    ) -> Float['b d']:

        times = rearrange(times, 'b -> b 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

class PairwiseConditioning(Module):
    """ Algorithm 21 """

    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        dim_pairwise = 128,
        num_transitions = 2,
        transition_expansion_factor = 2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            LinearNoBias(dim_pairwise_trunk + dim_pairwise_rel_pos_feats, dim_pairwise),
            nn.LayerNorm(dim_pairwise)
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = PreLayerNorm(Transition(dim = dim_pairwise, expansion_factor = transition_expansion_factor), dim = dim_pairwise)
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
    ) -> Float['b n n dp']:

        pairwise_repr = torch.cat((pairwise_trunk, pairwise_rel_pos_feats), dim = -1)

        pairwise_repr = self.dim_pairwise_init_proj(pairwise_repr)

        for transition in self.transitions:
            pairwise_repr = transition(pairwise_repr) + pairwise_repr

        return pairwise_repr

class SingleConditioning(Module):
    """ Algorithm 21 """

    def __init__(
        self,
        *,
        sigma_data: float,
        dim_single = 384,
        dim_fourier = 256,
        num_transitions = 2,
        transition_expansion_factor = 2,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps

        self.dim_single = dim_single
        self.sigma_data = sigma_data

        self.norm_single = nn.LayerNorm(dim_single)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, dim_single)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = PreLayerNorm(Transition(dim = dim_single, expansion_factor = transition_expansion_factor), dim = dim_single)
            transitions.append(transition)

        self.transitions = transitions

    @typecheck
    def forward(
        self,
        *,
        times: Float[' b'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
    ) -> Float['b n (dst+dsi)']:

        single_repr = torch.cat((single_trunk_repr, single_inputs_repr), dim = -1)

        assert single_repr.shape[-1] == self.dim_single

        single_repr = self.norm_single(single_repr)

        fourier_embed = self.fourier_embed(0.25 * log(times / self.sigma_data, eps = self.eps))

        normed_fourier = self.norm_fourier(fourier_embed)

        fourier_to_single = self.fourier_to_single(normed_fourier)

        single_repr = rearrange(fourier_to_single, 'b d -> b 1 d') + single_repr

        for transition in self.transitions:
            single_repr = transition(single_repr) + single_repr

        return single_repr

class DiffusionTransformer(Module):
    """ Algorithm 23 """

    def __init__(
        self,
        *,
        depth,
        heads,
        dim = 384,
        dim_single_cond = None,
        dim_pairwise = 128,
        attn_window_size = None,
        attn_pair_bias_kwargs: dict = dict(),
        serial = False
    ):
        super().__init__()
        dim_single_cond = default(dim_single_cond, dim)

        layers = ModuleList([])

        for _ in range(depth):

            pair_bias_attn = AttentionPairBias(
                dim = dim,
                dim_pairwise = dim_pairwise,
                heads = heads,
                window_size = attn_window_size,
                **attn_pair_bias_kwargs
            )

            transition = Transition(
                dim = dim
            )

            conditionable_pair_bias = ConditionWrapper(
                pair_bias_attn,
                dim = dim,
                dim_cond = dim_single_cond
            )

            conditionable_transition = ConditionWrapper(
                transition,
                dim = dim,
                dim_cond = dim_single_cond
            )

            layers.append(ModuleList([
                conditionable_pair_bias,
                conditionable_transition
            ]))

        self.layers = layers

        self.serial = serial

    @typecheck
    def forward(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None
    ):
        serial = self.serial

        for attn, transition in self.layers:

            attn_out = attn(
                noised_repr,
                cond = single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask
            )

            if serial:
                noised_repr = attn_out + noised_repr

            ff_out = transition(
                noised_repr,
                cond = single_repr
            )

            if not serial:
                ff_out = ff_out + attn_out

            noised_repr = noised_repr + ff_out

        return noised_repr

class AtomToTokenPooler(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        atoms_per_window = 27
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Sequential(
            LinearNoBias(dim, dim_out),
            nn.ReLU()
        )

        self.atoms_per_window = atoms_per_window

    @typecheck
    def forward(
        self,
        *,
        atom_feats: Float['b m da'],
        atom_mask: Bool['b m']
    ) -> Float['b n ds']:
        w = self.atoms_per_window

        atom_feats = self.proj(atom_feats)

        # masked mean pool the atom feats for each residue, for the token transformer
        # this is basically a simple 2-level hierarchical transformer

        windowed_atom_feats = rearrange(atom_feats, 'b (n w) da -> b n w da', w = w)
        windowed_atom_mask = rearrange(atom_mask, 'b (n w) -> b n w', w = w)

        assert windowed_atom_mask.any(dim = -1).all(), 'atom mask must contain one valid atom for each window'

        windowed_atom_feats = windowed_atom_feats.masked_fill(windowed_atom_mask[..., None], 0.)

        num = reduce(windowed_atom_feats, 'b n w d -> b n d', 'sum')
        den = reduce(windowed_atom_mask.float(), 'b n w -> b n 1', 'sum')

        tokens = num / den
        return tokens

class DiffusionModule(Module):
    """ Algorithm 20 """

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        atoms_per_window = 27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the residue or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms
        dim_pairwise = 128,
        sigma_data = 16,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 768,
        dim_single = 384,
        dim_fourier = 256,
        single_cond_kwargs: dict = dict(
            num_transitions = 2,
            transition_expansion_factor = 2,
        ),
        pairwise_cond_kwargs: dict = dict(
            num_transitions = 2
        ),
        atom_encoder_depth = 3,
        atom_encoder_heads = 4,
        token_transformer_depth = 24,
        token_transformer_heads = 16,
        atom_decoder_depth = 3,
        atom_decoder_heads = 4,
        serial = False
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window
        self.sigma_data = sigma_data
        # conditioning

        self.single_conditioner = SingleConditioning(
            sigma_data = sigma_data,
            dim_single = dim_single,
            dim_fourier = dim_fourier,
            **single_cond_kwargs
        )

        self.pairwise_conditioner = PairwiseConditioning(
            dim_pairwise_trunk = dim_pairwise_trunk,
            dim_pairwise_rel_pos_feats = dim_pairwise_rel_pos_feats,
            **pairwise_cond_kwargs
        )

        # atom attention encoding related modules

        self.atom_pos_to_atom_feat = LinearNoBias(3, dim_atom)

        self.single_repr_to_atom_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_atom)
        )

        self.pairwise_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_atompair)
        )

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBiasThenOuterSum(dim_atom, dim_atompair),
            nn.ReLU()
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_encoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_encoder_depth,
            heads = atom_encoder_heads,
            serial = serial
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token,
            atoms_per_window = atoms_per_window
        )

        # token attention related modules

        self.cond_tokens_with_cond_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_token)
        )

        self.token_transformer = DiffusionTransformer(
            dim = dim_token,
            dim_single_cond = dim_single,
            dim_pairwise = dim_pairwise,
            depth = token_transformer_depth,
            heads = token_transformer_heads,
            serial = serial
        )

        self.attended_token_norm = nn.LayerNorm(dim_token)

        # atom attention decoding related modules

        self.tokens_to_atom_decoder_input_cond = LinearNoBias(dim_token, dim_atom)

        self.atom_decoder = DiffusionTransformer(
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            depth = atom_decoder_depth,
            heads = atom_decoder_heads,
            serial = serial
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, 3)
        )

    @typecheck
    def forward(
        self,
        noised_atom_pos: Float['b m 3'],
        *,
        atom_feats: Float['b m da'],
        atompair_feats: Float['b m m dap'],
        atom_mask: Bool['b m'],
        times: Float[' b'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
    ):
        w = self.atoms_per_window

        # in the paper, it seems they pack the atom feats
        # but in this impl, will just use windows for simplicity when communicating between atom and residue resolutions. bit less efficient

        assert divisible_by(noised_atom_pos.shape[-2], w)

        conditioned_single_repr = self.single_conditioner(
            times = times,
            single_trunk_repr = single_trunk_repr,
            single_inputs_repr = single_inputs_repr
        )

        conditioned_pairwise_repr = self.pairwise_conditioner(
            pairwise_trunk = pairwise_trunk,
            pairwise_rel_pos_feats = pairwise_rel_pos_feats
        )

        # Line 2 : "Normalization": Scale positions to dimensionless vectors with approximately unit variance.
        # noised_atom_pos = noised_atom_pos / sqrt(times**2 + self.sigma_data**2)

        # lines 7-14 in Algorithm 5

        atom_feats_cond = atom_feats

        # the most surprising part of the paper; no geometric biases!

        atom_feats = self.atom_pos_to_atom_feat(noised_atom_pos) + atom_feats

        # condition atom feats cond (cl) with single repr

        single_repr_cond = self.single_repr_to_atom_feat_cond(conditioned_single_repr)

        single_repr_cond = repeat(single_repr_cond, 'b n ds -> b (n w) ds', w = w)
        atom_feats_cond = single_repr_cond + atom_feats_cond

        # condition atompair feats with pairwise repr

        pairwise_repr_cond = self.pairwise_repr_to_atompair_feat_cond(conditioned_pairwise_repr)
        pairwise_repr_cond = repeat(pairwise_repr_cond, 'b i j dp -> b (i w1) (j w2) dp', w1 = w, w2 = w)
        atompair_feats = pairwise_repr_cond + atompair_feats

        # condition atompair feats further with single atom repr

        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atompair_feats = atom_repr_cond + atompair_feats

        # furthermore, they did one more MLP on the atompair feats for attention biasing in atom transformer

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        # atom encoder

        atom_feats = self.atom_encoder(
            atom_feats,
            mask = atom_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_feats_skip = atom_feats

        tokens = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask
        )
        # token transformer

        tokens = self.cond_tokens_with_cond_single(conditioned_single_repr) + tokens

        self.token_transformer(
            tokens,
            mask = mask,
            single_repr = conditioned_single_repr,
            pairwise_repr = conditioned_pairwise_repr,
        )

        tokens = self.attended_token_norm(tokens)

        # atom decoder

        atom_decoder_input = self.tokens_to_atom_decoder_input_cond(tokens)
        atom_decoder_input = repeat(atom_decoder_input, 'b n da -> b (n w) da', w = w)

        atom_decoder_input = atom_decoder_input + atom_feats_skip

        atom_feats = self.atom_decoder(
            atom_decoder_input,
            mask = atom_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        # Line 8: "De-normalization: Rescale updates to positions and combine with input positions.
        # atom_pos_update = self.sigma_data**2 / (self.sigma_data**2 + times**2) * noised_atom_pos + self.sigma_data * times / sqrt(self.sigma_data**2 + times**2) * atom_pos_update


        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update

# elucidated diffusion model adapted for atom position diffusing
# from Karras et al.
# https://arxiv.org/abs/2206.00364

class DiffusionLossBreakdown(NamedTuple):
    mse: Float['']
    bond: Float['']
    smooth_lddt: Float['']

class ElucidatedAtomDiffusionReturn(NamedTuple):
    loss: Float['']
    denoised_atom_pos: Float['b m 3']
    loss_breakdown: DiffusionLossBreakdown
    noise_sigmas: Float[' b']

class ElucidatedAtomDiffusion(Module):
    @typecheck
    def __init__(
        self,
        net: DiffusionModule,
        *,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        smooth_lddt_loss_kwargs: dict = dict()
    ):
        super().__init__()
        self.net = net

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # smooth lddt loss

        self.smooth_lddt_loss = SmoothLDDTLoss(**smooth_lddt_loss_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    @typecheck
    def preconditioned_network_forward(
        self,
        noised_atom_pos: Float['b m 3'],
        sigma: Float[' b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        clamp = False,
    ):
        batch, device = noised_atom_pos.shape[0], noised_atom_pos.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_atom_pos,
            times = self.c_noise(sigma),
            **network_condition_kwargs
        )

        out = self.c_skip(padded_sigma) * noised_atom_pos +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Bool['b m'],
        num_sample_steps = None,
        clamp = True,
        **network_condition_kwargs
    ):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (*atom_mask.shape, 3)

        network_condition_kwargs.update(atom_mask = atom_mask)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # atom position is noise at the beginning

        init_sigma = sigmas[0]

        atom_pos = init_sigma * torch.randn(shape, device = self.device)

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(atom_pos_hat, sigma_hat, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
            # Not sure if normalization is requiered here :thinking face:
            # model_output = self.preconditioned_network_forward(atom_pos_hat * sqrt(sigma_hat**2 + self.sigma_data**2), sigma_hat, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(atom_pos_next, sigma_next, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = atom_pos_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            atom_pos = atom_pos_next

        atom_pos = atom_pos.clamp(-1., 1.)
        return atom_pos

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(
        self,
        normalized_atom_pos: Float['b m 3'],
        atom_mask: Bool['b m'],
        return_denoised_pos = False,
        additional_residue_feats: Float['b n 10'] | None = None,
        add_smooth_lddt_loss = False,
        add_bond_loss = False,
        nucleotide_loss_weight = 5.,
        ligand_loss_weight = 10.,
        return_loss_breakdown = False,
        **network_condition_kwargs
    ) -> ElucidatedAtomDiffusionReturn:

        batch_size = normalized_atom_pos.shape[0]

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(normalized_atom_pos)

        noised_atom_pos = normalized_atom_pos + padded_sigmas * noise  # alphas are 1. in the paper

        network_condition_kwargs.update(atom_mask = atom_mask)

        denoised_atom_pos = self.preconditioned_network_forward(
            noised_atom_pos,
            sigmas,
            network_condition_kwargs = network_condition_kwargs
        )

        total_loss = 0.

        # if additional residue feats is provided, get the mask for nucleotides and ligands

        if exists(additional_residue_feats):
            w = self.net.atoms_per_window

            is_nucleotide_or_ligand_fields = additional_residue_feats[..., 7:] != 0.
            atom_is_dna, atom_is_rna, atom_is_ligand = tuple(repeat(t != 0., 'b n -> b (n w)', w = w) for t in is_nucleotide_or_ligand_fields.unbind(dim = -1))

        # main diffusion mse loss

        losses = F.mse_loss(denoised_atom_pos, normalized_atom_pos, reduction = 'none') / 3.

        if exists(additional_residue_feats):
            # section 3.7.1 equation 4

            finetune_weight = torch.where(atom_is_dna | atom_is_rna, nucleotide_loss_weight, 1.)
            finetune_weight = torch.where(atom_is_ligand, ligand_loss_weight, finetune_weight)

            losses = einx.multiply('b m c, b m -> b m c',  losses, finetune_weight)

        # regular loss weight as defined in EDM paper

        loss_weights = self.loss_weight(padded_sigmas)

        losses = losses * loss_weights

        # account for atom mask

        mse_loss = losses[atom_mask].mean()

        total_loss = total_loss + mse_loss

        # proposed extra bond loss during finetuning

        bond_loss = self.zero

        if add_bond_loss:
            atompair_mask = einx.logical_and('b i, b j -> b i j', atom_mask, atom_mask)

            denoised_cdist = torch.cdist(denoised_atom_pos, denoised_atom_pos, p = 2)
            normalized_cdist = torch.cdist(normalized_atom_pos, normalized_atom_pos, p = 2)

            bond_losses = F.mse_loss(denoised_cdist, normalized_cdist, reduction = 'none')
            bond_losses = bond_losses * loss_weights

            bond_loss = bond_losses[atompair_mask].mean()

            total_loss = total_loss + bond_loss

        # proposed auxiliary smooth lddt loss

        smooth_lddt_loss = self.zero

        if add_smooth_lddt_loss:
            assert exists(additional_residue_feats)

            smooth_lddt_loss = self.smooth_lddt_loss(
                denoised_atom_pos,
                normalized_atom_pos,
                atom_is_dna,
                atom_is_rna,
                coords_mask = atom_mask
            )

            total_loss = total_loss + smooth_lddt_loss

        # calculate loss breakdown

        loss_breakdown = DiffusionLossBreakdown(mse_loss, bond_loss, smooth_lddt_loss)

        return ElucidatedAtomDiffusionReturn(total_loss, denoised_atom_pos, loss_breakdown, sigmas)

# modules todo

class SmoothLDDTLoss(Module):
    """ Algorithm 27 """

    @typecheck
    def __init__(
        self,
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0
    ):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        is_dna: Bool['b n'],
        is_rna: Bool['b n'],
        coords_mask: Bool['b n'] | None = None,
    ) -> Float['']:
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

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = einx.logical_and('b i, b j -> b i j', coords_mask, coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt_sum = (eps * mask).sum(dim=(-1, -2))
        lddt_count = mask.sum(dim=(-1, -2))
        lddt = lddt_sum / lddt_count.clamp(min=1)

        return 1 - lddt.mean()

class WeightedRigidAlign(Module):
    """ Algorithm 28 """
    def __init__(self):
        super().__init__()

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        weights: Float['b n']
    ) -> Float['b n 3']:
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

class ExpressCoordinatesInFrame(Module):
    """ Algorithm  29 """

    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps

    @typecheck
    def forward(
        self,
        coords: Float['b m 3'],
        frame: Float['b m 3 3'] | Float['b 3 3'] | Float['3 3']
    ) -> Float['b m 3']:
        """
        coords: coordinates to be expressed in the given frame (b, 3)
        frame: frame defined by three points (b, 3, 3)
        """

        if frame.ndim == 2:
            frame = rearrange(frame, 'fr fc -> 1 1 fr fc')
        elif frame.ndim == 3:
            frame = rearrange(frame, 'b fr fc -> b 1 fr fc')

        # Extract frame points
        a, b, c = frame.unbind(dim = -1)

        # Compute unit vectors of the frame
        e1 = F.normalize(a - b, dim = -1, eps = self.eps)
        e2 = F.normalize(c - b, dim = -1, eps = self.eps)
        e3 = torch.cross(e1, e2, dim = -1)

        # Express coordinates in the frame basis
        v = coords - b

        transformed_coords = torch.stack([
            einsum(v, e1, '... i, ... i -> ...'),
            einsum(v, e2, '... i, ... i -> ...'),
            einsum(v, e3, '... i, ... i -> ...')
        ], dim = -1)

        return transformed_coords

class ComputeAlignmentError(Module):
    """ Algorithm 30 """
    @typecheck
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.express_coordinates_in_frame = ExpressCoordinatesInFrame()

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        pred_frames: Float['b n 3 3'],
        true_frames: Float['b n 3 3']
    ) -> Float['b n']:
        """
        pred_coords: predicted coordinates (b, n, 3)
        true_coords: true coordinates (b, n, 3)
        pred_frames: predicted frames (b, n, 3, 3)
        true_frames: true frames (b, n, 3, 3)
        """
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pred_coords, pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(true_coords, true_frames)

        # Compute alignment errors
        alignment_errors = torch.sqrt(
            torch.sum((pred_coords_transformed - true_coords_transformed) ** 2, dim=-1) + self.eps
        )

        return alignment_errors

class CentreRandomAugmentation(Module):
    """ Algorithm 19 """
    @typecheck
    def __init__(self, trans_scale: float = 1.0):
        super().__init__()
        self.trans_scale = trans_scale

    @typecheck
    def forward(self, coords: Float['b n 3']) -> Float['b n 3']:
        """
        coords: coordinates to be augmented (b, n, 3)
        """
        # Center the coordinates
        centered_coords = coords - coords.mean(dim=1, keepdim=True)

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(coords.device)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(coords.device)

        # Apply rotation and translation
        augmented_coords = torch.einsum('bni,ij->bnj', centered_coords, rotation_matrix) + translation_vector

        return augmented_coords

    @typecheck
    def _random_rotation_matrix(self, device: torch.device) -> Float['3 3']:
        # Generate random rotation angles
        angles = torch.rand(3, device=device) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        rotation_matrix = torch.eye(3, device=device)
        rotation_matrix[0, 0] = cos_angles[0] * cos_angles[1]
        rotation_matrix[0, 1] = cos_angles[0] * sin_angles[1] * sin_angles[2] - sin_angles[0] * cos_angles[2]
        rotation_matrix[0, 2] = cos_angles[0] * sin_angles[1] * cos_angles[2] + sin_angles[0] * sin_angles[2]
        rotation_matrix[1, 0] = sin_angles[0] * cos_angles[1]
        rotation_matrix[1, 1] = sin_angles[0] * sin_angles[1] * sin_angles[2] + cos_angles[0] * cos_angles[2]
        rotation_matrix[1, 2] = sin_angles[0] * sin_angles[1] * cos_angles[2] - cos_angles[0] * sin_angles[2]
        rotation_matrix[2, 0] = -sin_angles[1]
        rotation_matrix[2, 1] = cos_angles[1] * sin_angles[2]
        rotation_matrix[2, 2] = cos_angles[1] * cos_angles[2]

        return rotation_matrix

    @typecheck
    def _random_translation_vector(self, device: torch.device) -> Float['3']:
        # Generate random translation vector
        translation_vector = torch.randn(3, device=device) * self.trans_scale
        return translation_vector

# input embedder

class EmbeddedInputs(NamedTuple):
    single_inputs: Float['b n ds']
    single_init: Float['b n ds']
    pairwise_init: Float['b n n dp']
    atom_feats: Float['b m da']
    atompair_feats: Float['b m m dap']

class InputFeatureEmbedder(Module):
    """ Algorithm 2 """

    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_additional_residue_feats,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        atom_transformer_blocks = 3,
        atom_transformer_heads = 4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window

        self.to_atom_feats = LinearNoBias(dim_atom_inputs, dim_atom)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBiasThenOuterSum(dim_atom, dim_atompair),
            nn.ReLU()
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        self.atom_transformer = DiffusionTransformer(
            depth = atom_transformer_blocks,
            heads = atom_transformer_heads,
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            **atom_transformer_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token,
            atoms_per_window = atoms_per_window
        )

        dim_single_input = dim_token + dim_additional_residue_feats

        self.dim_additional_residue_feats = dim_additional_residue_feats

        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(dim_single_input, dim_pairwise)

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m dai'],
        atom_mask: Bool['b m'],
        atompair_feats: Float['b m m dap'],
        additional_residue_feats: Float['b n rf'],
    ) -> EmbeddedInputs:

        assert additional_residue_feats.shape[-1] == self.dim_additional_residue_feats

        w = self.atoms_per_window

        atom_feats = self.to_atom_feats(atom_inputs)

        atom_feats_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atompair_feats = atom_feats_cond + atompair_feats

        atom_feats = self.atom_transformer(
            atom_feats,
            single_repr = atom_feats,
            pairwise_repr = atompair_feats
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        single_inputs = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask
        )

        single_inputs = torch.cat((single_inputs, additional_residue_feats), dim = -1)

        single_init = self.single_input_to_single_init(single_inputs)
        pairwise_init = self.single_input_to_pairwise_init(single_inputs)

        return EmbeddedInputs(single_inputs, single_init, pairwise_init, atom_feats, atompair_feats)

# distogram head

class DistogramHead(Module):

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        num_dist_bins = 38,   # think it is 38?
    ):
        super().__init__()

        self.to_distogram_logits = nn.Sequential(
            LinearNoBias(dim_pairwise, num_dist_bins),
            Rearrange('b ... l -> b l ...')
        )

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d']
    ) -> Float['b l n n']:

        logits = self.to_distogram_logits(pairwise_repr)
        return logits

# confidence head

class ConfidenceHeadLogits(NamedTuple):
    pae: Float['b pae n n'] | None
    pde: Float['b pde n n']
    plddt: Float['b plddt n']
    resolved: Float['b 2 n']

class ConfidenceHead(Module):
    """ Algorithm 31 """

    @typecheck
    def __init__(
        self,
        *,
        dim_single_inputs,
        atompair_dist_bins: Float['d'],
        dim_single = 384,
        dim_pairwise = 128,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        pairformer_depth = 4,
        pairformer_kwargs: dict = dict()
    ):
        super().__init__()

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

    @typecheck
    def forward(
        self,
        *,
        single_inputs_repr: Float['b n dsi'],
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        pred_atom_pos: Float['b n 3'],
        mask: Bool['b n'] | None = None,
        return_pae_logits = True

    ) -> ConfidenceHeadLogits:

        pairwise_repr = pairwise_repr + self.single_inputs_to_pairwise(single_inputs_repr)

        # interatomic distances - embed and add to pairwise

        interatom_dist = torch.cdist(pred_atom_pos, pred_atom_pos, p = 2)

        dist_from_dist_bins = einx.subtract('b m dist, dist_bins -> b m dist dist_bins', interatom_dist, self.atompair_dist_bins).abs()
        dist_bin_indices = dist_from_dist_bins.argmin(dim = -1)
        pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

        # pairformer stack

        single_repr, pairwise_repr = self.pairformer_stack(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask
        )

        # to logits

        symmetric_pairwise_repr = pairwise_repr + rearrange(pairwise_repr, 'b i j d -> b j i d')
        pde_logits = self.to_pde_logits(symmetric_pairwise_repr)

        plddt_logits = self.to_plddt_logits(single_repr)
        resolved_logits = self.to_resolved_logits(single_repr)

        # they only incorporate pae at some stage of training

        pae_logits = None

        if return_pae_logits:
            pae_logits = self.to_pae_logits(pairwise_repr)

        # return all logits

        return ConfidenceHeadLogits(pae_logits, pde_logits, plddt_logits, resolved_logits)

# main class

class LossBreakdown(NamedTuple):
    diffusion: Float['']
    distogram: Float['']
    pae: Float['']
    pde: Float['']
    plddt: Float['']
    resolved: Float['']
    confidence: Float['']
    diffusion_loss_breakdown: DiffusionLossBreakdown

class Alphafold3(Module):
    """ Algorithm 1 """

    @typecheck
    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_additional_residue_feats,
        dim_template_feats,
        dim_template_model = 64,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair = 16,
        dim_input_embedder_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        dim_token = 768,
        atompair_dist_bins: Float[' dist_bins'] = torch.linspace(3, 20, 37),
        ignore_index = -1,
        num_dist_bins = 38,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        sigma_data = 16,
        loss_confidence_weight = 1e-4,
        loss_distogram_weight = 1e-2,
        loss_diffusion_weight = 4.,
        input_embedder_kwargs: dict = dict(
            atom_transformer_blocks = 3,
            atom_transformer_heads = 4,
            atom_transformer_kwargs = dict()
        ),
        confidence_head_kwargs: dict = dict(
            pairformer_depth = 4
        ),
        template_embedder_kwargs: dict = dict(
            pairformer_stack_depth = 2,
            pairwise_block_kwargs = dict(),
        ),
        msa_module_kwargs: dict = dict(
            depth = 4,
            dim_msa = 64,
            dim_msa_input = None,
            outer_product_mean_dim_hidden = 32,
            msa_pwa_dropout_row_prob = 0.15,
            msa_pwa_heads = 8,
            msa_pwa_dim_head = 32,
            pairwise_block_kwargs = dict()
        ),
        pairformer_stack: dict = dict(
            depth = 48,
            pair_bias_attn_dim_head = 64,
            pair_bias_attn_heads = 16,
            dropout_row_prob = 0.25,
            pairwise_block_kwargs = dict()
        ),
        relative_position_encoding_kwargs: dict = dict(
            r_max = 32,
            s_max = 2,
        ),
        diffusion_module_kwargs: dict = dict(
            single_cond_kwargs = dict(
                num_transitions = 2,
                transition_expansion_factor = 2,
            ),
            pairwise_cond_kwargs = dict(
                num_transitions = 2
            ),
            atom_encoder_depth = 3,
            atom_encoder_heads = 4,
            token_transformer_depth = 24,
            token_transformer_heads = 16,
            atom_decoder_depth = 3,
            atom_decoder_heads = 4
        ),
        edm_kwargs: dict = dict(
            sigma_min = 0.002,
            sigma_max = 80,
            rho = 7,
            P_mean = -1.2,
            P_std = 1.2,
            S_churn = 80,
            S_tmin = 0.05,
            S_tmax = 50,
            S_noise = 1.003,
        )
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

        # input feature embedder

        self.input_embedder = InputFeatureEmbedder(
            dim_atom_inputs = dim_atom_inputs,
            dim_additional_residue_feats = dim_additional_residue_feats,
            atoms_per_window = atoms_per_window,
            dim_atom = dim_atom,
            dim_atompair = dim_atompair,
            dim_token = dim_input_embedder_token,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            **input_embedder_kwargs
        )

        dim_single_inputs = dim_input_embedder_token + dim_additional_residue_feats

        # relative positional encoding
        # used by pairwise in main alphafold2 trunk
        # and also in the diffusion module separately from alphafold3

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            **relative_position_encoding_kwargs
        )

        # f_tokenbond
        self.token_bonds_embedder = nn.Sequential(
            Rearrange('b n m -> b n m 1'),
            nn.Linear(1, dim_pairwise),
            nn.ReLU(),
            nn.Linear(dim_pairwise, dim_pairwise)
        )

        # templates

        self.template_embedder = TemplateEmbedder(
            dim_template_feats = dim_template_feats,
            dim = dim_template_model,
            dim_pairwise = dim_pairwise,
            **template_embedder_kwargs
        )

        # msa

        self.msa_module = MSAModule(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            **msa_module_kwargs
        )

        # main pairformer trunk, 48 layers

        self.pairformer = PairformerStack(
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            **pairformer_stack
        )

        # recycling related

        self.recycle_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_single)
        )

        self.recycle_pairwise = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_pairwise)
        )

        # diffusion

        self.diffusion_module = DiffusionModule(
            dim_pairwise_trunk = dim_pairwise,
            dim_pairwise_rel_pos_feats = dim_pairwise,
            atoms_per_window = atoms_per_window,
            dim_pairwise = dim_pairwise,
            sigma_data = sigma_data,
            dim_atom = dim_atom,
            dim_atompair = dim_atompair,
            dim_token = dim_token,
            dim_single = dim_single + dim_single_inputs,
            **diffusion_module_kwargs
        )

        self.edm = ElucidatedAtomDiffusion(
            self.diffusion_module,
            sigma_data = sigma_data,
            **edm_kwargs
        )

        # logit heads

        self.distogram_head = DistogramHead(
            dim_pairwise = dim_pairwise,
            num_dist_bins = num_dist_bins
        )

        self.confidence_head = ConfidenceHead(
            dim_single_inputs = dim_single_inputs,
            atompair_dist_bins = atompair_dist_bins,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            num_plddt_bins = num_plddt_bins,
            num_pde_bins = num_pde_bins,
            num_pae_bins = num_pae_bins,
            **confidence_head_kwargs
        )

        # loss related

        self.ignore_index = ignore_index
        self.loss_distogram_weight = loss_distogram_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.loss_diffusion_weight = loss_diffusion_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m dai'],
        atom_mask: Bool['b m'],
        atompair_feats: Float['b m m dap'],
        additional_residue_feats: Float['b n rf'],
        msa: Float['b s n d'],
        templates: Float['b t n n dt'],
        template_mask: Bool['b t'],
        num_recycling_steps: int = 1,
        token_bonds: Float['b n n'] | None = None,
        diffusion_add_bond_loss: bool = False,
        diffusion_add_smooth_lddt_loss: bool = False,
        residue_atom_indices: Int['b n'] | None = None,
        num_sample_steps: int | None = None,
        atom_pos: Float['b m 3'] | None = None,
        distance_labels: Int['b n n'] | None = None,
        pae_labels: Int['b n n'] | None = None,
        pde_labels: Int['b n n'] | None = None,
        plddt_labels: Int['b n'] | None = None,
        resolved_labels: Int['b n'] | None = None,
        return_loss_breakdown = False
    ) -> Float['b m 3'] | Float[''] | Tuple[Float[''], LossBreakdown]:

        w = self.atoms_per_window

        # embed inputs

        (
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats
        ) = self.input_embedder(
            atom_inputs = atom_inputs,
            atom_mask = atom_mask,
            atompair_feats = atompair_feats,
            additional_residue_feats = additional_residue_feats
        )

        # relative positional encoding

        relative_position_encoding = self.relative_position_encoding(
            additional_residue_feats = additional_residue_feats
        )

        pairwise_init = pairwise_init + relative_position_encoding

        # Handle token_bonds embedding
        if token_bonds is None:
            # Default to a single chain if token_bonds is not provided
            token_bonds = torch.eye(pairwise_init.shape[1], dtype=pairwise_init.dtype, device=pairwise_init.device)
            token_bonds = token_bonds.unsqueeze(0).expand(pairwise_init.shape[0], -1, -1)
        else:
            token_bonds = token_bonds.to(pairwise_init.dtype).to(pairwise_init.device)

        # Embed token_bonds and add it to pairwise_init
        token_bonds_embed = self.token_bonds_embedder(token_bonds)
        pairwise_init = pairwise_init + token_bonds_embed

        # pairwise mask

        mask = reduce(atom_mask, 'b (n w) -> b n', w = w, reduction = 'any')
        pairwise_mask = einx.logical_and('b i, b j -> b i j', mask, mask)

        # init recycled single and pairwise

        recycled_pairwise = recycled_single = None
        single = pairwise = None

        # for each recycling step

        for _ in range(num_recycling_steps):

            # handle recycled single and pairwise if not first step

            recycled_single = recycled_pairwise = 0.

            if exists(single):
                recycled_single = self.recycle_single(single)

            if exists(pairwise):
                recycled_pairwise = self.recycle_pairwise(pairwise)

            single = single_init + recycled_single
            pairwise = pairwise_init + recycled_pairwise

            # else go through main transformer trunk from alphafold2

            # templates

            embedded_template = self.template_embedder(
                templates = templates,
                template_mask = template_mask,
                pairwise_repr = pairwise,
                mask = mask
            )

            pairwise = embedded_template + pairwise

            # msa

            embedded_msa = self.msa_module(
                msa = msa,
                single_repr = single,
                pairwise_repr = pairwise,
                mask = mask
            )

            pairwise = embedded_msa + pairwise

            # main attention trunk (pairformer)

            single, pairwise = self.pairformer(
                single_repr = single,
                pairwise_repr = pairwise,
                mask = mask
            )

        # determine whether to return loss if any labels were to be passed in
        # otherwise will sample the atomic coordinates

        atom_pos_given = exists(atom_pos)

        confidence_head_labels = (pae_labels, pde_labels, plddt_labels, resolved_labels)
        all_labels = (distance_labels, *confidence_head_labels)

        has_labels = any([*map(exists, all_labels)])

        return_loss = atom_pos_given or has_labels

        # setup all the data necessary for conditioning the diffusion module

        diffusion_cond = dict(
            atom_feats = atom_feats,
            atompair_feats = atompair_feats,
            atom_mask = atom_mask,
            mask = mask,
            single_trunk_repr = single,
            single_inputs_repr = single_inputs,
            pairwise_trunk = pairwise,
            pairwise_rel_pos_feats = relative_position_encoding
        )

        # if neither atom positions or any labels are passed in, sample a structure and return

        if not return_loss:
            return self.edm.sample(
                num_sample_steps = num_sample_steps,
                **diffusion_cond
            )

        # losses default to 0

        distogram_loss = diffusion_loss = confidence_loss = pae_loss = pde_loss = plddt_loss = resolved_loss = self.zero

        # otherwise, noise and make it learn to denoise

        if exists(atom_pos):
            diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm(
                atom_pos,
                additional_residue_feats = additional_residue_feats,
                add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss,
                add_bond_loss = diffusion_add_bond_loss,
                return_denoised_pos = True,
                **diffusion_cond
            )

        # calculate all logits and losses

        ignore = self.ignore_index

        # distogram head

        if exists(distance_labels):
            distance_labels = torch.where(pairwise_mask, distance_labels, ignore)
            distogram_logits = self.distogram_head(pairwise)
            distogram_loss = F.cross_entropy(distogram_logits, distance_labels, ignore_index = ignore)

        # confidence head

        should_call_confidence_head = any([*map(exists, confidence_head_labels)])
        return_pae_logits = exists(pae_labels)

        if should_call_confidence_head:
            assert exists(atom_pos), 'diffusion module needs to have been called'

            assert exists(residue_atom_indices)

            pred_atom_pos = einx.get_at('b (n [w]) c, b n -> b n c', denoised_atom_pos, residue_atom_indices)

            logits = self.confidence_head(
                single_repr = single,
                single_inputs_repr = single_inputs,
                pairwise_repr = pairwise,
                pred_atom_pos = pred_atom_pos,
                mask = mask,
                return_pae_logits = return_pae_logits
            )

            if exists(pae_labels):
                pae_labels = torch.where(pairwise_mask, pae_labels, ignore)
                pae_loss = F.cross_entropy(logits.pae, pae_labels, ignore_index = ignore)

            if exists(pde_labels):
                pde_labels = torch.where(pairwise_mask, pde_labels, ignore)
                pde_loss = F.cross_entropy(logits.pde, pde_labels, ignore_index = ignore)

            if exists(plddt_labels):
                plddt_labels = torch.where(mask, plddt_labels, ignore)
                plddt_loss = F.cross_entropy(logits.plddt, plddt_labels, ignore_index = ignore)

            if exists(resolved_labels):
                resolved_labels = torch.where(mask, resolved_labels, ignore)
                resolved_loss = F.cross_entropy(logits.resolved, resolved_labels, ignore_index = ignore)

            confidence_loss = pae_loss + pde_loss + plddt_loss + resolved_loss

        # combine all the losses

        loss = (
            distogram_loss * self.loss_distogram_weight +
            diffusion_loss * self.loss_diffusion_weight +
            confidence_loss * self.loss_confidence_weight
        )

        if not return_loss_breakdown:
            return loss

        loss_breakdown = LossBreakdown(
            pae = pae_loss,
            pde = pde_loss,
            plddt = plddt_loss,
            resolved = resolved_loss,
            distogram = distogram_loss,
            diffusion = diffusion_loss,
            confidence = confidence_loss,
            diffusion_loss_breakdown = diffusion_loss_breakdown
        )

        return loss, loss_breakdown
