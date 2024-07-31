from __future__ import annotations

from math import pi, sqrt
from pathlib import Path
from itertools import product
from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from loguru import logger

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from typing import List, Literal, Tuple, NamedTuple, Dict, Callable

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    typecheck
)

from alphafold3_pytorch.attention import (
    Attention,
    pad_at_dim,
    slice_at_dim,
    pad_or_slice_to,
    pad_to_multiple,
    concat_previous_window,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.inputs import (
    IS_MOLECULE_TYPES,
    IS_PROTEIN_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
    IS_BIOMOLECULE_INDICES,
    NUM_MOLECULE_IDS,
    ADDITIONAL_MOLECULE_FEATS
)


IS_DNA_INDEX = 1
IS_RNA_INDEX = 2

IS_PROTEIN, IS_DNA, IS_RNA, IS_LIGAND, IS_METAL_ION = map(
    lambda x: IS_MOLECULE_TYPES - x if x < 0 else x, [
        IS_PROTEIN_INDEX, IS_DNA_INDEX, IS_RNA_INDEX, IS_LIGAND_INDEX, IS_METAL_ION_INDEX])


from frame_averaging_pytorch import FrameAverage

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from colt5_attention import ConditionalRoutedAttention

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from importlib.metadata import version

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

"""
global ein notation:

b - batch
ba - batch with augmentation
h - heads
n - molecule sequence length
i - molecule sequence length (source)
j - molecule sequence length (target)
l - present (i.e., non-missing) atom sequence length
m - atom sequence length
nw - windowed sequence length
d - feature dimension
ds - feature dimension (single)
dp - feature dimension (pairwise)
dap - feature dimension (atompair)
dapi - feature dimension (atompair input)
da - feature dimension (atom)
dai - feature dimension (atom input)
dtf - additional token feats derived from msa (f_profile and f_deletion_mean)
t - templates
s - msa
r - registers
"""

"""
additional_token_feats: [*]
- concatted to the single rep

0: f_profile
1: f_deletion_mean
"""

"""
additional_molecule_feats: [*, 5]:
- used for deriving relative positions

0: molecule_index
1: token_index
2: asym_id
3: entity_id
4: sym_id
"""

"""
is_molecule_types: [*, 5]

0: is_protein
1: is_rna
2: is_dna
3: is_ligand
4: is_metal_ions_or_misc
"""

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(x, *args, **kwargs):
    return x

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def divisible_by(num, den):
    return (num % den) == 0

def compact(*args):
    return tuple(filter(exists, args))

# tensor helpers

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def exclusive_cumsum(t, dim = -1):
    return t.cumsum(dim = dim) - t

# decorators

def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)
    return inner

def save_args_and_kwargs(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        self._args_and_kwargs = (args, kwargs)
        self._version = version('alphafold3_pytorch')

        return fn(self, *args, **kwargs)
    return inner

@typecheck
def pad_and_window(
    t: Float['b n ...'] | Int['b n ...'],
    window_size: int
):
    t = pad_to_multiple(t, window_size, dim = 1)
    t = rearrange(t, 'b (n w) ... -> b n w ...', w = window_size)
    return t

# to atompair input functions

@typecheck
def atom_ref_pos_to_atompair_inputs(
    atom_ref_pos: Float['... m 3'],
    atom_ref_space_uid: Int['... m'],
) -> Float['... m m 5']:

    # Algorithm 5 - lines 2-6
    # allow for either batched or single

    atom_ref_pos, unpack_one = pack_one(atom_ref_pos, '* m c')
    atom_ref_space_uid, _ = pack_one(atom_ref_space_uid, '* m')

    assert atom_ref_pos.shape[0] == atom_ref_space_uid.shape[0]

    # line 2

    pairwise_rel_pos = einx.subtract('b i c, b j c -> b i j c', atom_ref_pos, atom_ref_pos)

    # line 3

    same_ref_space_mask = einx.equal('b i, b j -> b i j', atom_ref_space_uid, atom_ref_space_uid)

    # line 5 - pairwise inverse squared distance

    atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim = -1, p = 2) ** 2) ** -1

    # concat all into atompair_inputs for projection into atompair_feats within Alphafold3

    atompair_inputs, _ = pack((
        pairwise_rel_pos,
        atom_inv_square_dist,
        same_ref_space_mask.float(),
    ), 'b i j *')

    # mask out

    atompair_inputs = einx.where(
        'b i j, b i j dapi, -> b i j dapi',
        same_ref_space_mask, atompair_inputs, 0.
    )

    # reconstitute optional batch dimension

    atompair_inputs = unpack_one(atompair_inputs, '* i j dapi')

    # return

    return atompair_inputs

# packed atom representation functions

@typecheck
def lens_to_mask(
    lens: Int['b ...'],
    max_len: int | None = None
) -> Bool['... m']:

    device = lens.device
    if not exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device = device)
    return einx.less('m, ... -> ... m', arange, lens)

@typecheck
def mean_pool_with_lens(
    feats: Float['b m d'],
    lens: Int['b n']
) -> Float['b n d']:

    seq_len = feats.shape[1]

    mask = lens > 0
    assert (lens.sum(dim = -1) <= seq_len).all(), 'one of the lengths given exceeds the total sequence length of the features passed in'

    cumsum_feats = feats.cumsum(dim = 1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value = 0.)

    cumsum_indices = lens.cumsum(dim = 1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value = 0)

    sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    avg = einx.divide('b n d, b n', summed, lens.clamp(min = 1))
    avg = einx.where('b n, b n d, -> b n d', mask, avg, 0.)
    return avg

@typecheck
def repeat_consecutive_with_lens(
    feats: Float['b n ...'] | Bool['b n ...'] | Bool['b n'] | Int['b n'],
    lens: Int['b n'],
    mask_value: float | int | bool | None = None,
) -> Float['b m ...'] | Bool['b m ...'] | Bool['b m'] | Int['b m']:

    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device = device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add('w, b n -> b n w', arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.sum(dim = -1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter

    seq_arange = torch.arange(seq, device = device)
    seq_arange = repeat(seq_arange, 'n -> (n w)', w = window_size)

    output_indices = einx.set_at('b [m],  b nw, nw -> b [m]', output_indices, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    # final mask

    if mask_value is None:
        mask_value = False if dtype == torch.bool else 0

    output = einx.where(
        'b n, b n ..., -> b n ...',
        output_mask, output, mask_value
    )

    return output

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

        self.left_right_proj = nn.Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim = -1)
        )

        self.left_right_gate = LinearNoBias(dim, dim_hidden * 2)

        self.out_gate = LinearNoBias(dim, dim_hidden)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'incoming':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(dim_hidden)

        self.to_out = Sequential(
            LinearNoBias(dim_hidden, dim),
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

        left, right = self.left_right_proj(x).chunk(2, dim = -1)

        if exists(mask):
            left = left * mask
            right = right * mask

        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)

        out_gate = self.out_gate(x).sigmoid()
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
        num_memory_kv = 0,
        **attn_kwargs
    ):
        super().__init__()

        self.window_size = window_size

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            num_memory_kv = num_memory_kv,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        to_attn_bias_linear = LinearNoBias(dim_pairwise, heads)
        nn.init.zeros_(to_attn_bias_linear.weight)

        self.to_attn_bias = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            to_attn_bias_linear,
            Rearrange('b ... h -> b h ...')
        )

    @typecheck
    def forward(
        self,
        single_repr: Float['b n ds'],
        *,
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        attn_bias: Float['b n n'] | Float['b nw w (w*2)'] | None = None,
        **kwargs
    ) -> Float['b n ds']:

        w, has_window_size = self.window_size, exists(self.window_size)

        # take care of windowing logic
        # for sequence-local atom transformer

        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size = w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size = w)
        else:
            assert not windowed_pairwise, 'cannot pass in windowed pairwise repr if no window_size given to AttentionPairBias'
            assert not exists(windowed_attn_bias) or not windowed_attn_bias, 'cannot pass in windowed attention bias if no window_size set for AttentionPairBias'

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')
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
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            **kwargs
        )

        out = unpack_one(out)

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

        # maybe masked mean for outer product

        if exists(msa_mask):
            a = einx.multiply('b s i d, b s -> b s i d', a, msa_mask.float())
            b = einx.multiply('b s j e, b s -> b s j e', b, msa_mask.float())

            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')

            num_msa = reduce(msa_mask.float(), '... s -> ...', 'sum')

            outer_product_mean = einx.divide('b i j d e, b', outer_product, num_msa.clamp(min = self.eps))
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
            outer_product_mean = outer_product / num_msa

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
        max_num_msa: int | None = None,
        layerscale_output: bool = True
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

        self.layerscale_output = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.

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

        # account for no msa

        if exists(msa_mask):
            has_msa = reduce(msa_mask, 'b s -> b', 'any')

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

        if exists(msa_mask):
            pairwise_repr = einx.where(
                'b, b ..., -> b ...',
                has_msa, pairwise_repr, 0.
            )

        return pairwise_repr * self.layerscale_output

# pairformer stack

class PairformerStack(Module):
    """ Algorithm 17 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 48,
        recurrent_depth = 1, # effective depth will be depth * recurrent_depth
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        num_register_tokens = 0,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict()
    ):
        super().__init__()
        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob,
            **pair_bias_attn_kwargs
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

        # https://arxiv.org/abs/2405.16039 and https://arxiv.org/abs/2405.15071
        # although possibly recycling already takes care of this

        assert recurrent_depth > 0
        self.recurrent_depth = recurrent_depth

        self.num_registers = num_register_tokens
        self.has_registers = num_register_tokens > 0

        if self.has_registers:
            self.single_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_single))
            self.pairwise_row_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))
            self.pairwise_col_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        # prepend register tokens

        if self.has_registers:
            batch_size, num_registers = single_repr.shape[0], self.num_registers
            single_registers = repeat(self.single_registers, 'r d -> b r d', b = batch_size)
            single_repr = torch.cat((single_registers, single_repr), dim = 1)

            row_registers = repeat(self.pairwise_row_registers, 'r d -> b r n d', b = batch_size, n = pairwise_repr.shape[-2])
            pairwise_repr = torch.cat((row_registers, pairwise_repr), dim = 1)
            col_registers = repeat(self.pairwise_col_registers, 'r d -> b n r d', b = batch_size, n = pairwise_repr.shape[1])
            pairwise_repr = torch.cat((col_registers, pairwise_repr), dim = 2)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)

        # main transformer block layers

        for _ in range(self.recurrent_depth):
            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in self.layers:

                pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

                single_repr = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask) + single_repr
                single_repr = single_transition(single_repr) + single_repr

        # splice out registers

        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]

        return single_repr, pairwise_repr

# embedding related

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
        additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    ) -> Float['b n n dp']:

        device = additional_molecule_feats.device

        res_idx, token_idx, asym_id, entity_id, sym_id = additional_molecule_feats.unbind(dim = -1)
        
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
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim = -1, keepdim = True).indices
            one_hots = F.one_hot(indices.long(), num_classes = len(bins))
            return one_hots.float()

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
        eps = 1e-5,
        layerscale_output = True
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

        self.layerscale = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.

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

        v, unpack_one = pack_one(v, '* i j d')

        has_templates = reduce(template_mask, 'b t -> b', 'any')

        if exists(mask):
            mask = repeat(mask, 'b n -> (b t) n', t = num_templates)

        for block in self.pairformer_stack:
            v = block(
                pairwise_repr = v,
                mask = mask
            ) + v

        u = self.final_norm(v)

        u = unpack_one(u)

        # masked mean pool template repr

        u = einx.where(
            'b t, b t ..., -> b t ...',
            template_mask, u, 0.
        )

        num = reduce(u, 'b t i j d -> b i j d', 'sum')
        den = reduce(template_mask.float(), 'b t -> b', 'sum')

        avg_template_repr = einx.divide('b i j d, b -> b i j d', num, den.clamp(min = self.eps))

        out = self.to_out(avg_template_repr)

        out = einx.where(
            'b, b ..., -> b ...',
            has_templates, out, 0.
        )

        return out * self.layerscale

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
        attn_num_memory_kv = False,
        trans_expansion_factor = 2,
        num_register_tokens = 0,
        serial = False,
        add_residual = True,
        use_linear_attn = False,
        linear_attn_kwargs = dict(
            heads = 8,
            dim_head = 16
        ),
        use_colt5_attn = False,
        colt5_attn_kwargs = dict(
            heavy_dim_head = 64,
            heavy_heads = 8,
            num_heavy_tokens_q = 512,
            num_heavy_tokens_kv = 512
        )

    ):
        super().__init__()
        self.attn_window_size = attn_window_size

        dim_single_cond = default(dim_single_cond, dim)

        layers = ModuleList([])

        for _ in range(depth):

            linear_attn = None

            if use_linear_attn:
                linear_attn = TaylorSeriesLinearAttn(
                    dim = dim,
                    prenorm = True,
                    gate_value_heads = True,
                    **linear_attn_kwargs
                )

            colt5_attn = None

            if use_colt5_attn:
                colt5_attn = ConditionalRoutedAttention(
                    dim = dim,
                    has_light_attn = False,
                    **colt5_attn_kwargs
                )

            pair_bias_attn = AttentionPairBias(
                dim = dim,
                dim_pairwise = dim_pairwise,
                heads = heads,
                window_size = attn_window_size,
                num_memory_kv = attn_num_memory_kv,
                **attn_pair_bias_kwargs
            )

            transition = Transition(
                dim = dim,
                expansion_factor = trans_expansion_factor
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
                linear_attn,
                colt5_attn,
                conditionable_pair_bias,
                conditionable_transition
            ]))

        self.layers = layers

        self.serial = serial
        self.add_residual = add_residual

        self.has_registers = num_register_tokens > 0
        self.num_registers = num_register_tokens

        if self.has_registers:
            assert not exists(attn_window_size), 'register tokens disabled for windowed attention'

            self.registers = nn.Parameter(torch.zeros(num_register_tokens, dim))

    @typecheck
    def forward(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):
        w = self.attn_window_size
        has_windows = exists(w)

        serial = self.serial

        # handle windowing

        pairwise_is_windowed = pairwise_repr.ndim == 5

        if has_windows and not pairwise_is_windowed:
            pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size = w)

        # register tokens

        if self.has_registers:
            num_registers = self.num_registers
            registers = repeat(self.registers, 'r d -> b r d', b = noised_repr.shape[0])
            noised_repr, registers_ps = pack((registers, noised_repr), 'b * d')

            single_repr = F.pad(single_repr, (0, 0, num_registers, 0), value = 0.)
            pairwise_repr = F.pad(pairwise_repr, (0, 0, num_registers, 0, num_registers, 0), value = 0.)

            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)

        # main transformer

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                noised_repr = linear_attn(noised_repr, mask = mask) + noised_repr

            if exists(colt5_attn):
                noised_repr = colt5_attn(noised_repr, mask = mask) + noised_repr

            attn_out = attn(
                noised_repr,
                cond = single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask,
                windowed_mask = windowed_mask
            )

            if serial:
                noised_repr = attn_out + noised_repr

            ff_out = transition(
                noised_repr,
                cond = single_repr
            )

            if serial:
                noised_repr = ff_out + noised_repr

            # in the algorithm, they omitted the residual, but it could be an error
            # attn + ff + residual was used in GPT-J and PaLM, but later found to be unstable configuration, so it seems unlikely attn + ff would work
            # but in the case they figured out something we have not, you can use their exact formulation by setting `serial = False` and `add_residual = False`

            residual = noised_repr if self.add_residual else 0.

            if not serial:
                ff_out = ff_out + attn_out + residual

        # splice out registers

        if self.has_registers:
            _, noised_repr = unpack(noised_repr, registers_ps, 'b * d')

        return noised_repr

class AtomToTokenPooler(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Sequential(
            LinearNoBias(dim, dim_out),
            nn.ReLU()
        )

    @typecheck
    def forward(
        self,
        *,
        atom_feats: Float['b m da'],
        atom_mask: Bool['b m'],
        molecule_atom_lens: Int['b n']
    ) -> Float['b n ds']:

        atom_feats = self.proj(atom_feats)
        tokens = mean_pool_with_lens(atom_feats, molecule_atom_lens)
        return tokens

class DiffusionModule(Module):
    """ Algorithm 20 """

    @typecheck
    def __init__(
        self,
        *,
        dim_pairwise_trunk,
        dim_pairwise_rel_pos_feats,
        atoms_per_window = 27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the molecule or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms
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
        serial = False,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
        use_linear_attn = False,
        linear_attn_kwargs: dict = dict(
            heads = 8,
            dim_head = 16
        )
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

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
            dim_pairwise = dim_pairwise,
            **pairwise_cond_kwargs
        )

        # atom attention encoding related modules

        self.atom_pos_to_atom_feat = LinearNoBias(3, dim_atom)

        self.missing_atom_feat = nn.Parameter(torch.zeros(dim_atom))

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
            LinearNoBias(dim_atom, dim_atompair * 2),
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
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            **atom_encoder_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token
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
            serial = serial,
            **token_transformer_kwargs
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
            serial = serial,
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            **atom_decoder_kwargs
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
        atompair_feats: Float['b m m dap'] | Float['b nw w (w*2) dap'],
        atom_mask: Bool['b m'],
        times: Float[' b'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
        molecule_atom_lens: Int['b n'],
        atom_parent_ids: Int['b m'] | None = None,
        missing_atom_mask: Bool['b m']| None = None
    ):
        w = self.atoms_per_window
        device = noised_atom_pos.device

        batch_size, seq_len = single_trunk_repr.shape[:2]
        atom_seq_len = atom_feats.shape[1]

        conditioned_single_repr = self.single_conditioner(
            times = times,
            single_trunk_repr = single_trunk_repr,
            single_inputs_repr = single_inputs_repr
        )

        conditioned_pairwise_repr = self.pairwise_conditioner(
            pairwise_trunk = pairwise_trunk,
            pairwise_rel_pos_feats = pairwise_rel_pos_feats
        )

        # lines 7-14 in Algorithm 5

        atom_feats_cond = atom_feats

        # the most surprising part of the paper; no geometric biases!

        noised_atom_pos_feats = self.atom_pos_to_atom_feat(noised_atom_pos)

        # for missing atoms, replace the noise atom pos features with a missing embedding

        if exists(missing_atom_mask):
            noised_atom_pos_feats = einx.where('b m, d, b m d -> b m d', missing_atom_mask, self.missing_atom_feat, noised_atom_pos_feats)

        # sum the noised atom position features to the atom features

        atom_feats = noised_atom_pos_feats + atom_feats

        # condition atom feats cond (cl) with single repr

        single_repr_cond = self.single_repr_to_atom_feat_cond(conditioned_single_repr)

        single_repr_cond = repeat_consecutive_with_lens(single_repr_cond, molecule_atom_lens)
        single_repr_cond = pad_or_slice_to(single_repr_cond, length = atom_feats_cond.shape[1], dim = 1)

        atom_feats_cond = single_repr_cond + atom_feats_cond

        # window the atom pair features before passing to atom encoder and decoder if necessary

        atompair_is_windowed = atompair_feats.ndim == 5

        if not atompair_is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = self.atoms_per_window)

        # condition atompair feats with pairwise repr

        pairwise_repr_cond = self.pairwise_repr_to_atompair_feat_cond(conditioned_pairwise_repr)

        indices = torch.arange(seq_len, device = device)
        indices = repeat(indices, 'n -> b n', b = batch_size)

        indices = repeat_consecutive_with_lens(indices, molecule_atom_lens)
        indices = pad_or_slice_to(indices, atom_seq_len, dim = -1)
        indices = pad_and_window(indices, w)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, 'b n w -> b n w 1', w = w)
        col_indices = rearrange(col_indices, 'b n w -> b n 1 w', w = w)

        col_indices = concat_previous_window(col_indices, dim_seq = 1, dim_window = -1)
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        pairwise_repr_cond = einx.get_at('b [i j] dap, b nw w1 w2, b nw w1 w2 -> b nw w1 w2 dap', pairwise_repr_cond, row_indices, col_indices)

        atompair_feats = pairwise_repr_cond + atompair_feats

        # condition atompair feats further with single atom repr

        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)
        atom_repr_cond = pad_and_window(atom_repr_cond, w)

        atom_repr_cond_row, atom_repr_cond_col = atom_repr_cond.chunk(2, dim = -1)

        atom_repr_cond_col = concat_previous_window(atom_repr_cond_col, dim_seq = 1, dim_window = 2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap -> b nw w1 w2 dap', atompair_feats, atom_repr_cond_col)

        # furthermore, they did one more MLP on the atompair feats for attention biasing in atom transformer

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        # take care of restricting atom attention to be intra molecular, if the atom_parent_ids were passed in

        windowed_mask = None

        if exists(atom_parent_ids):
            atom_parent_ids_rows = pad_and_window(atom_parent_ids, w)
            atom_parent_ids_columns = concat_previous_window(atom_parent_ids_rows, dim_seq = 1, dim_window = 2)

            windowed_mask = einx.equal('b n i, b n j -> b n i j', atom_parent_ids_rows, atom_parent_ids_columns)

        # atom encoder

        atom_feats = self.atom_encoder(
            atom_feats,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_feats_skip = atom_feats

        tokens = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        # token transformer

        tokens = self.cond_tokens_with_cond_single(conditioned_single_repr) + tokens

        tokens = self.token_transformer(
            tokens,
            mask = mask,
            single_repr = conditioned_single_repr,
            pairwise_repr = conditioned_pairwise_repr,
        )

        tokens = self.attended_token_norm(tokens)

        # atom decoder

        atom_decoder_input = self.tokens_to_atom_decoder_input_cond(tokens)

        atom_decoder_input = repeat_consecutive_with_lens(atom_decoder_input, molecule_atom_lens)
        atom_decoder_input = pad_or_slice_to(atom_decoder_input, length = atom_feats_skip.shape[1], dim = 1)

        atom_decoder_input = atom_decoder_input + atom_feats_skip

        atom_feats = self.atom_decoder(
            atom_decoder_input,
            mask = atom_mask,
            windowed_mask = windowed_mask,
            single_repr = atom_feats_cond,
            pairwise_repr = atompair_feats
        )

        atom_pos_update = self.atom_feat_to_atom_pos_update(atom_feats)

        return atom_pos_update

# elucidated diffusion model adapted for atom position diffusing
# from Karras et al.
# https://arxiv.org/abs/2206.00364

class DiffusionLossBreakdown(NamedTuple):
    diffusion_mse: Float['']
    diffusion_bond: Float['']
    diffusion_smooth_lddt: Float['']

class ElucidatedAtomDiffusionReturn(NamedTuple):
    loss: Float['']
    denoised_atom_pos: Float['ba m 3']
    loss_breakdown: DiffusionLossBreakdown
    noise_sigmas: Float[' ba']

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
        P_std = 1.5,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        smooth_lddt_loss_kwargs: dict = dict(),
        weighted_rigid_align_kwargs: dict = dict()
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

        # weighted rigid align

        self.weighted_rigid_align = WeightedRigidAlign(**weighted_rigid_align_kwargs)

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
        atom_mask: Bool['b m'] | None = None,
        num_sample_steps = None,
        clamp = False,
        use_tqdm_pbar = True,
        tqdm_pbar_title = 'sampling time step',
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

        maybe_tqdm_wrapper = tqdm if use_tqdm_pbar else identity

        for sigma, sigma_next, gamma in maybe_tqdm_wrapper(sigmas_and_gammas, desc = tqdm_pbar_title):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(atom_pos_hat, sigma_hat, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(atom_pos_next, sigma_next, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = atom_pos_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            atom_pos = atom_pos_next

        if clamp:
            atom_pos = atom_pos.clamp(-1., 1.)

        return atom_pos

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(
        self,
        atom_pos_ground_truth: Float['b m 3'],
        atom_mask: Bool['b m'],
        atom_feats: Float['b m da'],
        atompair_feats: Float['b m m dap'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
        molecule_atom_lens: Int['b n'],
        missing_atom_mask: Bool['b m'] | None = None,
        atom_parent_ids: Int['b m'] | None = None,
        return_denoised_pos = False,
        is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'] | None = None,
        additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}'] | None = None,
        add_smooth_lddt_loss = False,
        add_bond_loss = False,
        nucleotide_loss_weight = 5.,
        ligand_loss_weight = 10.,
        return_loss_breakdown = False,
    ) -> ElucidatedAtomDiffusionReturn:

        # diffusion loss

        batch_size = atom_pos_ground_truth.shape[0]

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(atom_pos_ground_truth)

        noised_atom_pos = atom_pos_ground_truth + padded_sigmas * noise  # alphas are 1. in the paper

        denoised_atom_pos = self.preconditioned_network_forward(
            noised_atom_pos,
            sigmas,
            network_condition_kwargs = dict(
                atom_feats = atom_feats,
                atom_mask = atom_mask,
                missing_atom_mask = missing_atom_mask,
                atompair_feats = atompair_feats,
                atom_parent_ids = atom_parent_ids,
                mask = mask,
                single_trunk_repr = single_trunk_repr,
                single_inputs_repr = single_inputs_repr,
                pairwise_trunk = pairwise_trunk,
                pairwise_rel_pos_feats = pairwise_rel_pos_feats,
                molecule_atom_lens = molecule_atom_lens
            )
        )

        # total loss, for accumulating all auxiliary losses

        total_loss = 0.

        # if additional molecule feats is provided
        # calculate the weights for mse loss (wl)

        align_weights = atom_pos_ground_truth.new_ones(atom_pos_ground_truth.shape[:2])

        if exists(is_molecule_types):
            is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim = -1)

            is_nucleotide_or_ligand_fields = tuple(repeat_consecutive_with_lens(t, molecule_atom_lens) for t in is_nucleotide_or_ligand_fields)
            is_nucleotide_or_ligand_fields = tuple(pad_or_slice_to(t, length = align_weights.shape[-1], dim = -1) for t in is_nucleotide_or_ligand_fields)

            _, atom_is_dna, atom_is_rna, atom_is_ligand, _ = is_nucleotide_or_ligand_fields

            # section 3.7.1 equation 4

            # upweighting of nucleotide and ligand atoms is additive per equation 4

            align_weights = torch.where(atom_is_dna | atom_is_rna, 1 + nucleotide_loss_weight, align_weights)
            align_weights = torch.where(atom_is_ligand, 1 + ligand_loss_weight, align_weights)

        # section 3.7.1 equation 2 - weighted rigid aligned ground truth

        atom_pos_aligned_ground_truth = self.weighted_rigid_align(
            pred_coords = denoised_atom_pos,
            true_coords = atom_pos_ground_truth,
            weights = align_weights,
            mask = atom_mask
        )

        # main diffusion mse loss

        losses = F.mse_loss(denoised_atom_pos, atom_pos_aligned_ground_truth, reduction = 'none') / 3.
        losses = einx.multiply('b m c, b m -> b m c',  losses, align_weights)

        # regular loss weight as defined in EDM paper

        loss_weights = self.loss_weight(padded_sigmas)

        losses = losses * loss_weights

        # if there are missing atoms, update the atom mask to not include them in the loss

        if exists(missing_atom_mask):
            atom_mask = atom_mask & ~ missing_atom_mask

        # account for atom mask

        mse_loss = losses[atom_mask].mean()

        total_loss = total_loss + mse_loss

        # proposed extra bond loss during finetuning

        bond_loss = self.zero

        if add_bond_loss:
            atompair_mask = einx.logical_and('b i, b j -> b i j', atom_mask, atom_mask)

            denoised_cdist = torch.cdist(denoised_atom_pos, denoised_atom_pos, p = 2)
            normalized_cdist = torch.cdist(atom_pos_ground_truth, atom_pos_ground_truth, p = 2)

            bond_losses = F.mse_loss(denoised_cdist, normalized_cdist, reduction = 'none')
            bond_losses = bond_losses * loss_weights

            bond_loss = bond_losses[atompair_mask].mean()

            total_loss = total_loss + bond_loss

        # proposed auxiliary smooth lddt loss

        smooth_lddt_loss = self.zero

        if add_smooth_lddt_loss:
            assert exists(is_molecule_types)

            smooth_lddt_loss = self.smooth_lddt_loss(
                denoised_atom_pos,
                atom_pos_ground_truth,
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
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms
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

        # Calculate masked averaging
        lddt_sum = (eps * mask).sum(dim=(-1, -2))
        lddt_count = mask.sum(dim=(-1, -2))
        lddt = lddt_sum / lddt_count.clamp(min=1)

        return 1. - lddt.mean()

class WeightedRigidAlign(Module):
    """ Algorithm 28 """

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],       # predicted coordinates
        true_coords: Float['b n 3'],       # true coordinates
        weights: Float['b n'],             # weights for each atom
        mask: Bool['b n'] | None = None    # mask for variable lengths
    ) -> Float['b n 3']:

        batch_size, num_points, dim = pred_coords.shape

        if exists(mask):
            # zero out all predicted and true coordinates where not an atom
            pred_coords = einx.where('b n, b n c, -> b n c', mask, pred_coords, 0.)
            true_coords = einx.where('b n, b n c, -> b n c', mask, true_coords, 0.)
            weights = einx.where('b n, b n, -> b n', mask, weights, 0.)

        # Take care of weights broadcasting for coordinate dimension
        weights = rearrange(weights, 'b n -> b n 1')

        # Compute weighted centroids
        true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

        # Center the coordinates
        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            logger.warning(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        # Compute the weighted covariance matrix
        cov_matrix = einsum(weights * true_coords_centered, pred_coords_centered, 'b n i, b n j -> b i j')

        # Compute the SVD of the covariance matrix
        U, S, V = torch.svd(cov_matrix)
        U_T = U.transpose(-2, -1)

        # Catch ambiguous rotation by checking the magnitude of singular values
        if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
            logger.warning(
                "Warning: Excessively low rank of "
                + "cross-correlation between aligned point clouds. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        det = torch.det(einsum(V, U_T, 'b i j, b j k -> b i k'))
        # Ensure proper rotation matrix with determinant 1
        diag = torch.eye(dim, dtype=det.dtype, device=det.device)[None].repeat(batch_size, 1, 1)
        diag[:, -1, -1] = det
        rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

        # Apply the rotation and translation
        true_aligned_coords = einsum(rot_matrix, true_coords_centered, 'b i j, b n j -> b n i') + pred_centroid
        true_aligned_coords.detach_()

        return true_aligned_coords

class ExpressCoordinatesInFrame(Module):
    """ Algorithm  29 """

    def __init__(
        self,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps

    @typecheck
    def forward(
        self,
        coords: Float['b m 3'],
        frame: Float['b m 3 3'] | Float['b 3 3'] | Float['3 3']
    ) -> Float['b m 3']:
        """
        coords: coordinates to be expressed in the given frame
        frame: frame defined by three points
        """

        if frame.ndim == 2:
            frame = rearrange(frame, 'fr fc -> 1 1 fr fc')
        elif frame.ndim == 3:
            frame = rearrange(frame, 'b fr fc -> b 1 fr fc')

        # Extract frame atoms
        a, b, c = frame.unbind(dim=-1)
        w1 = F.normalize(a - b, dim=-1, eps=self.eps)
        w2 = F.normalize(c - b, dim=-1, eps=self.eps)

        # Build orthonormal basis
        e1 = F.normalize(w1 + w2, dim=-1, eps=self.eps)
        e2 = F.normalize(w2 - w1, dim=-1, eps=self.eps)
        e3 = torch.cross(e1, e2, dim=-1)

        # Project onto frame basis
        d = coords - b
        transformed_coords = torch.stack(
            [
                einsum(d, e1, '... i, ... i -> ...'),
                einsum(d, e2, '... i, ... i -> ...'),
                einsum(d, e3, '... i, ... i -> ...'),
            ],
            dim=-1,
        )

        return transformed_coords

class ComputeAlignmentError(Module):
    """ Algorithm 30 """

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8
    ):
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
    ) -> Float['b n n']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        pred_frames: predicted frames
        true_frames: true frames
        """
        num_res = pred_coords.shape[1]
        
        pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
        seq2pair = partial(rearrange, pattern='b (n m) ... -> b n m ...', n = num_res, m = num_res)
        
        pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = num_res))
        pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = num_res))
        pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = num_res))
        pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = num_res))
        
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)

        # Compute alignment errors
        alignment_errors = torch.sqrt(
            torch.sum((pred_coords_transformed - true_coords_transformed) ** 2, dim=-1) + self.eps
        )
        
        alignment_errors = seq2pair(alignment_errors)

        return alignment_errors

class CentreRandomAugmentation(Module):
    """ Algorithm 19 """

    @typecheck
    def __init__(self, trans_scale: float = 1.0):
        super().__init__()
        self.trans_scale = trans_scale
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    @typecheck
    def forward(
        self,
        coords: Float['b n 3'],
        mask: Bool['b n'] | None = None
    ) -> Float['b n 3']:
        """
        coords: coordinates to be augmented
        """
        batch_size = coords.shape[0]

        # Center the coordinates
        # Accounting for masking

        if exists(mask):
            coords = einx.where('b n, b n c, -> b n c', mask, coords, 0.)
            num = reduce(coords, 'b n c -> b c', 'sum')
            den = reduce(mask.float(), 'b n -> b', 'sum')
            coords_mean = einx.divide('b c, b -> b 1 c', num, den.clamp(min = 1.))
        else:
            coords_mean = coords.mean(dim = 1, keepdim = True)

        centered_coords = coords - coords_mean

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, 'b c -> b 1 c')

        # Apply rotation and translation
        augmented_coords = einsum(centered_coords, rotation_matrix, 'b n i, b j i -> b n j') + translation_vector

        return augmented_coords

    @typecheck
    def _random_rotation_matrix(self, batch_size: int) -> Float['b 3 3']:
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device = self.device) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device = self.device)
        rotation_matrix = repeat(eye, 'i j -> b i j', b = batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    @typecheck
    def _random_translation_vector(self, batch_size: int) -> Float['b 3']:
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device = self.device) * self.trans_scale
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
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        dim_additional_token_feats = 2,
        num_molecule_types = NUM_MOLECULE_IDS,
        atom_transformer_blocks = 3,
        atom_transformer_heads = 4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window

        self.to_atom_feats = LinearNoBias(dim_atom_inputs, dim_atom)

        self.to_atompair_feats = LinearNoBias(dim_atompair_inputs, dim_atompair)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, dim_atompair * 2),
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
            dim_out = dim_token
        )

        dim_single_input = dim_token + dim_additional_token_feats

        self.dim_additional_token_feats = dim_additional_token_feats

        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(dim_single_input, dim_pairwise)

        # this accounts for the `restypes` in the additional molecule features

        self.single_molecule_embed = nn.Embedding(num_molecule_types, dim_single)
        self.pairwise_molecule_embed = nn.Embedding(num_molecule_types, dim_pairwise)

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m dai'],
        atompair_inputs: Float['b m m dapi'] | Float['b nw w1 w2 dapi'],
        atom_mask: Bool['b m'],
        molecule_atom_lens: Int['b n'],
        molecule_ids: Int['b n'],
        additional_token_feats: Float['b n {self.dim_additional_token_feats}'] | None = None,

    ) -> EmbeddedInputs:

        w = self.atoms_per_window

        atom_feats = self.to_atom_feats(atom_inputs)
        atompair_feats = self.to_atompair_feats(atompair_inputs)

        # window the atom pair features before passing to atom encoder and decoder

        is_windowed = atompair_inputs.ndim == 5

        if not is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = w)

        # condition atompair with atom repr

        atom_feats_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)

        atom_feats_cond = pad_and_window(atom_feats_cond, w)

        atom_feats_cond_row, atom_feats_cond_col = atom_feats_cond.chunk(2, dim = -1)
        atom_feats_cond_col = concat_previous_window(atom_feats_cond_col, dim_seq = 1, dim_window = -2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap',atompair_feats, atom_feats_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap',atompair_feats, atom_feats_cond_col)

        # initial atom transformer

        atom_feats = self.atom_transformer(
            atom_feats,
            single_repr = atom_feats,
            pairwise_repr = atompair_feats
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        single_inputs = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        if exists(additional_token_feats):
            single_inputs = torch.cat((
                single_inputs,
                additional_token_feats
            ), dim = -1)

        single_init = self.single_input_to_single_init(single_inputs)
        pairwise_init = self.single_input_to_pairwise_init(single_inputs)

        # account for molecule id (restypes)

        molecule_ids = torch.where(molecule_ids >= 0, molecule_ids, 0) # account for padding

        single_molecule_embed = self.single_molecule_embed(molecule_ids)

        pairwise_molecule_embed = self.pairwise_molecule_embed(molecule_ids)
        pairwise_molecule_embed = einx.add('b i dp, b j dp -> b i j dp', pairwise_molecule_embed, pairwise_molecule_embed)

        # sum to single init and pairwise init, equivalent to one-hot in additional residue features

        single_init = single_init + single_molecule_embed
        pairwise_init = pairwise_init + pairwise_molecule_embed

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
    pae: Float['b pae n n'] | Float['b pae m m'] |  None
    pde: Float['b pde n n'] | Float['b pde m m']
    plddt: Float['b plddt n'] | Float['b plddt m']
    resolved: Float['b 2 n'] | Float['b 2 m']

class ConfidenceHead(Module):
    """ Algorithm 31 """

    @typecheck
    def __init__(
        self,
        *,
        dim_single_inputs,
        atom_resolution = False,  # @amorehead discovers that the public api has per-atom resolution confidences. improvise a solution
        dim_atom = 128,
        atompair_dist_bins: List[float],
        dim_single = 384,
        dim_pairwise = 128,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        pairformer_depth = 4,
        pairformer_kwargs: dict = dict()
    ):
        super().__init__()

        atompair_dist_bins = Tensor(atompair_dist_bins)

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

        # atom resolution
        # for now, just embed per atom distances, sum to atom features, project to pairwise dimension

        self.atom_resolution = atom_resolution

        if atom_resolution:
            self.atom_feats_to_single = LinearNoBias(dim_atom, dim_single)
            self.atom_feats_to_pairwise = LinearNoBiasThenOuterSum(dim_atom, dim_pairwise)

        # tensor typing

        self.da = dim_atom

    @typecheck
    def forward(
        self,
        *,
        single_inputs_repr: Float['b n dsi'],
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        pred_atom_pos: Float['b n 3'] | Float['b m 3'],
        molecule_atom_indices: Int['b n'] | None = None,
        molecule_atom_lens: Int['b n'] | None = None,
        mask: Bool['b n'] | None = None,
        atom_feats: Float['b m {self.da}'] | None = None,
        return_pae_logits = True

    ) -> ConfidenceHeadLogits:

        pairwise_repr = pairwise_repr + self.single_inputs_to_pairwise(single_inputs_repr)

        # pluck out the representative atoms for non-atomic resolution confidence head

        is_atom_seq = pred_atom_pos.shape[-2] > single_inputs_repr.shape[-2]

        # handle atom resolution vs not

        if self.atom_resolution:
            assert exists(atom_feats), 'atom_feats must be passed in if atom_resolution is turned on for ConfidenceHead'
            assert is_atom_seq, '`pred_atom_pos` must be passed in with atomic length'
            assert exists(molecule_atom_lens)

        if is_atom_seq:
            assert exists(molecule_atom_indices), 'molecule_atom_indices must be passed into ConfidenceHead if pred_atom_pos is atomic length'
            pred_molecule_pos = einx.get_at('b [m] c, b n -> b n c', pred_atom_pos, molecule_atom_indices)
        else:
            pred_molecule_pos = pred_atom_pos

        # interatomic distances - embed and add to pairwise

        intermolecule_dist = torch.cdist(pred_molecule_pos, pred_molecule_pos, p = 2)

        dist_from_dist_bins = einx.subtract('b m dist, dist_bins -> b m dist dist_bins', intermolecule_dist, self.atompair_dist_bins).abs()
        dist_bin_indices = dist_from_dist_bins.argmin(dim = -1)
        pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

        # pairformer stack

        single_repr, pairwise_repr = self.pairformer_stack(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask
        )

        # handle maybe atom level resolution

        if self.atom_resolution:
            single_repr = repeat_consecutive_with_lens(single_repr, molecule_atom_lens)

            pairwise_repr = repeat_consecutive_with_lens(pairwise_repr, molecule_atom_lens)

            molecule_atom_lens = repeat(molecule_atom_lens, 'b ... -> (b r) ...', r = pairwise_repr.shape[1])
            pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')
            pairwise_repr = repeat_consecutive_with_lens(pairwise_repr, molecule_atom_lens)
            pairwise_repr = unpack_one(pairwise_repr)

            interatomic_dist = torch.cdist(pred_atom_pos, pred_atom_pos, p = 2)

            dist_from_dist_bins = einx.subtract('b m dist, dist_bins -> b m dist dist_bins', interatomic_dist, self.atompair_dist_bins).abs()
            dist_bin_indices = dist_from_dist_bins.argmin(dim = -1)
            pairwise_repr = pairwise_repr + self.dist_bin_pairwise_embed(dist_bin_indices)

            single_repr = single_repr + self.atom_feats_to_single(atom_feats)
            pairwise_repr = pairwise_repr + self.atom_feats_to_pairwise(atom_feats)

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

# more confidence / clash calculations

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
        self.eps = eps
        self.register_buffer('pae_breaks', pae_breaks)
        self.register_buffer('pde_breaks', pde_breaks)

    @typecheck
    def _calculate_bin_centers(
        self,
        breaks: Float[' breaks'],
    ) -> Float[' breaks+1']:
        """
        Args:
            breaks: [num_bins -1] bin edges

        Returns:
            bin_centers: [num_bins] bin centers
        """

        step = breaks[1] - breaks[0]

        bin_centers = breaks + step / 2
        last_bin_center = breaks[-1:] + step

        bin_centers = torch.concat(
            [bin_centers, last_bin_center]
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
        plddt = self.compute_plddt(confidence_head_logits.plddt)

        # Section 5.9.1 equation 17
        ptm = self.compute_ptm(confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=False)

        iptm = None

        if multimer_mode:
            # Section 5.9.2 equation 18
            iptm = self.compute_ptm(confidence_head_logits.pae, asym_id, has_frame, ptm_residue_weight, interface=True)

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
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        residue_weights: Float['b n'] | None = None,
        interface: bool = False,
        compute_chain_wise_iptm: bool = False,
    ):
        if not exists(residue_weights):
            residue_weights = torch.ones_like(has_frame)

        residue_weights = residue_weights * has_frame

        num_batch = logits.shape[0]
        num_res = logits.shape[-1]
        logits = rearrange(logits, 'b c i j -> b i j c')

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
                enumerated_unique_chain = enumerate(unique_chains[b])

                for (i, chain_i), (j, chain_j) in product(enumerated_unique_chain, enumerated_unique_chain):
                    if chain_i == chain_j:
                        continue

                    mask_i = (asym_id[b] == chain_i)
                    mask_j = (asym_id[b] == chain_j)
                    pair_mask = einx.multiply('i, j -> i j', mask_i, mask_j)

                    pair_residue_weights = pair_mask * (
                        residue_weights[b, None, :] * residue_weights[b, :, None])

                    if pair_residue_weights.sum() == 0:
                        # chain i or chain j doesnot have any valid frame
                        continue

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
                pair_mask *= einx.not_equal('... i, ... j -> ... i j', asym_id, asym_id)

            predicted_tm_term *= pair_mask

            pair_residue_weights = pair_mask * einx.multiply('... i, ... j -> ... i j', residue_weights, residue_weights)

            normed_residue_mask = pair_residue_weights / (self.eps + torch.sum(
                pair_residue_weights, dim=-1, keepdims=True))

            per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
            weighted_argmax = (residue_weights * per_alignment).argmax(dim=-1)
            return per_alignment[torch.arange(num_batch) , weighted_argmax]

    @typecheck
    def compute_pde(
        self,
        logits: Float['b pde n n'],
        tok_repr_atm_mask: Bool[' b n'],
    )-> Float[' b n n']:

        logits = rearrange(logits, 'b pde i j -> b i j pde')
        bin_centers = self._calculate_bin_centers(self.pde_breaks)
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
        len_unique_chains = len(unique_chains)

        return_has_clash = False
        for i, j in product(range(len_unique_chains), range(len_unique_chains)):
            if j < (i + 1):
                continue

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
                return_has_clash = True
                break
    
        return torch.tensor(return_has_clash, dtype=torch.bool, device=atom_pos.device)
                
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

        if exists(atom_mask):
            valid_indices = valid_indices * atom_mask
        
        has_clash = [self.compute_has_clash(*compute_clash_args) for compute_clash_args in zip(atom_pos, asym_id, indices, valid_indices)]
        return torch.stack(has_clash)

class ComputeRankingScore(Module):

    def __init__(
        self,
        eps = 1e-8,
        score_iptm_weight = 0.8,
        score_ptm_weight = 0.2,
        score_disorder_weight = 0.5
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
        plddt: Float['b m'],
        atom_mask: Bool['b m'],
        atom_is_molecule_types: Bool['b m 5'],
    )-> Float[' b']:
        
        is_protein_mask = atom_is_molecule_types[..., IS_PROTEIN_INDEX]
        mask = atom_mask * is_protein_mask

        atom_rasa = 1 - plddt

        disorder = ( (atom_rasa > 0.581) * mask ).sum(dim=-1) / ( self.eps + mask.sum(dim=1)) 
        return disorder

    @typecheck
    def compute_full_complex_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        molecule_atom_lens: Int['b n'],
        atom_pos: Float['b m 3'],
        atom_mask: Bool['b m'],
        is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'],
        return_confidence_score: bool = False
    ) -> Float[' b'] | Tuple[Float[' b'], Tuple[ConfidenceScore, Bool[' b']]]:

        # Section 5.9.3.1
        
        device = atom_pos.device
        batch_size, seq_len= asym_id.shape

        indices = torch.arange(seq_len, device = device)

        indices = repeat(indices, 'n -> b n', b = batch_size)
        valid_indices = torch.ones_like(indices).bool()

        # valid_indices at padding position has value False
        indices = repeat_consecutive_with_lens(indices, molecule_atom_lens)
        valid_indices = repeat_consecutive_with_lens(valid_indices, molecule_atom_lens)

        # broadcast is_molecule_types to atom
        atom_is_molecule_types = einx.get_at('b [n] is_type, b m -> b m is_type', is_molecule_types, indices) * valid_indices[..., None]

        confidence_score = self.compute_confidence_score(
            confidence_head_logits, asym_id, has_frame, multimer_mode=True
        )
        has_clash = self.compute_clash(
            atom_pos, atom_mask, molecule_atom_lens, asym_id, 
        )

        disorder = self.compute_disorder(confidence_score.plddt, atom_mask, atom_is_molecule_types)

        # Section 5.9.3 equation 19

        weighted_score = (
            confidence_score.iptm * self.score_iptm_weight +
            confidence_score.ptm * self.score_ptm_weight +
            disorder * self.score_disorder_weight
            - 100 * has_clash
        )

        if not return_confidence_score:
            return weighted_score

        return weighted_score, (confidence_score, has_clash)

    @typecheck
    def compute_single_chain_metric(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
    ) -> Float[' b']:

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
        asym_id: Int['b n'],
        has_frame: Bool['b n'],
        interface_chains: List,
    ) -> Float[' b']:

        batch = asym_id.shape[0]

        # Section 5.9.3.3

        # interface_chains: List[chain_id_tuple]
        # chain_id_tuple: 
        #  - correspond to the asym_id of one or two chain
        #  - compute R(C) for one chain
        #  - compute 1/2 [R(A) + R(b)] for two chain

        chain_wise_iptm, chain_wise_iptm_mask, unique_chains = self.compute_confidence_score.compute_ptm(
            confidence_head_logits.pae, asym_id, has_frame, compute_chain_wise_iptm=True
        )

        # Section 5.9.3 equation 20
        interface_metric = torch.zeros(batch).type_as(chain_wise_iptm)

        # R(c) = mean(Mij) restricted to i = c or j = c
        masked_chain_wise_iptm = chain_wise_iptm * chain_wise_iptm_mask
        iptm_sum = masked_chain_wise_iptm + rearrange(masked_chain_wise_iptm, 'b i j -> b j i')
        iptm_count = chain_wise_iptm_mask.int() + rearrange(chain_wise_iptm_mask.int(), 'b i j -> b j i')

        for b, chains in enumerate(interface_chains):
            for chain in chains:
                idx = unique_chains[b].index(chain)
                interface_metric[b] += iptm_sum[b, idx].sum() / iptm_count[b, idx].sum().clamp(min=1)
            interface_metric[b] /= len(chains)
        return interface_metric
            
    @typecheck
    def compute_modified_residue_score(
        self,
        confidence_head_logits: ConfidenceHeadLogits,
        atom_mask: Bool['b m'],
        atom_is_modified_residue: Int['b m'],
    ) -> Float[' b']:

        # Section 5.9.3.4

        plddt = self.compute_confidence_score.compute_plddt(confidence_head_logits.plddt)

        mask = atom_is_modified_residue * atom_mask
        plddt_mean =  (plddt * mask).sum(dim=-1) / ( self.eps +  mask.sum(dim=-1)) 

        return plddt_mean

# model selection

@typecheck
def get_cid_molecule_type(
    cid: int,
    asym_id: Int[' n'],
    is_molecule_types: Bool[f'n {IS_MOLECULE_TYPES}'],
    return_one_hot: bool = False,
) -> int | Bool[f' {IS_MOLECULE_TYPES}']:
    """
    
    get the molecule type for where asym_id == cid
    """

    cid_is_molecule_types = is_molecule_types[asym_id == cid]
    molecule_type, rest_molecule_type = cid_is_molecule_types[0], cid_is_molecule_types[1:]

    valid = einx.equal('b i, i -> b i', rest_molecule_type, molecule_type).all()

    assert valid, f"Ambiguous molecule types for chain {cid}"

    if not return_one_hot:
        molecule_type = molecule_type.int().argmax().item()

    return molecule_type

class ComputeModelSelectionScore(Module):

    @typecheck
    def __init__(
        self,
        eps: float = 1e-8,
        dist_breaks: Float[' dist_break'] = torch.linspace(2.3125,21.6875,63,),
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0,
        contact_mask_threshold: float = 8.0
    ):

        super().__init__()
        self.compute_confidence_score = ComputeConfidenceScore(eps=eps)
        self.eps = eps
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff
        self.contact_mask_threshold = contact_mask_threshold

        self.register_buffer('dist_breaks', dist_breaks)

    @typecheck
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

        pde = self.compute_confidence_score.compute_pde(pde_logits, tok_repr_atm_mask)

        dist_logits = rearrange(dist_logits, 'b dist i j -> b i j dist')
        dist_probs = F.softmax(dist_logits, dim=-1)

        contact_mask = dist_breaks < self.contact_mask_threshold
        contact_mask = F.pad(contact_mask, (0, 1), value = True)

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

    @typecheck
    def compute_lddt(
        self,
        pred_coords: Float['b m 3'],
        true_coords: Float['b m 3'],
        is_dna: Bool['b m'],
        is_rna: Bool['b m'],
        pairwise_mask: Bool['b m m'],
        coords_mask: Bool['b m'] | None = None,
    ) -> Float[' b']:
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

    @typecheck
    def compute_chain_pair_lddt(
        self,
        asym_mask_a: Bool['b m'] | Bool [' m'],
        asym_mask_b: Bool['b m'] | Bool [' m'],
        pred_coords: Float['b m 3'] | Float['m 3'],
        true_coords: Float['b m 3'] | Float['m 3'], 
        is_molecule_types: Bool[f'b m {IS_MOLECULE_TYPES}'] | Bool[f'm {IS_MOLECULE_TYPES}'],
        coords_mask: Bool['b m'] | Bool [' m'] | None = None,
    ) -> Float[' b']:
        """
        
        plddt between atoms maked by asym_mask_a and asym_mask_b
        """

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

    @typecheck
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
    
    @typecheck
    def compute_weighted_lddt(
        self,
        # atom level input
        pred_coords: Float['b m 3'],
        true_coords: Float['b m 3'],
        atom_mask: Bool['b m'] | None,
        # token level input
        asym_id: Int['b n'],
        is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'],
        molecule_atom_lens: Int['b n'],
        # additional input
        chains_list: List[Tuple[int, int] | Tuple[int]],
        is_fine_tuning: bool = False,
    ):

        device = pred_coords.device
        batch_size = pred_coords.shape[0]

        # broadcast asym_id and is_molecule_types to atom level
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

# main class

class LossBreakdown(NamedTuple):
    total_loss: Float['']
    total_diffusion: Float['']
    distogram: Float['']
    pae: Float['']
    pde: Float['']
    plddt: Float['']
    resolved: Float['']
    confidence: Float['']
    diffusion_mse: Float['']
    diffusion_bond: Float['']
    diffusion_smooth_lddt: Float['']

class Alphafold3(Module):
    """ Algorithm 1 """

    @save_args_and_kwargs
    @typecheck
    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_template_feats,
        dim_template_model = 64,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair_inputs = 5,
        dim_atompair = 16,
        dim_input_embedder_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        dim_token = 768,
        dim_additional_token_feats = 2,                 # in paper, they include two meta information per token (f_profile, f_deletion_mean)
        num_molecule_types: int = NUM_MOLECULE_IDS,     # restype in additional residue information, apparently 32. will do 33 to account for metal ions
        num_atom_embeds: int | None = None,
        num_atompair_embeds: int | None = None,
        num_molecule_mods: int | None = None,
        distance_bins: List[float] = torch.linspace(3, 20, 38).float().tolist(),
        ignore_index = -1,
        num_dist_bins: int | None = None,
        num_plddt_bins = 50,
        num_pde_bins = 64,
        num_pae_bins = 64,
        sigma_data = 16,
        num_rollout_steps = 20,
        diffusion_num_augmentations = 4,
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
            layerscale_output = True,
        ),
        msa_module_kwargs: dict = dict(
            depth = 4,
            dim_msa = 64,
            dim_msa_input = None,
            outer_product_mean_dim_hidden = 32,
            msa_pwa_dropout_row_prob = 0.15,
            msa_pwa_heads = 8,
            msa_pwa_dim_head = 32,
            pairwise_block_kwargs = dict(),
            layerscale_output = True,
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
            atom_decoder_heads = 4,
            serial = True # believe they have an error on Algorithm 23. lacking a residual - default to serial architecture until further news
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
        ),
        augment_kwargs: dict = dict(),
        stochastic_frame_average = False,
        confidence_head_atom_resolution = False
    ):
        super().__init__()

        # store atom and atompair input dimensions for shape validation

        self.dim_atom_inputs = dim_atom_inputs
        self.dim_atompair_inputs = dim_atompair_inputs

        # optional atom and atom bond embeddings

        num_atom_embeds = default(num_atom_embeds, 0)
        num_atompair_embeds = default(num_atompair_embeds, 0)

        has_atom_embeds = num_atom_embeds > 0
        has_atompair_embeds = num_atompair_embeds > 0

        if has_atom_embeds:
            self.atom_embeds = nn.Embedding(num_atom_embeds, dim_atom)

        if has_atompair_embeds:
            self.atompair_embeds = nn.Embedding(num_atompair_embeds, dim_atompair)

        self.has_atom_embeds = has_atom_embeds
        self.has_atompair_embeds = has_atompair_embeds

        # residue or nucleotide modifications

        num_molecule_mods = default(num_molecule_mods, 0)
        has_molecule_mod_embeds = num_molecule_mods > 0

        if has_molecule_mod_embeds:
            self.molecule_mod_embeds = nn.Embedding(num_molecule_mods, dim_single)

        self.has_molecule_mod_embeds = has_molecule_mod_embeds

        # atoms per window

        self.atoms_per_window = atoms_per_window

        # augmentation

        self.num_augmentations = diffusion_num_augmentations
        self.augmenter = CentreRandomAugmentation(**augment_kwargs)

        # stochastic frame averaging
        # https://arxiv.org/abs/2305.05577

        self.stochastic_frame_average = stochastic_frame_average

        if stochastic_frame_average:
            self.frame_average = FrameAverage(
                dim = 3,
                stochastic = True,
                return_stochastic_as_augmented_pos = True
            )

        # input feature embedder

        self.input_embedder = InputFeatureEmbedder(
            num_molecule_types = num_molecule_types,
            dim_atom_inputs = dim_atom_inputs,
            dim_atompair_inputs = dim_atompair_inputs,
            atoms_per_window = atoms_per_window,
            dim_atom = dim_atom,
            dim_atompair = dim_atompair,
            dim_token = dim_input_embedder_token,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            dim_additional_token_feats = dim_additional_token_feats,
            **input_embedder_kwargs
        )

        # they concat some MSA related information per token (`f_profile`, `f_deletion_mean`)
        # line 2 of Algorithm 2
        # the `f_restypes` is handled elsewhere

        dim_single_inputs = dim_input_embedder_token + dim_additional_token_feats

        self.dim_additional_token_feats = dim_additional_token_feats

        # relative positional encoding
        # used by pairwise in main alphafold2 trunk
        # and also in the diffusion module separately from alphafold3

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            **relative_position_encoding_kwargs
        )

        # token bonds
        # Algorithm 1 - line 5

        self.token_bond_to_pairwise_feat = nn.Sequential(
            Rearrange('... -> ... 1'),
            LinearNoBias(1, dim_pairwise)
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

        distance_bins_tensor = Tensor(distance_bins)

        self.register_buffer('distance_bins', distance_bins_tensor)
        num_dist_bins = default(num_dist_bins, len(distance_bins_tensor))

        assert len(distance_bins_tensor) == num_dist_bins, '`distance_bins` must have a length equal to the `num_dist_bins` passed in'

        self.distogram_head = DistogramHead(
            dim_pairwise = dim_pairwise,
            num_dist_bins = num_dist_bins
        )

        self.num_rollout_steps = num_rollout_steps

        self.confidence_head = ConfidenceHead(
            dim_single_inputs = dim_single_inputs,
            atompair_dist_bins = distance_bins,
            dim_single = dim_single,
            dim_pairwise = dim_pairwise,
            num_plddt_bins = num_plddt_bins,
            num_pde_bins = num_pde_bins,
            num_pae_bins = num_pae_bins,
            atom_resolution = confidence_head_atom_resolution,
            **confidence_head_kwargs
        )

        # loss related

        self.ignore_index = ignore_index
        self.loss_distogram_weight = loss_distogram_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.loss_diffusion_weight = loss_diffusion_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # some shorthand for jaxtyping

        self.w = atoms_per_window
        self.dapi = self.dim_atompair_inputs
        self.dai = self.dim_atom_inputs

    @property
    def device(self):
        return self.zero.device

    @property
    def state_dict_with_init_args(self):
        return dict(
            version = self._version,
            init_args_and_kwargs = self._args_and_kwargs,
            state_dict = self.state_dict()
        )

    @typecheck
    def save(self, path: str | Path, overwrite = False):
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok = True, parents = True)

        package = dict(
            model = self.state_dict_with_init_args
        )

        torch.save(package, str(path))

    @typecheck
    def load(
        self,
        path: str | Path,
        strict = False,
        map_location = 'cpu'
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists() and path.is_file()

        package = torch.load(str(path), map_location = map_location)

        model_package = package['model']
        current_version = version('alphafold3_pytorch')

        if model_package['version'] != current_version:
            logger.warning(f'loading a saved model from version {model_package["version"]} but you are on version {current_version}')

        self.load_state_dict(model_package['state_dict'], strict = strict)

        return package.get('id', None)

    @staticmethod
    @typecheck
    def init_and_load(
        path: str | Path,
        map_location = 'cpu'
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.is_file()

        package = torch.load(str(path), map_location = map_location)

        model_package = package['model']

        args, kwargs = model_package['init_args_and_kwargs']
        alphafold3 = Alphafold3(*args, **kwargs)

        alphafold3.load(path)
        return alphafold3

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m {self.dai}'],
        atompair_inputs: Float['b m m {self.dapi}'] | Float['b nw {self.w} {self.w*2} {self.dapi}'],
        additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}'],
        is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'],
        molecule_atom_lens: Int['b n'],
        molecule_ids: Int['b n'],
        additional_token_feats: Float['b n {self.dim_additional_token_feats}'] | None = None,
        atom_ids: Int['b m'] | None = None,
        atompair_ids: Int['b m m'] | Int['b nw {self.w} {self.w*2}'] | None = None,
        is_molecule_mod: Bool['b n num_mods'] | None = None,
        atom_mask: Bool['b m'] | None = None,
        missing_atom_mask: Bool['b m'] | None = None,
        atom_parent_ids: Int['b m'] | None = None,
        token_bonds: Bool['b n n'] | None = None,
        msa: Float['b s n d'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        templates: Float['b t n n dt'] | None = None,
        template_mask: Bool['b t'] | None = None,
        num_recycling_steps: int = 1,
        diffusion_add_bond_loss: bool = False,
        diffusion_add_smooth_lddt_loss: bool = False,
        distogram_atom_indices: Int['b n'] | None = None,
        molecule_atom_indices: Int['b n'] | None = None, # the 'token centre atoms' mentioned in the paper, unsure where it is used in the architecture
        num_sample_steps: int | None = None,
        atom_pos: Float['b m 3'] | None = None,
        distance_labels: Int['b n n'] | Int['b m m'] | None = None,
        pae_labels: Int['b n n'] | Int['b m m'] | None = None,
        pde_labels: Int['b n n'] | Int['b m m'] | None = None,
        plddt_labels: Int['b n'] | Int['b m'] | None = None,
        resolved_labels: Int['b n'] | Int['b m'] | None = None,
        return_loss_breakdown = False,
        return_loss: bool = None,
        return_present_sampled_atoms: bool = False,
        return_confidence_head_logits: bool = False,
        num_rollout_steps: int | None = None,
        rollout_show_tqdm_pbar: bool = False
    ) -> (
        Float['b m 3'] |
        Float['l 3'] |
        Tuple[Float['b m 3'] | Float['l 3'], ConfidenceHeadLogits] |
        Float[''] |
        Tuple[Float[''], LossBreakdown]
    ):

        atom_seq_len = atom_inputs.shape[-2]

        # validate atom and atompair input dimensions

        assert atom_inputs.shape[-1] == self.dim_atom_inputs, f'expected {self.dim_atom_inputs} for atom_inputs feature dimension, but received {atom_inputs.shape[-1]}'
        assert atompair_inputs.shape[-1] == self.dim_atompair_inputs, f'expected {self.dim_atompair_inputs} for atompair_inputs feature dimension, but received {atompair_inputs.shape[-1]}'

        # soft validate

        valid_atom_len_mask = molecule_atom_lens >= 0

        molecule_atom_lens = molecule_atom_lens.masked_fill(~valid_atom_len_mask, 0)

        if exists(molecule_atom_indices):
            valid_molecule_atom_mask = molecule_atom_indices >= 0 & valid_atom_len_mask
            molecule_atom_indices = molecule_atom_indices.masked_fill(~valid_molecule_atom_mask, 0)
            assert (molecule_atom_indices < molecule_atom_lens)[valid_molecule_atom_mask].all(), 'molecule_atom_indices cannot have an index that exceeds the length of the atoms for that molecule as given by molecule_atom_lens'

        if exists(distogram_atom_indices):
            valid_distogram_mask = distogram_atom_indices >= 0 & valid_atom_len_mask
            distogram_atom_indices = distogram_atom_indices.masked_fill(~valid_distogram_mask, 0)
            assert (distogram_atom_indices < molecule_atom_lens)[valid_distogram_mask].all(), 'distogram_atom_indices cannot have an index that exceeds the length of the atoms for that molecule as given by molecule_atom_lens'

        assert exists(molecule_atom_lens) or exists(atom_mask)

        # if atompair inputs are not windowed, window it

        is_atompair_inputs_windowed = atompair_inputs.ndim == 5

        if not is_atompair_inputs_windowed:
            atompair_inputs = full_pairwise_repr_to_windowed(atompair_inputs, window_size = self.atoms_per_window)

        # handle atom mask

        total_atoms = molecule_atom_lens.sum(dim = -1)
        atom_mask = lens_to_mask(total_atoms, max_len = atom_seq_len)

        # handle offsets for molecule atom indices

        if exists(molecule_atom_indices):
            molecule_atom_indices = molecule_atom_indices + exclusive_cumsum(molecule_atom_lens)

        # get atom sequence length and molecule sequence length depending on whether using packed atomic seq

        seq_len = molecule_atom_lens.shape[-1]

        # embed inputs

        (
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats
        ) = self.input_embedder(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            atom_mask = atom_mask,
            additional_token_feats = additional_token_feats,
            molecule_atom_lens = molecule_atom_lens,
            molecule_ids = molecule_ids
        )

        # handle maybe atom and atompair embeddings

        assert not (exists(atom_ids) ^ self.has_atom_embeds), 'you either set `num_atom_embeds` and did not pass in `atom_ids` or vice versa'
        assert not (exists(atompair_ids) ^ self.has_atompair_embeds), 'you either set `num_atompair_embeds` and did not pass in `atompair_ids` or vice versa'

        if self.has_atom_embeds:
            atom_embeds = self.atom_embeds(atom_ids)
            atom_feats = atom_feats + atom_embeds

        if self.has_atompair_embeds:
            atompair_embeds = self.atompair_embeds(atompair_ids)

            if atompair_embeds.ndim == 4:
                atompair_embeds = full_pairwise_repr_to_windowed(atompair_embeds, window_size = self.atoms_per_window)

            atompair_feats = atompair_feats + atompair_embeds

        # handle maybe molecule modifications

        assert not (exists(is_molecule_mod) ^ self.has_molecule_mod_embeds), 'you either set `num_molecule_mods` and did not pass in `is_molecule_mod` or vice versa'

        if self.has_molecule_mod_embeds:
            single_init, seq_unpack_one = pack_one(single_init, '* ds')

            is_molecule_mod, _ = pack_one(is_molecule_mod, '* mods')

            if not is_molecule_mod.is_sparse:
                is_molecule_mod = is_molecule_mod.to_sparse()

            seq_indices, mod_id = is_molecule_mod.indices()
            scatter_values = self.molecule_mod_embeds(mod_id)

            seq_indices = repeat(seq_indices, 'n -> n ds', ds = single_init.shape[-1])
            single_init = single_init.scatter_add(0, seq_indices, scatter_values)

            single_init = seq_unpack_one(single_init)

        # relative positional encoding

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats = additional_molecule_feats
        )

        # only apply relative positional encodings to biomolecules that are chained
        # not to ligands + metal ions

        is_chained_biomol = is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim = -1) # first three types are chained biomolecules (protein, rna, dna)
        paired_is_chained_biomol = einx.logical_and('b i, b j -> b i j', is_chained_biomol, is_chained_biomol)

        relative_position_encoding = einx.where(
            'b i j, b i j d, -> b i j d',
            paired_is_chained_biomol, relative_position_encoding, 0.
        )

        # add relative positional encoding to pairwise init

        pairwise_init = pairwise_init + relative_position_encoding

        # token bond features

        if exists(token_bonds):
            # well do some precautionary standardization
            # (1) mask out diagonal - token to itself does not count as a bond
            # (2) symmetrize, in case it is not already symmetrical (could also throw an error)

            token_bonds = token_bonds | rearrange(token_bonds, 'b i j -> b j i')
            diagonal = torch.eye(seq_len, device = self.device, dtype = torch.bool)
            token_bonds = token_bonds.masked_fill(diagonal, False)
        else:
            seq_arange = torch.arange(seq_len, device = self.device)
            token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1

        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.float())

        pairwise_init = pairwise_init + token_bonds_feats

        # molecule mask and pairwise mask

        mask = molecule_atom_lens > 0
        pairwise_mask = einx.logical_and('b i, b j -> b i j', mask, mask)

        # prepare mask for msa module and template embedder
        # which is equivalent to the `is_protein` of the `is_molecular_types` input

        is_protein_mask = is_molecule_types[..., IS_PROTEIN_INDEX]

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

            if exists(templates):
                embedded_template = self.template_embedder(
                    templates = templates,
                    template_mask = template_mask,
                    pairwise_repr = pairwise,
                    mask = is_protein_mask
                )

                pairwise = embedded_template + pairwise

            # msa

            if exists(msa):
                embedded_msa = self.msa_module(
                    msa = msa,
                    single_repr = single,
                    pairwise_repr = pairwise,
                    mask = is_protein_mask,
                    msa_mask = msa_mask
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

        can_return_loss = atom_pos_given or has_labels

        # default whether to return loss by whether labels or atom positions are given

        return_loss = default(return_loss, can_return_loss)

        # if neither atom positions or any labels are passed in, sample a structure and return

        if not return_loss:
            sampled_atom_pos = self.edm.sample(
                num_sample_steps = num_sample_steps,
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_parent_ids = atom_parent_ids,
                atom_mask = atom_mask,
                mask = mask,
                single_trunk_repr = single,
                single_inputs_repr = single_inputs,
                pairwise_trunk = pairwise,
                pairwise_rel_pos_feats = relative_position_encoding,
                molecule_atom_lens = molecule_atom_lens
            )

            if exists(atom_mask):
                sampled_atom_pos = einx.where('b m, b m c, -> b m c', atom_mask, sampled_atom_pos, 0.)

            if return_confidence_head_logits:
                confidence_head_atom_pos_input = sampled_atom_pos.clone()

            if exists(missing_atom_mask) and return_present_sampled_atoms:
                sampled_atom_pos = sampled_atom_pos[~missing_atom_mask]

            if not return_confidence_head_logits:
                return sampled_atom_pos

            confidence_head_logits = self.confidence_head(
                single_repr = single.detach(),
                single_inputs_repr = single_inputs.detach(),
                pairwise_repr = pairwise.detach(),
                pred_atom_pos = confidence_head_atom_pos_input.detach(),
                molecule_atom_indices = molecule_atom_indices,
                molecule_atom_lens = molecule_atom_lens,
                atom_feats = atom_feats,
                mask = mask,
                return_pae_logits = True
            )

            return sampled_atom_pos, confidence_head_logits

        # if being forced to return loss, but do not have sufficient information to return losses, just return 0

        if return_loss and not can_return_loss:
            zero = self.zero.requires_grad_()

            if not return_loss_breakdown:
                return zero

            return zero, LossBreakdown(*((zero,) * 11))

        # losses default to 0

        distogram_loss = diffusion_loss = confidence_loss = pae_loss = pde_loss = plddt_loss = resolved_loss = self.zero

        # calculate distogram logits and losses

        ignore = self.ignore_index

        # distogram head

        if not exists(distance_labels) and atom_pos_given and exists(distogram_atom_indices):
            molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)
            molecule_dist = torch.cdist(molecule_pos, molecule_pos, p = 2)
            dist_from_dist_bins = einx.subtract('b m dist, dist_bins -> b m dist dist_bins', molecule_dist, self.distance_bins).abs()
            distance_labels = dist_from_dist_bins.argmin(dim = -1)

            # account for representative distogram atom missing from residue (-1 set on distogram_atom_indices field)

            valid_distogram_mask = einx.logical_and('b i, b j -> b i j', valid_distogram_mask, valid_distogram_mask)
            distance_labels.masked_fill_(~valid_distogram_mask, ignore)

        if exists(distance_labels):
            distance_labels = torch.where(pairwise_mask, distance_labels, ignore)
            distogram_logits = self.distogram_head(pairwise)
            distogram_loss = F.cross_entropy(distogram_logits, distance_labels, ignore_index = ignore)

        # otherwise, noise and make it learn to denoise

        calc_diffusion_loss = exists(atom_pos)

        if calc_diffusion_loss:

            num_augs = self.num_augmentations + int(self.stochastic_frame_average)

            # take care of augmentation
            # they did 48 during training, as the trunk did the heavy lifting

            if num_augs > 1:
                (
                    atom_pos,
                    atom_mask,
                    missing_atom_mask,
                    atom_feats,
                    atom_parent_ids,
                    atompair_feats,
                    mask,
                    pairwise_mask,
                    single,
                    single_inputs,
                    pairwise,
                    relative_position_encoding,
                    additional_molecule_feats,
                    is_molecule_types,
                    molecule_atom_indices,
                    molecule_atom_lens,
                    pae_labels,
                    pde_labels,
                    plddt_labels,
                    resolved_labels,

                ) = tuple(
                    maybe(repeat)(t, 'b ... -> (b a) ...', a = num_augs)
                    for t in (
                        atom_pos,
                        atom_mask,
                        missing_atom_mask,
                        atom_feats,
                        atom_parent_ids,
                        atompair_feats,
                        mask,
                        pairwise_mask,
                        single,
                        single_inputs,
                        pairwise,
                        relative_position_encoding,
                        additional_molecule_feats,
                        is_molecule_types,
                        molecule_atom_indices,
                        molecule_atom_lens,
                        pae_labels,
                        pde_labels,
                        plddt_labels,
                        resolved_labels
                    )
                )

                # handle stochastic frame averaging

                aug_atom_mask = atom_mask

                if self.stochastic_frame_average:
                    fa_atom_pos, atom_pos = atom_pos[:1], atom_pos[1:]
                    fa_atom_mask, aug_atom_mask = atom_mask[:1], atom_mask[1:]

                    fa_atom_pos = self.frame_average(
                        fa_atom_pos,
                        frame_average_mask = fa_atom_mask
                    )

                # normal random augmentations, 48 times in paper

                atom_pos = self.augmenter(atom_pos, mask = aug_atom_mask)

                # concat back the stochastic frame averaged position

                if self.stochastic_frame_average:
                    atom_pos = torch.cat((fa_atom_pos, atom_pos), dim = 0)

            diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm(
                atom_pos,
                additional_molecule_feats = additional_molecule_feats,
                is_molecule_types = is_molecule_types,
                add_smooth_lddt_loss = diffusion_add_smooth_lddt_loss,
                add_bond_loss = diffusion_add_bond_loss,
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_parent_ids = atom_parent_ids,
                missing_atom_mask = missing_atom_mask,
                atom_mask = atom_mask,
                mask = mask,
                single_trunk_repr = single,
                single_inputs_repr = single_inputs,
                pairwise_trunk = pairwise,
                pairwise_rel_pos_feats = relative_position_encoding,
                molecule_atom_lens = molecule_atom_lens,
                return_denoised_pos = True,
            )

        # confidence head

        should_call_confidence_head = any([*map(exists, confidence_head_labels)])
        return_pae_logits = exists(pae_labels)

        if calc_diffusion_loss and should_call_confidence_head and exists(molecule_atom_indices):

            # rollout

            num_rollout_steps = default(num_rollout_steps, self.num_rollout_steps)

            denoised_atom_pos = self.edm.sample(
                num_sample_steps = num_rollout_steps,
                atom_feats = atom_feats,
                atompair_feats = atompair_feats,
                atom_mask = atom_mask,
                mask = mask,
                single_trunk_repr = single,
                single_inputs_repr = single_inputs,
                pairwise_trunk = pairwise,
                pairwise_rel_pos_feats = relative_position_encoding,
                molecule_atom_lens = molecule_atom_lens,
                use_tqdm_pbar = rollout_show_tqdm_pbar,
                tqdm_pbar_title = 'training rollout'
            )

            ch_logits = self.confidence_head(
                single_repr = single.detach(),
                single_inputs_repr = single_inputs.detach(),
                pairwise_repr = pairwise.detach(),
                pred_atom_pos = denoised_atom_pos.detach(),
                molecule_atom_indices = molecule_atom_indices,
                molecule_atom_lens = molecule_atom_lens,
                mask = mask,
                atom_feats = atom_feats,
                return_pae_logits = return_pae_logits
            )

            # determine which mask to use for labels depending on atom resolution or not for confidence head

            label_mask = mask

            if self.confidence_head.atom_resolution:
                label_mask = atom_mask

            label_pairwise_mask = einx.logical_and('... i, ... j -> ... i j', label_mask, label_mask)

            # cross entropy losses

            assert len(set([t.shape[-1] for t in compact(pde_labels, plddt_labels, resolved_labels)])) == 1
            assert pde_labels.shape[-1] == ch_logits.pde.shape[-1]

            if exists(pae_labels):
                pae_labels = torch.where(label_pairwise_mask, pae_labels, ignore)
                pae_loss = F.cross_entropy(ch_logits.pae, pae_labels, ignore_index = ignore)

            if exists(pde_labels):
                pde_labels = torch.where(label_pairwise_mask, pde_labels, ignore)
                pde_loss = F.cross_entropy(ch_logits.pde, pde_labels, ignore_index = ignore)

            if exists(plddt_labels):
                plddt_labels = torch.where(label_mask, plddt_labels, ignore)
                plddt_loss = F.cross_entropy(ch_logits.plddt, plddt_labels, ignore_index = ignore)

            if exists(resolved_labels):
                resolved_labels = torch.where(label_mask, resolved_labels, ignore)
                resolved_loss = F.cross_entropy(ch_logits.resolved, resolved_labels, ignore_index = ignore)

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
            total_loss = loss,
            total_diffusion = diffusion_loss,
            pae = pae_loss,
            pde = pde_loss,
            plddt = plddt_loss,
            resolved = resolved_loss,
            distogram = distogram_loss,
            confidence = confidence_loss,
            **diffusion_loss_breakdown._asdict()
        )

        return loss, loss_breakdown

# an alphafold3 that can download pretrained weights from huggingface

class Alphafold3WithHubMixin(Alphafold3, PyTorchModelHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: Dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = 'cpu',
        strict: bool = False,
        model_filename: str = 'alphafold3.bin',
        **model_kwargs,
    ):
        model_file = Path(model_id) / model_filename

        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id = model_id,
                filename = model_filename,
                revision = revision,
                cache_dir = cache_dir,
                force_download = force_download,
                proxies = proxies,
                resume_download = resume_download,
                token = token,
                local_files_only = local_files_only,
            )

        model = cls.init_and_load(
            model_file,
            strict = strict,
            map_location = map_location
        )

        return model
