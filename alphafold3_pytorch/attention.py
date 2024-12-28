from __future__ import annotations
from beartype.typing import NamedTuple, Tuple
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module

import einx
from einops import einsum, repeat, rearrange, pack, unpack
from einops.layers.torch import Rearrange

from alphafold3_pytorch.tensor_typing import (
    Shaped,
    Float,
    Int,
    Bool,
    typecheck
)

# alias

LinearNoBias = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def softclamp(t, value):
    return (t / value).tanh() * value

@typecheck
def pad_at_dim(
    t,
    pad: Tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

@typecheck
def slice_at_dim(
    t: Tensor,
    dim_slice: slice,
    *,
    dim: int
) -> Tensor:
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

@typecheck
def pad_to_length(
    t: Tensor,
    length: int,
    *,
    dim: int = -1,
    value = 0
) -> Tensor:
    padding = max(length - t.shape[dim], 0)

    if padding == 0:
        return t

    return pad_at_dim(t, (0, padding), dim = dim, value = value)

@typecheck
def pad_or_slice_to(
    t: Tensor,
    length: int,
    *,
    dim: int,
    pad_value = 0
) -> Tensor:
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_to_length(t, length, dim = dim, value = pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim = dim)

    return t

@typecheck
def pad_to_multiple(
    t: Tensor,
    multiple: int,
    *,
    dim = -1,
    value = 0.
) -> Tensor:
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim = dim, value = value)

@typecheck
def concat_previous_window(
    t: Tensor,
    *,
    dim_seq: int,
    dim_window: int
) -> Tensor:
    t = pad_at_dim(t, (1, 0), dim = dim_seq, value = 0.)

    t = torch.cat((
        slice_at_dim(t, slice(None, -1), dim = dim_seq),
        slice_at_dim(t, slice(1, None), dim = dim_seq),
    ), dim = dim_window)

    return t

# for changing full attention bias matrix to a local windowed one for atom attention

@typecheck
def full_pairwise_repr_to_windowed(
    pairwise_repr: Shaped['... m m dp'],
    window_size: int
) -> Shaped['... n w (w*2) dp']:

    seq_len, device = pairwise_repr.shape[-2], pairwise_repr.device

    padding_needed = (window_size - (seq_len % window_size)) % window_size
    pairwise_repr = F.pad(pairwise_repr, (0, 0, 0, padding_needed, 0, padding_needed), value = 0.)
    pairwise_repr = rearrange(pairwise_repr, '... (i w1) (j w2) d -> ... i j w1 w2 d', w1 = window_size, w2 = window_size)
    pairwise_repr = concat_previous_window(pairwise_repr, dim_seq = -4, dim_window = -2)

    # get the diagonal

    n = torch.arange(pairwise_repr.shape[-4], device = device)

    # pairwise_repr = einx.get_at('... [i j] w1 w2 d, n, n -> ... n w1 w2 d', pairwise_repr, n, n)

    pairwise_repr = pairwise_repr[..., n, n, :, :, :]

    return pairwise_repr

@typecheck
def full_attn_bias_to_windowed(
    attn_bias: Shaped['... m m'],
    window_size: int
) -> Shaped['... n w (w*2)']:

    attn_bias = rearrange(attn_bias, '... -> ... 1')
    attn_bias = full_pairwise_repr_to_windowed(attn_bias, window_size = window_size)
    return rearrange(attn_bias, '... 1 -> ...')

# multi-head attention

class Attention(Module):
    @typecheck
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        gate_output = True,
        query_bias = True,
        window_size = None,
        num_memory_kv: int = 0,
        enable_attn_softclamp = False,
        attn_softclamp_value = 50.,
        softmax_full_precision = False,
        accept_value_residual = False
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        m - memory key / value seq
        """

        dim_inner = dim_head * heads

        self.attend = Attend(
            dropout = dropout,
            window_size = window_size,
            enable_attn_softclamp = enable_attn_softclamp,
            attn_softclamp_value = attn_softclamp_value,
            softmax_full_precision = softmax_full_precision
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = nn.Linear(dim, dim_inner, bias = query_bias)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.memory_kv = None

        if num_memory_kv > 0:
            self.memory_kv = nn.Parameter(torch.zeros(2, heads, num_memory_kv, dim_head))
            nn.init.normal_(self.memory_kv, std = 0.02)

        # gating of value
        # allows attention to attend to nothing

        self.to_gates = None

        if gate_output:
            self.to_gates = nn.Sequential(LinearNoBias(dim, dim_inner), nn.Sigmoid())

        # learned value residual mixing
        # even greater improvements on top of value residual learning, discovered by open source community

        self.accept_value_residual = accept_value_residual

        if accept_value_residual:
            self.to_value_residual_mix = nn.Sequential(
                LinearNoBias(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

    @typecheck
    def forward(
        self,
        seq: Float['b i d'],
        mask: Bool['b n']| None = None,
        context: Float['b j d'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None,
        attn_bias: Float['... i j'] | Float['... nw w (w*2)'] | None = None,
        return_values: bool = False,
        value_residual: Float['b h j dh'] | None = None,

    ) -> (
        Float['b i d'] |
        tuple[Float['b i d'], Float['b h j dh']]
    ):

        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim = -1)

        # split heads

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        # handle value residual

        orig_v = v

        assert not (self.accept_value_residual ^ exists(value_residual))

        if exists(value_residual):
            mix = self.to_value_residual_mix(seq)
            v = v.lerp(value_residual, mix)

        # attention

        out = self.attend(
            q, k, v,
            attn_bias = attn_bias,
            mask = mask,
            windowed_mask = windowed_mask,
            memory_kv = self.memory_kv
        )

        # merge heads

        out = self.merge_heads(out)

        # gate output

        if exists(self.to_gates):
            gates = self.to_gates(seq)
            out = out * gates

        # combine heads

        out = self.to_out(out)

        # maybe return values

        if not return_values:
            return out

        return out, orig_v

# the main attention function

class Attend(Module):
    def __init__(
        self,
        dropout = 0.,
        window_size = None,
        scale: float | None = None,
        enable_attn_softclamp = False,
        attn_softclamp_value = 50.,
        softmax_full_precision = False
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        w - local attention windows
        """

        self.scale = scale
        self.dropout = dropout

        self.is_local_attn = exists(window_size)
        self.window_size = window_size

        self.attn_dropout = nn.Dropout(dropout)

        # softclamp attention logits
        # being adopted by a number of recent llms (gemma, grok)

        self.enable_attn_softclamp = enable_attn_softclamp
        self.attn_softclamp_value = attn_softclamp_value

        # whether to use full precision for softmax
        self.softmax_full_precision = softmax_full_precision

    @typecheck
    def local_attn(
        self,
        q: Float['b h n d'],
        k: Float['b h n d'],
        v: Float['b h n d'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None,
        attn_bias: Float['... n n'] | Float['... nw w (w*2)'] | None = None,
        memory_kv: Float['2 h m d'] | None = None
    ) -> Float['b h n d']:
        """
        simple local attention with a radius of 1 window size
        """

        window_size, batch, seq_len, device = self.window_size, q.shape[0], q.shape[-2], q.device

        # constitute mask if not given

        if not exists(mask):
            mask = torch.ones((batch, seq_len), device = device, dtype = torch.bool)

        # pad to multiple of window size if needed

        padding_needed = (window_size - (seq_len % window_size)) % window_size

        if padding_needed > 0:
            q, k, v = tuple(pad_at_dim(t, (0, padding_needed), value = 0., dim = -2) for t in (q, k, v))
            mask = F.pad(mask, (0, padding_needed), value = False)

        # break into windows

        q, k, v = tuple(rearrange(t, 'b h (n w) d -> b h n w d', w = window_size) for t in (q, k, v))
        mask = rearrange(mask, 'b (n w) -> b n w', w = window_size)

        # just do radius of 1 for now
        # perhaps not even necessary, and could try shifted windows (a la Swin)

        k, v = tuple(pad_at_dim(t, (1, 0), dim = -3) for t in (k, v))
        mask = pad_at_dim(mask, (1, 0), dim = -2, value = False)

        k, v = tuple(torch.cat((t[..., :-1, :, :], t[..., 1:, :, :]), dim = -2) for t in (k, v))
        mask = torch.cat((mask[..., :-1, :], mask[..., 1:, :]), dim = -1)

        # handle attention bias (inefficiently)

        is_full_attn_bias = attn_bias.shape[-1] == attn_bias.shape[-2]

        if exists(attn_bias) and is_full_attn_bias:
            attn_bias = full_attn_bias_to_windowed(attn_bias, window_size = window_size)

        # carry out attention as usual

        scale = q.shape[-1] ** -0.5

        q = q * scale

        # append memory key / values for local attention windows

        if exists(memory_kv):
            batch, seq, num_mem_kv = k.shape[0], k.shape[2], memory_kv.shape[-2]

            mk, mv = memory_kv
            mk, mv = tuple(repeat(t, 'h m d -> b h n m d', b = batch, n = seq) for t in (mk, mv))
            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(attn_bias):
                attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value = 0.)

            if exists(windowed_mask):
                windowed_mask = pad_at_dim(windowed_mask, (num_mem_kv, 0), value = True)

            if exists(mask):
                mask = pad_at_dim(mask, (num_mem_kv, 0), value = True)

        # similarity

        sim = einsum(q, k, '... i d, ... j d -> ... i j')

        if exists(attn_bias):
            if attn_bias.ndim == 4:
                attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')

            assert attn_bias.ndim == sim.ndim
            sim = sim + attn_bias

        # maybe softclamp

        if self.enable_attn_softclamp:
            sim = softclamp(sim, self.attn_softclamp_value)

        # windowed masking - for masking out atoms not belonging to the same molecule / polypeptide / nucleic acid in sequence-local attention

        if exists(windowed_mask):
            sim = einx.where(
                'b n i j, b h n i j, -> b h n i j',
                windowed_mask, sim, max_neg_value(sim)
            )

        # mask out buckets of padding

        sim = einx.where(
            'b n j, b h n i j, -> b h n i j',
            mask, sim, max_neg_value(sim)
        )

        # local attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum(attn, v, "... i j, ... j d -> ... i d")

        # un-window the output

        out = rearrange(out, "b h n w d -> b h (n w) d")

        # excise the padding for windowing

        out = out[..., :seq_len, :]

        return out

    @typecheck
    def forward(
        self,
        q: Float['b h i d'],
        k: Float['b h j d'],
        v: Float['b h j d'],
        mask: Bool['b j'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None,
        attn_bias: Float['... i j'] | Float['... nw w (w*2)'] | None = None,
        memory_kv: Float['2 h m d'] | None = None
    ) -> Float['b h i d']:

        dtype = q.dtype

        is_windowed_attn_bias = None

        if exists(attn_bias):
            is_windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        # local windowed attention
        # todo (handle attn bias efficiently)

        if self.is_local_attn:
            return self.local_attn(q, k, v, mask = mask, windowed_mask = windowed_mask, attn_bias = attn_bias, memory_kv = memory_kv)

        assert not exists(is_windowed_attn_bias) or not is_windowed_attn_bias

        # append memory key / values

        if exists(memory_kv):
            batch, num_mem_kv = q.shape[0], memory_kv.shape[-2]

            mk, mv = memory_kv
            mk, mv = tuple(repeat(t, 'h m d -> b h m d', b = batch) for t in (mk, mv))
            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(attn_bias):
                attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value = 0.)

            if exists(mask):
                mask = pad_at_dim(mask, (num_mem_kv, 0), value = True)

        # default attention

        scale = default(self.scale, q.shape[-1] ** -0.5)

        q = q * scale

        # similarity

        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        # attn bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # maybe softclamp

        if self.enable_attn_softclamp:
            sim = softclamp(sim, self.attn_softclamp_value)

        # masking

        if exists(mask):
            sim = einx.where(
                'b j, b h i j, -> b h i j',
                mask, sim, max_neg_value(sim)
            )

        # attention cast float32 - in case there are instabilities with float16

        softmax_kwargs = dict()

        if self.softmax_full_precision:
            softmax_kwargs.update(dtype = torch.float32)

        # attention

        attn = sim.softmax(dim = -1, **softmax_kwargs)
        attn = attn.to(dtype)

        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        return out
