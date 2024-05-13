from __future__ import annotations
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

from einops import einsum, repeat, rearrange
from einops.layers.torch import Rearrange

from alphafold3_pytorch.typing import Float, Int, Bool, typecheck

# constants

class Config(NamedTuple):
    enable_flash: bool
    enable_math: bool
    enable_mem_efficient: bool

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# multi-head attention

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        gate_output = False,
        flash = True,
        efficient_attn_config: Config = Config(True, True, True)
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        i - source sequence
        j - context sequence
        """

        dim_inner = dim_head * heads

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            attn_config = efficient_attn_config
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        # used in alphafold2

        self.to_gates = None

        if gate_output:
            gate_linear = nn.Linear(dim, dim_inner)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, 1.)

            self.to_gates = gate_linear

    @typecheck
    def forward(
        self,
        seq: Float['b i d'],
        mask: Bool['b n']| None = None,
        attn_bias: Float['b h i j'] | None = None,
        context: Float['b j d'] | None = None
    ) -> Float['b i d']:

        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        out = self.attend(
            q, k, v,
            attn_bias = attn_bias,
            mask = mask
        )

        out = self.merge_heads(out)

        if exists(self.to_gates):
            gates = self.to_gates(seq)
            out = out * gates.sigmoid()

        return self.to_out(out)

# attending, both vanilla as well as in-built flash attention

class Attend(Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale: float | None = None,
        attn_config: Config = Config(True, True, True)
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        d - dimension
        n, i, j - sequence (base sequence length, source, target)
        """

        self.scale = scale
        self.dropout = dropout

        self.flash = flash
        self.attn_config = attn_config
        self.attn_dropout = nn.Dropout(dropout)

    @typecheck
    def flash_attn(
        self,
        q: Float['b h i d'],
        k: Float['b h j d'],
        v: Float['b h j d'],
        mask: Bool['b j'] | None = None
    ) -> Float['b h i d']:

        _, heads, seq_len, _ = q.shape

        attn_mask = None

        if exists(mask):
            mask = repeat(mask, 'b j -> b h i j', h = heads, i = seq_len)

        with torch.backends.cuda.sdp_kernel(**self.attn_config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = attn_mask,
                scale = self.scale,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    @typecheck
    def forward(
        self,
        q: Float['b h i d'],
        k: Float['b h j d'],
        v: Float['b h j d'],
        attn_bias: Float['b h i j'] | None = None,
        mask: Bool['b j'] | None = None
    ) -> Float['b h i d']:

        can_use_flash = self.flash and not exists(attn_bias), 'flash attention does not support attention bias with gradients'

        if can_use_flash:
            return self.flash_attn(q, k, v, mask = mask)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        q = q * scale

        # similarity

        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        # attn bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        if exists(mask):
            mask_value = max_neg_value(sim)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        return out
