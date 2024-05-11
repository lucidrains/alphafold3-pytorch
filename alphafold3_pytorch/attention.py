from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

import einx
from einops import einsum, repeat
from einops.layers.torch import Rearrange

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
        flash = True
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

        self.attend = Attend(flash = flash, dropout = dropout)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        seq,
        mask = None,
        context = None
    ):
        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = self.merge_heads(out)
        return self.to_out(out)

# attending, both vanilla as well as in-built flash attention

class Attend(Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None,
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

    def flash_attn(self, q, k, v, mask = None):
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

    def forward(self, q, k, v, mask = None):

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        q = q * scale

        # similarity

        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        # masking

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        return out
