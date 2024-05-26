# from __future__ import annotations

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn import Module

# import einx
# from einops import einsum, repeat, rearrange
# from einops.layers.torch import Rearrange

# from alphafold3_pytorch.models.components.constants import AttentionConfig
# from alphafold3_pytorch.utils.model_utils import max_neg_value, pad_at_dim
# from alphafold3_pytorch.utils.utils import default, exists
# from alphafold3_pytorch.utils.typing import (
#     Float,
#     Bool,
#     typecheck
# )


# # multi-head attention

# class Attention(Module):
#     """Attention model."""

#     @typecheck
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_head = 64,
#         heads = 8,
#         dropout = 0.,
#         gate_output = True,
#         query_bias = True,
#         flash = True,
#         window_size = None,
#         efficient_attn_config: AttentionConfig = AttentionConfig(True, True, True)
#     ):
#         super().__init__()
#         """
#         ein notation:

#         b - batch
#         h - heads
#         n - sequence
#         d - dimension
#         e - dimension (pairwise rep)
#         i - source sequence
#         j - context sequence
#         """

#         dim_inner = dim_head * heads

#         self.attend = Attend(
#             flash = flash,
#             dropout = dropout,
#             window_size = window_size,
#             attn_config = efficient_attn_config
#         )

#         self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
#         self.merge_heads = Rearrange('b h n d -> b n (h d)')

#         self.to_q = nn.Linear(dim, dim_inner, bias = query_bias)
#         self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
#         self.to_out = nn.Linear(dim_inner, dim, bias = False)

#         # gating of value
#         # allows attention to attend to nothing

#         self.to_gates = None

#         if gate_output:
#             gate_linear = nn.Linear(dim, dim_inner)
#             nn.init.zeros_(gate_linear.weight)
#             nn.init.constant_(gate_linear.bias, 1.)

#             self.to_gates = gate_linear

#     @typecheck
#     def forward(
#         self,
#         seq: Float["b i d"], # type: ignore
#         mask: Bool["b n"]| None = None, # type: ignore
#         context: Float["b j d"] | None = None, # type: ignore
#         attn_bias: Float["... i j"] | None = None # type: ignore

#     ) -> Float["b i d"]: # type: ignore
#         """
#         Run multi-head attention on a sequence.

#         :param seq: The input sequence.
#         :param mask: The mask to apply to the sequence.
#         :param context: The context sequence to reference.
#         :param attn_bias: The attention bias to apply.
#         :return: The output sequence.
#         """

#         q = self.to_q(seq)

#         context_seq = default(context, seq)
#         k, v = self.to_kv(context_seq).chunk(2, dim = -1)

#         q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

#         # attention

#         out = self.attend(
#             q, k, v,
#             attn_bias = attn_bias,
#             mask = mask
#         )

#         # merge heads

#         out = self.merge_heads(out)

#         # gate output

#         if exists(self.to_gates):
#             gates = self.to_gates(seq)
#             out = out * gates.sigmoid()

#         # combine heads

#         return self.to_out(out)

# # attending, both vanilla as well as in-built flash attention

# class Attend(Module):
#     """Attention module."""
#     def __init__(
#         self,
#         dropout = 0.,
#         flash = False,
#         window_size = None,
#         scale: float | None = None,
#         attn_config: AttentionConfig = AttentionConfig(True, True, True)
#     ):
#         super().__init__()
#         """
#         ein notation:

#         b - batch
#         h - heads
#         n - sequence
#         d - dimension
#         e - dimension (pairwise rep)
#         i - source sequence
#         j - context sequence
#         w - local attention windows
#         """

#         self.scale = scale
#         self.dropout = dropout

#         self.is_local_attn = exists(window_size)
#         self.window_size = window_size

#         self.flash = flash
#         self.attn_config = attn_config
#         self.attn_dropout = nn.Dropout(dropout)

#     @typecheck
#     def flash_attn(
#         self,
#         q: Float["b h i d"], # type: ignore
#         k: Float["b h j d"], # type: ignore
#         v: Float["b h j d"], # type: ignore
#         mask: Bool["b j"] | None = None # type: ignore
#     ) -> Float["b h i d"]: # type: ignore
#         """
#         Run flash attention.

#         :param q: The query tensor.
#         :param k: The key tensor.
#         :param v: The value tensor.
#         :param mask: The mask to apply to the sequence.
#         :return: The output tensor.
#         """

#         _, heads, seq_len, _ = q.shape

#         attn_mask = None

#         if exists(mask):
#             mask = repeat(mask, 'b j -> b h i j', h = heads, i = seq_len)

#         with torch.backends.cuda.sdp_kernel(**self.attn_config._asdict()):
#             out = F.scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask = attn_mask,
#                 scale = self.scale,
#                 dropout_p = self.dropout if self.training else 0.
#             )

#         return out

#     @typecheck
#     def local_attn(
#         self,
#         q: Float["b h n d"], # type: ignore
#         k: Float["b h n d"], # type: ignore
#         v: Float["b h n d"], # type: ignore
#         mask: Bool["b n"] | None = None, # type: ignore
#         attn_bias: Float["... n n"] | None = None # type: ignore
#     ) -> Float["b h n d"]: # type: ignore
#         """
#         Run simple local attention with a radius of 1 window size.

#         :param q: The query tensor.
#         :param k: The key tensor.
#         :param v: The value tensor.
#         :param mask: The mask to apply to the sequence.
#         :param attn_bias: The attention bias to apply.
#         :return: The output tensor.
#         """

#         window_size, batch, seq_len, device = self.window_size, q.shape[0], q.shape[-2], q.device

#         # constitute mask if not given

#         if not exists(mask):
#             mask = torch.ones((batch, seq_len), device = device, dtype = torch.bool)

#         # pad to multiple of window size if needed

#         padding_needed = (window_size - (seq_len % window_size)) % window_size

#         if padding_needed > 0:
#             q, k, v = tuple(pad_at_dim(t, (0, padding_needed), value = 0., dim = -2) for t in (q, k, v))
#             mask = F.pad(mask, (0, padding_needed), value = False)

#         # break into windows

#         q, k, v = tuple(rearrange(t, 'b h (n w) d -> b h n w d', w = window_size) for t in (q, k, v))
#         mask = rearrange(mask, 'b (n w) -> b n w', w = window_size)

#         # just do radius of 1 for now
#         # perhaps not even necessary, and could try shifted windows (a la Swin)

#         k, v = tuple(pad_at_dim(t, (1, 1), dim = -2) for t in (k, v))
#         mask = F.pad(mask, (1, 1), value = False)

#         k, v = tuple(torch.cat((t[..., :-2, :], t[..., 1:-1, :], t[..., 2:, :]), dim = -2) for t in (k, v))
#         mask = torch.cat((mask[..., :-2], mask[..., 1:-1], mask[..., 2:]), dim = -1)

#         # handle attention bias (inefficiently)

#         if exists(attn_bias):
#             attn_bias = F.pad(attn_bias, (0, padding_needed, 0, padding_needed), value = 0.)
#             attn_bias = rearrange(attn_bias, '... (i w1) (j w2) -> ... i j w1 w2', w1 = window_size, w2 = window_size)
#             attn_bias = pad_at_dim(attn_bias, (1, 1), dim = -3, value = 0.)

#             attn_bias = torch.cat((
#                 attn_bias[..., :-2, :, :],
#                 attn_bias[..., 1:-1, :, :],
#                 attn_bias[..., 2:, :, :]
#             ), dim = -1)

#             # get the diagonal

#             n = torch.arange(attn_bias.shape[-3], device = device)

#             attn_bias = einx.get_at(
#                 '... [i j] w1 w2, n, n -> ... n w1 w2',
#                 attn_bias, n, n
#             )

#         # carry out attention as usual

#         scale = q.shape[-1] ** -0.5

#         q = q * scale

#         sim = einsum(q, k, "... i d, ... j d -> ... i j")

#         if exists(attn_bias):
#             if attn_bias.ndim == 4:
#                 attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')

#             assert attn_bias.ndim == sim.ndim
#             sim = sim + attn_bias

#         sim = einx.where(
#             'b n j, b h n i j, -> b h n i j',
#             mask, sim, max_neg_value(sim)
#         )

#         attn = sim.softmax(dim = -1)

#         out = einsum(attn, v, "... i j, ... j d -> ... i d")

#         # un-window the output

#         out = rearrange(out, "b h n w d -> b h (n w) d")

#         # excise the padding for windowing

#         out = out[..., :seq_len, :]

#         return out

#     @typecheck
#     def forward(
#         self,
#         q: Float["b h i d"], # type: ignore
#         k: Float["b h j d"], # type: ignore
#         v: Float["b h j d"], # type: ignore
#         mask: Bool["b j"] | None = None, # type: ignore
#         attn_bias: Float["... i j"] | None = None, # type: ignore
#     ) -> Float["b h i d"]: # type: ignore
#         """
#         Run attention.

#         :param q: The query tensor.
#         :param k: The key tensor.
#         :param v: The value tensor.
#         :param mask: The mask to apply to the sequence.
#         :param attn_bias: The attention bias to apply.
#         :return: The output tensor.
#         """

#         # local windowed attention
#         # todo (handle attn bias efficiently)

#         if self.is_local_attn:
#             return self.local_attn(q, k, v, mask = mask, attn_bias = attn_bias)

#         # forward to using flash attention if applicable

#         can_use_flash = self.flash and not exists(attn_bias), 'flash attention does not support attention bias with gradients'

#         if can_use_flash:
#             return self.flash_attn(q, k, v, mask = mask)

#         # default attention

#         scale = default(self.scale, q.shape[-1] ** -0.5)

#         q = q * scale

#         # similarity

#         sim = einsum(q, k, "b h i d, b h j d -> b h i j")

#         # attn bias

#         if exists(attn_bias):
#             sim = sim + attn_bias

#         # masking

#         if exists(mask):
#             sim = einx.where(
#                 'b j, b h i j, -> b h i j',
#                 mask, sim, max_neg_value(sim)
#             )

#         # attention

#         attn = sim.softmax(dim = -1)
#         attn = self.attn_dropout(attn)

#         # aggregate values

#         out = einsum(attn, v, "b h i j, b h j d -> b h i d")

#         return out
