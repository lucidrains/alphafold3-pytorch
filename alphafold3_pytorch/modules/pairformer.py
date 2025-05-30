from __future__ import annotations

from functools import partial, wraps

import torch
from torch import nn

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
)

from beartype.typing import (
    Literal,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float,
    Bool,
    typecheck,
    checkpoint,
)

from alphafold3_pytorch.attention import (
    Attention,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed
)


from alphafold3_pytorch.utils.helpers import *
from alphafold3_pytorch.modules.basic_models import *


from alphafold3_pytorch.utils.model_utils import (

    pack_one
)
from alphafold3_pytorch.utils.utils import not_exists


# personal libraries

from hyper_connections.hyper_connections_with_multi_input_streams import HyperConnections

# einstein notation related

import einx
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange
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

        self.left_right_proj = nn.Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim = -1)
        )

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
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, '... -> ... 1')

        left, right = self.left_right_proj(x).chunk(2, dim = -1)

        if exists(mask):
            left = left * mask
            right = right * mask

        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)

        out_gate = self.out_gate(x).sigmoid()

        return self.to_out(out) * out_gate

# there are two types of attention in this paper, triangle and attention-pair-bias
# they differ by how the attention bias is computed
# triangle is axial attention w/ itself projected for bias

class AttentionPairBias(Module):
    """An Attention module with pair bias computation."""

    def __init__(self, *, heads, dim_pairwise, window_size=None, num_memory_kv=0, **attn_kwargs):
        super().__init__()

        self.window_size = window_size

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            num_memory_kv = num_memory_kv,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
        self.to_attn_bias = nn.Sequential(LinearNoBias(dim_pairwise, heads), Rearrange("b ... h -> b h ..."))

    @typecheck
    def forward(
        self,
        single_repr: Float["b n ds"],  # type: ignore
        *,
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        attn_bias: Float["b n n"] | Float["b nw w (w*2)"] | None = None,  # type: ignore
        return_values: bool = False,
        value_residual: Float['b _ _ _'] | None = None,
        **kwargs,
    ) -> (
        Float['b n ds'] |
        tuple[Float['b n ds'], Float['b _ _ _']]
    ):  # type: ignore

        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param attn_bias: The attention bias tensor.
        :return: The output tensor.
        """
        b, dp = pairwise_repr.shape[0], pairwise_repr.shape[-1]
        dtype, device = pairwise_repr.dtype, pairwise_repr.device
        w, has_window_size = self.window_size, exists(self.window_size)

        # take care of windowing logic
        # for sequence-local atom transformer

        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size=w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size=w)
        else:
            assert (
                not windowed_pairwise
            ), "Cannot pass in windowed pairwise representation if no `window_size` given to `AttentionPairBias`."
            assert (
                not_exists(windowed_attn_bias) or not windowed_attn_bias
            ), "Cannot pass in windowed attention bias if no `window_size` is set for `AttentionPairBias`."

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")
        else:
            attn_bias = 0.0

        if pairwise_repr.numel() > MAX_CONCURRENT_TENSOR_ELEMENTS:
            # create a stub tensor and normalize it to maintain gradients to `to_attn_bias_norm`
            stub_pairwise_repr = torch.zeros((b, dp), dtype=dtype, device=device)
            stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0

            # adjust `attn_bias_norm` dimensions to match `pairwise_repr`
            attn_bias_norm = pairwise_repr + (
                stub_attn_bias_norm[:, None, None, None, :]
                if windowed_pairwise
                else stub_attn_bias_norm[:, None, None, :]
            )

            # apply bias transformation
            attn_bias = self.to_attn_bias(attn_bias_norm) + attn_bias
        else:
            attn_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr)) + attn_bias

        # attention

        out, values = self.attn(
            single_repr,
            attn_bias = attn_bias,
            value_residual = value_residual,
            return_values = True,
            **kwargs
        )

        # whether to return values for value residual learning

        if not return_values:
            return out

        return out, values

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
        return_values = False,
        **kwargs
    ) -> (
        Float['b n n d'] |
        tuple[Float['b n n d'], Tensor]
    ):

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')

        out, values = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            return_values = True,
            **kwargs
        )

        out = unpack_one(out)

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        out = self.dropout(out)

        if not return_values:
            return out

        return out, values

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
        accept_value_residual = False
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
            dim_head = tri_attn_dim_head,
            accept_value_residual = accept_value_residual
        )

        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = 'outgoing', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = 'incoming', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_attn_starting = pre_ln(TriangleAttention(node_type = 'starting', dropout = dropout_row_prob, dropout_type = 'row', **tri_attn_kwargs))
        self.tri_attn_ending = pre_ln(TriangleAttention(node_type = 'ending', dropout = dropout_col_prob, dropout_type = 'col', **tri_attn_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        *,
        mask: Bool['b n'] | None = None,
        value_residuals: tuple[Tensor, Tensor] | None = None,
        return_values = False,
    ):
        pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask = mask) + pairwise_repr

        attn_start_value_residual, attn_end_value_residual = default(value_residuals, (None, None))

        attn_start_out, attn_start_values = self.tri_attn_starting(pairwise_repr, mask = mask, value_residual = attn_start_value_residual, return_values = True)
        pairwise_repr = attn_start_out + pairwise_repr

        attn_end_out, attn_end_values = self.tri_attn_ending(pairwise_repr, mask = mask, value_residual = attn_end_value_residual, return_values = True)
        pairwise_repr = attn_end_out + pairwise_repr

        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr

        if not return_values:
            return pairwise_repr

        return pairwise_repr, (attn_start_values, attn_end_values)

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
        
        dtype = msa.dtype

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)

        # maybe masked mean for outer product

        if exists(msa_mask):
            a = einx.multiply('b s i d, b s -> b s i d', a, msa_mask.type(dtype))
            b = einx.multiply('b s j e, b s -> b s j e', b, msa_mask.type(dtype))

            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')

            num_msa = reduce(msa_mask.type(dtype), '... s -> ...', 'sum')

            outer_product_mean = einx.divide('b i j d e, b', outer_product, num_msa.clamp(min = self.eps))
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
            outer_product_mean = outer_product / num_msa

        # flatten

        outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # masking for pairwise repr

        if exists(mask):
            mask = to_pairwise_mask(mask)
            outer_product_mean = einx.multiply(
                'b i j d, b i j', outer_product_mean, mask.type(dtype)
            )

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
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
        recurrent_depth = 1, # effective depth will be depth * recurrent_depth
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        num_register_tokens = 0,
        checkpoint = False,
        add_value_residual = False,
        num_residual_streams = 1,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict()
    ):
        super().__init__()

        # residual / hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        layers = ModuleList([])

        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob,
            **pair_bias_attn_kwargs
        )

        for i in range(depth):

            is_first = i == 0
            accept_value_residual = add_value_residual and not is_first

            single_pre_ln = partial(PreLayerNorm, dim = dim_single)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                accept_value_residual = accept_value_residual,
                **pairwise_block_kwargs
            )

            pair_bias_attn = AttentionPairBias(accept_value_residual = accept_value_residual, **pair_bias_attn_kwargs)
            single_transition = Transition(dim = dim_single)

            layers.append(ModuleList([
                init_hyper_conn(dim = dim_pairwise, branch = pairwise_block),
                init_hyper_conn(dim = dim_single, additional_input_paths = [('pairwise_repr', dim_pairwise)], branch = single_pre_ln(pair_bias_attn)),
                init_hyper_conn(dim = dim_single, branch = single_pre_ln(single_transition)),
            ]))

        self.layers = layers

        self.add_value_residual = add_value_residual

        # checkpointing

        self.checkpoint = checkpoint

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
    def to_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        single_repr = self.expand_streams(single_repr)
        pairwise_repr = self.expand_streams(pairwise_repr)

        for _ in range(self.recurrent_depth):

            value_residual = None
            pairwise_value_residuals = None

            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in self.layers:

                pairwise_repr, pairwise_attn_values = pairwise_block(pairwise_repr, mask = mask, value_residuals = pairwise_value_residuals, return_values = True)

                single_repr, attn_values = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask, return_values = True, value_residual = value_residual)

                if self.add_value_residual:
                    value_residual = default(value_residual, attn_values)
                    pairwise_value_residuals = default(pairwise_value_residuals, pairwise_attn_values)

                single_repr = single_transition(single_repr)

        single_repr = self.reduce_streams(single_repr)
        pairwise_repr = self.reduce_streams(pairwise_repr)

        return single_repr, pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None

    ) -> Tuple[Float['b n ds'], Float['b n n dp']]:

        def pairwise_block_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals = inputs
                pairwise_repr, pairwise_attn_values = layer(pairwise_repr, mask = mask, value_residuals = maybe_pairwise_value_residuals, return_values = True)

                if self.add_value_residual:
                    maybe_pairwise_value_residuals = default(maybe_pairwise_value_residuals, pairwise_attn_values)

                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        def pair_bias_attn_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals  = inputs
                single_repr, attn_values = layer(single_repr, pairwise_repr = pairwise_repr, mask = mask, return_values = True, value_residual = maybe_value_residual)

                if self.add_value_residual:
                    maybe_value_residual = default(maybe_value_residual, attn_values)

                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        def single_transition_wrapper(layer):
            @wraps(layer)
            def inner(inputs, *args, **kwargs):
                single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals = inputs
                single_repr = layer(single_repr)
                return single_repr, pairwise_repr, mask, maybe_value_residual, maybe_pairwise_value_residuals
            return inner

        wrapped_layers = []

        for (
            pairwise_block,
            pair_bias_attn,
            single_transition
        ) in self.layers:

            wrapped_layers.append(pairwise_block_wrapper(pairwise_block))
            wrapped_layers.append(pair_bias_attn_wrapper(pair_bias_attn))
            wrapped_layers.append(single_transition_wrapper(single_transition))

        single_repr = self.expand_streams(single_repr)
        pairwise_repr = self.expand_streams(pairwise_repr)

        for _ in range(self.recurrent_depth):
            inputs = (single_repr, pairwise_repr, mask, None, None)

            for layer in wrapped_layers:
                inputs = checkpoint(layer, inputs)

            single_repr, pairwise_repr, *_ = inputs

        single_repr = self.reduce_streams(single_repr)
        pairwise_repr = self.reduce_streams(pairwise_repr)

        return single_repr, pairwise_repr

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

        # maybe checkpoint

        if should_checkpoint(self, (single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        # main transformer block layers

        single_repr, pairwise_repr = to_layers_fn(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask
        )

        # splice out registers

        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]

        return single_repr, pairwise_repr