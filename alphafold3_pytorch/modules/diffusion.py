from __future__ import annotations

from math import pi, sqrt
from functools import partial, wraps

import torch
from torch import nn
from torch import Tensor
from torch.amp import autocast
import torch.nn.functional as F

from torch.nn import (
    Module,
    ModuleList,
)

from beartype.typing import (
    Dict,
    List,
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

from alphafold3_pytorch.attention import (
    pad_or_slice_to,
    concat_previous_window,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.inputs import (
    IS_PROTEIN_INDEX,
    IS_DNA_INDEX,
    IS_RNA_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
)


from alphafold3_pytorch.utils.model_utils import (
    ExpressCoordinatesInFrame,
    calculate_weighted_rigid_align_weights,
    pack_one
)

from alphafold3_pytorch.utils.helpers import *
from alphafold3_pytorch.modules.basic_models import *
from alphafold3_pytorch.modules.pairformer import (
    AttentionPairBias,
)

# personal libraries


from taylor_series_linear_attention import TaylorSeriesLinearAttn

from colt5_attention import ConditionalRoutedAttention

from hyper_connections.hyper_connections_with_multi_input_streams import HyperConnections

# other external libs

from tqdm import tqdm
from loguru import logger
# einstein notation related

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack


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
    ) -> (
        Float['b n d'] |
        tuple[Float['b n d'], Float['b _ _ _']]
    ):
        x = self.adaptive_norm(x, cond = cond)

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        gamma = self.to_adaln_zero_gamma(cond)
        out = out * gamma

        if tuple_output:
            out = (out, *rest)

        return out
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
        use_linear_attn = False,
        checkpoint = False,
        add_value_residual = False,
        num_residual_streams = 1,
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

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        layers = ModuleList([])

        for i in range(depth):
            is_first = i == 0

            linear_attn = None

            if use_linear_attn:
                linear_attn = TaylorSeriesLinearAttn(
                    dim = dim,
                    prenorm = True,
                    gate_value_heads = True,
                    remove_even_power_dups = True,
                    **linear_attn_kwargs
                )

                linear_attn = init_hyper_conn(dim = dim, branch = linear_attn)

            colt5_attn = None

            if use_colt5_attn:
                colt5_attn = ConditionalRoutedAttention(
                    dim = dim,
                    has_light_attn = False,
                    **colt5_attn_kwargs
                )

                colt5_attn = init_hyper_conn(dim = dim, branch = colt5_attn)

            accept_value_residual = add_value_residual and not is_first

            pair_bias_attn = AttentionPairBias(
                dim = dim,
                dim_pairwise = dim_pairwise,
                heads = heads,
                window_size = attn_window_size,
                num_memory_kv = attn_num_memory_kv,
                accept_value_residual = accept_value_residual,
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
                init_hyper_conn(dim = dim, branch = conditionable_pair_bias),
                init_hyper_conn(dim = dim, branch = conditionable_transition)
            ]))

        self.checkpoint = checkpoint

        self.layers = layers

        self.add_value_residual = add_value_residual

        self.has_registers = num_register_tokens > 0
        self.num_registers = num_register_tokens

        if self.has_registers:
            assert not exists(attn_window_size), 'register tokens disabled for windowed attention'

            self.registers = nn.Parameter(torch.zeros(num_register_tokens, dim))

    @typecheck
    def to_checkpointed_serial_layers(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):

        wrapped_layers = []

        def efficient_attn_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual = inputs
                noised_repr = fn(noised_repr, mask = mask)
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual
            return inner

        def attn_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual = inputs
                noised_repr, attn_values = fn(noised_repr, cond = single_repr, pairwise_repr = pairwise_repr, mask = mask, windowed_mask = windowed_mask, value_residual = maybe_value_residual, return_values = True)

                if self.add_value_residual:
                    maybe_value_residual = default(maybe_value_residual, attn_values)

                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual
            return inner

        def transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual = inputs
                noised_repr = fn(noised_repr, cond = single_repr)
                return noised_repr, single_repr, pairwise_repr, mask, windowed_mask, maybe_value_residual
            return inner

        # wrap layers

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                wrapped_layers.append(efficient_attn_wrapper(linear_attn))

            if exists(colt5_attn):
                wrapped_layers.append(efficient_attn_wrapper(colt5_attn))

            wrapped_layers.append(attn_wrapper(attn))
            wrapped_layers.append(transition_wrapper(transition))

        # forward

        noised_repr = self.expand_streams(noised_repr)

        inputs = (noised_repr, single_repr, pairwise_repr, mask, windowed_mask, None)

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        noised_repr, *_ = inputs

        noised_repr = self.reduce_streams(noised_repr)

        return noised_repr

    @typecheck
    def to_serial_layers(
        self,
        noised_repr: Float['b n d'],
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'] | Float['b nw w (w*2) dp'],
        mask: Bool['b n'] | None = None,
        windowed_mask: Bool['b nw w (w*2)'] | None = None
    ):

        value_residual = None

        noised_repr = self.expand_streams(noised_repr)

        for linear_attn, colt5_attn, attn, transition in self.layers:

            if exists(linear_attn):
                noised_repr = linear_attn(noised_repr, mask = mask)

            if exists(colt5_attn):
                noised_repr = colt5_attn(noised_repr, mask = mask)

            noised_repr, attn_values = attn(
                noised_repr,
                cond = single_repr,
                pairwise_repr = pairwise_repr,
                mask = mask,
                windowed_mask = windowed_mask,
                return_values = True,
                value_residual = value_residual
            )

            if self.add_value_residual:
                value_residual = default(value_residual, attn_values)

            noised_repr = transition(
                noised_repr,
                cond = single_repr
            )

        noised_repr = self.reduce_streams(noised_repr)

        return noised_repr

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

        if should_checkpoint(self, (noised_repr, single_repr, pairwise_repr)):
            to_layers_fn = self.to_checkpointed_serial_layers
        else:
            to_layers_fn = self.to_serial_layers

        noised_repr = to_layers_fn(
            noised_repr,
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask,
            windowed_mask = windowed_mask,
        )

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
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
        use_linear_attn = False,
        checkpoint = False,
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
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            checkpoint = checkpoint,
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
            checkpoint = checkpoint,
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
            use_linear_attn = use_linear_attn,
            linear_attn_kwargs = linear_attn_kwargs,
            checkpoint = checkpoint,
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

        single_repr_cond = batch_repeat_interleave(single_repr_cond, molecule_atom_lens)
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

        indices = batch_repeat_interleave(indices, molecule_atom_lens)
        indices = pad_or_slice_to(indices, atom_seq_len, dim = -1)
        indices = pad_and_window(indices, w)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, 'b n w -> b n w 1', w = w)
        col_indices = rearrange(col_indices, 'b n w -> b n 1 w', w = w)

        col_indices = concat_previous_window(col_indices, dim_seq = 1, dim_window = -1)
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        # pairwise_repr_cond = einx.get_at('b [i j] dap, b nw w1 w2, b nw w1 w2 -> b nw w1 w2 dap', pairwise_repr_cond, row_indices, col_indices)

        row_indices, unpack_one = pack_one(row_indices, 'b *')
        col_indices, _ = pack_one(col_indices, 'b *')

        rowcol_indices = col_indices + row_indices * pairwise_repr_cond.shape[2]
        rowcol_indices = repeat(rowcol_indices, 'b rc -> b rc dap', dap = pairwise_repr_cond.shape[-1])
        pairwise_repr_cond, _ = pack_one(pairwise_repr_cond, 'b * dap')

        pairwise_repr_cond = pairwise_repr_cond.gather(1, rowcol_indices)
        pairwise_repr_cond = unpack_one(pairwise_repr_cond, 'b * dap')
        
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

        atom_decoder_input = batch_repeat_interleave(atom_decoder_input, molecule_atom_lens)
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
        step_scale = 1.5,
        augment_during_sampling = True,
        lddt_mask_kwargs: dict = dict(),
        smooth_lddt_loss_kwargs: dict = dict(),
        weighted_rigid_align_kwargs: dict = dict(),
        multi_chain_permutation_alignment_kwargs: dict = dict(),
        centre_random_augmentation_kwargs: dict = dict(),
        karras_formulation = True,  # use the original EDM formulation from Karras et al. Table 1 in https://arxiv.org/abs/2206.00364 - differences are that the noise and sampling schedules are scaled by sigma data, as well as loss weight adds the sigma data instead of multiply in denominator
        verbose = False,
    ):
        super().__init__()

        self.verbose = verbose
        self.net = net

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.step_scale = step_scale

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # centre random augmenter

        self.augment_during_sampling = augment_during_sampling
        self.centre_random_augmenter = CentreRandomAugmentation(**centre_random_augmentation_kwargs)

        # weighted rigid align

        self.weighted_rigid_align = WeightedRigidAlign(**weighted_rigid_align_kwargs)

        # multi-chain permutation alignment

        self.multi_chain_permutation_alignment = MultiChainPermutationAlignment(
            **multi_chain_permutation_alignment_kwargs,
            weighted_rigid_align=self.weighted_rigid_align,
        )

        # smooth lddt loss

        self.smooth_lddt_loss = SmoothLDDTLoss(**smooth_lddt_loss_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to use original karras formulation or not

        self.karras_formulation = karras_formulation

    @property
    def device(self):
        return next(self.net.parameters()).device

    @property
    def dtype(self):
        return next(self.net.parameters()).dtype

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
        batch, dtype, device = (
            noised_atom_pos.shape[0],
            noised_atom_pos.dtype,
            noised_atom_pos.device,
        )

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, dtype=dtype, device=device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_atom_pos,
            times = sigma,
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

        steps = torch.arange(num_sample_steps, device=self.device, dtype=self.dtype)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.

        return sigmas * self.sigma_data

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Bool['b m'] | None = None,
        num_sample_steps = None,
        clamp = False,
        use_tqdm_pbar = True,
        tqdm_pbar_title = 'sampling time step',
        return_all_timesteps = False,
        **network_condition_kwargs
    ) -> Float['b m 3'] | Float['ts b m 3']:

        dtype = self.dtype

        step_scale, num_sample_steps = self.step_scale, default(num_sample_steps, self.num_sample_steps)

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

        atom_pos = init_sigma * torch.randn(shape, dtype = dtype, device = self.device)

        # gradually denoise

        maybe_tqdm_wrapper = tqdm if use_tqdm_pbar else identity

        maybe_augment_fn = self.centre_random_augmenter if self.augment_during_sampling else identity

        all_atom_pos = [atom_pos]

        for sigma, sigma_next, gamma in maybe_tqdm_wrapper(sigmas_and_gammas, desc = tqdm_pbar_title):
            sigma, sigma_next, gamma = tuple(t.item() for t in (sigma, sigma_next, gamma))

            atom_pos = maybe_augment_fn(atom_pos.float()).type(dtype)

            eps = self.S_noise * torch.randn(
                shape, dtype = dtype, device = self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            atom_pos_hat = atom_pos + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(atom_pos_hat, sigma_hat, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
            denoised_over_sigma = (atom_pos_hat - model_output) / sigma_hat

            atom_pos_next = atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma * step_scale

            # second order correction, if not the last timestep

            if self.karras_formulation and sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(atom_pos_next, sigma_next, clamp = clamp, network_condition_kwargs = network_condition_kwargs)
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = atom_pos_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma) * step_scale

            atom_pos = atom_pos_next

            all_atom_pos.append(atom_pos)

        # if returning atom positions across all timesteps for visualization
        # then stack the `all_atom_pos`

        if return_all_timesteps:
            atom_pos = torch.stack(all_atom_pos)

        if clamp:
            atom_pos = atom_pos.clamp(-1., 1.)

        return atom_pos

    # training

    def karras_loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def loss_weight(self, sigma):
        """ for some reason, in paper they add instead of multiply as in original paper """
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma + self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp() * self.sigma_data

    @typecheck
    def forward(
        self,
        atom_pos_ground_truth: Float['b m 3'],
        atom_mask: Bool['b m'],
        atom_feats: Float['b m da'],
        atompair_feats: Float['b m m dap'] | Float['b nw w (w*2) dap'],
        mask: Bool['b n'],
        single_trunk_repr: Float['b n dst'],
        single_inputs_repr: Float['b n dsi'],
        pairwise_trunk: Float['b n n dpt'],
        pairwise_rel_pos_feats: Float['b n n dpr'],
        molecule_atom_lens: Int['b n'],
        token_bonds: Bool['b n n'],
        molecule_atom_indices: Int['b n'] | None = None,
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
        single_structure_input=False,
        verbose=None,
        filepath: List[str] | Tuple[str] | None = None,
    ) -> ElucidatedAtomDiffusionReturn:
        verbose = default(verbose, self.verbose)

        # diffusion loss

        if verbose:
            logger.info("Sampling noise distribution within EDM")

        dtype = atom_pos_ground_truth.dtype
        batch_size = atom_pos_ground_truth.shape[0]

        sigmas = self.noise_distribution(batch_size).type(dtype)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(atom_pos_ground_truth)

        noised_atom_pos = atom_pos_ground_truth + padded_sigmas * noise  # alphas are 1. in the paper

        if verbose:
            logger.info("Running preconditioned network forward pass within EDM")

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

        # section 3.7.1 equation 2 - weighted rigid aligned ground truth

        if verbose:
            logger.info("Calculating weighted rigid aligned ground truth within EDM")

        align_weights = calculate_weighted_rigid_align_weights(
            atom_pos_ground_truth=atom_pos_ground_truth,
            molecule_atom_lens=molecule_atom_lens,
            is_molecule_types=is_molecule_types,
            nucleotide_loss_weight=nucleotide_loss_weight,
            ligand_loss_weight=ligand_loss_weight,
        )

        atom_pos_aligned_ground_truth = self.weighted_rigid_align(
            pred_coords=denoised_atom_pos.float(),
            true_coords=atom_pos_ground_truth.float(),
            weights=align_weights.float(),
            mask=atom_mask,
        ).type(dtype)

        # section 4.2 - multi-chain permutation alignment

        if exists(molecule_atom_indices) and single_structure_input:
            if verbose:
                logger.info("Running multi-chain permutation alignment within EDM")

            try:
                atom_pos_aligned_ground_truth = self.multi_chain_permutation_alignment(
                    pred_coords=denoised_atom_pos,
                    true_coords=atom_pos_aligned_ground_truth,
                    molecule_atom_lens=molecule_atom_lens,
                    molecule_atom_indices=molecule_atom_indices,
                    token_bonds=token_bonds,
                    additional_molecule_feats=additional_molecule_feats,
                    is_molecule_types=is_molecule_types,
                    mask=atom_mask,
                )
            except Exception as e:
                # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                logger.warning(f"Skipping multi-chain permutation alignment {f'for {filepath}' if exists(filepath) else ''} due to: {e}")

        # main diffusion mse loss

        if verbose:
            logger.info("Calculating main diffusion loss within EDM")

        losses = F.mse_loss(denoised_atom_pos, atom_pos_aligned_ground_truth, reduction = 'none') / 3.
        losses = einx.multiply('b m c, b m -> b m c',  losses, align_weights)

        # regular loss weight as defined in EDM paper

        loss_weight_fn = self.karras_loss_weight if self.karras_formulation else self.loss_weight

        loss_weights = loss_weight_fn(padded_sigmas)

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
            if verbose:
                logger.info("Calculating bond loss within EDM")

            atompair_mask = to_pairwise_mask(atom_mask)

            denoised_cdist = torch.cdist(denoised_atom_pos, denoised_atom_pos, p = 2)
            normalized_cdist = torch.cdist(atom_pos_ground_truth, atom_pos_ground_truth, p = 2)

            bond_losses = F.mse_loss(denoised_cdist, normalized_cdist, reduction = 'none')
            bond_losses = bond_losses * loss_weights

            if atompair_mask.sum() > MAX_CONCURRENT_TENSOR_ELEMENTS:
                if verbose:
                    logger.info("Subsetting atom pairs for backprop within EDM")
                
                # randomly subset the atom pairs to supervise

                flat_atompair_mask_indices = torch.arange(atompair_mask.numel(), device=self.device)[atompair_mask.view(-1)]
                num_true_atompairs = flat_atompair_mask_indices.size(0)

                num_atompairs_to_ignore = num_true_atompairs - MAX_CONCURRENT_TENSOR_ELEMENTS
                ignored_atompair_indices = flat_atompair_mask_indices[torch.randperm(num_true_atompairs)[:num_atompairs_to_ignore]]
                
                atompair_mask.view(-1)[ignored_atompair_indices] = False

            bond_loss = bond_losses[atompair_mask].mean()

            total_loss = total_loss + bond_loss

        # proposed auxiliary smooth lddt loss

        smooth_lddt_loss = self.zero

        if add_smooth_lddt_loss:
            if verbose:
                logger.info("Calculating smooth lDDT loss within EDM")

            assert exists(is_molecule_types)

            is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim=-1)

            is_nucleotide_or_ligand_fields = tuple(
                batch_repeat_interleave(t, molecule_atom_lens)
                for t in is_nucleotide_or_ligand_fields
            )
            is_nucleotide_or_ligand_fields = tuple(
                pad_or_slice_to(t, length=align_weights.shape[-1], dim=-1)
                for t in is_nucleotide_or_ligand_fields
            )

            _, atom_is_dna, atom_is_rna, _, _ = is_nucleotide_or_ligand_fields

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

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

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

        eps = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        eps = eps.sigmoid().mean(dim = -1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask = mask, dim = (-1, -2), eps = 1)

        return 1. - lddt.mean()

class WeightedRigidAlign(Module):
    """Algorithm 28."""

    @typecheck
    @autocast("cuda", enabled=False)
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
        true_coords: Float["b m 3"],  # type: ignore - true coordinates
        weights: Float["b m"] | None = None,  # type: ignore - weights for each atom
        mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
        return_transforms: bool = False,
    ) -> Union[Float["b m 3"], Tuple[Float["b m 3"], Float["b 3 3"], Float["b 1 3"]]]:  # type: ignore
        """Compute the weighted rigid alignment.

        The check for ambiguous rotation and low rank of cross-correlation between aligned point
        clouds is inspired by
        https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param weights: Weights for each atom.
        :param mask: The mask for variable lengths.
        :param return_transform: Whether to return the transformation matrix.
        :return: The optimally aligned coordinates.
        """

        batch_size, num_points, dim = pred_coords.shape

        if not exists(weights):
            # if no weights are provided, assume uniform weights
            weights = torch.ones_like(pred_coords[..., 0])

        if exists(mask):
            # zero out all predicted and true coordinates where not an atom
            pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
            true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
            weights = einx.where("b n, b n, -> b n", mask, weights, 0.0)

        # Take care of weights broadcasting for coordinate dimension
        weights = rearrange(weights, "b n -> b n 1")

        # Compute weighted centroids
        true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        )
        pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        )

        # Center the coordinates
        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            logger.warning(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        # Compute the weighted covariance matrix
        cov_matrix = einsum(
            weights * true_coords_centered, pred_coords_centered, "b n i, b n j -> b i j"
        )

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

        det = torch.det(einsum(V, U_T, "b i j, b j k -> b i k"))

        # Ensure proper rotation matrix with determinant 1
        diag = torch.eye(dim, dtype=det.dtype, device=det.device)
        diag = repeat(diag, "i j -> b i j", b=batch_size).clone()

        diag[:, -1, -1] = det
        rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

        # Apply the rotation and translation
        true_aligned_coords = (
            einsum(rot_matrix, true_coords_centered, "b i j, b n j -> b n i") + pred_centroid
        )
        true_aligned_coords.detach_()

        if return_transforms:
            translation = true_centroid - einsum(
                rot_matrix, pred_centroid, "b i j, b ... j -> b ... i"
            )
            return true_aligned_coords, rot_matrix, translation

        return true_aligned_coords

class MultiChainPermutationAlignment(Module):
    """Section 4.2 of the AlphaFold 3 Supplement."""

    @typecheck
    def __init__(
        self,
        weighted_rigid_align: WeightedRigidAlign,
        **kwargs,
    ):
        super().__init__()
        self.weighted_rigid_align = weighted_rigid_align

    @staticmethod
    @typecheck
    def split_ground_truth_labels(gt_features: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        """Split ground truth features according to chains.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param gt_features: A dictionary within a PyTorch Dataset iteration, which is returned by
            the upstream DataLoader.iter() method. In the DataLoader pipeline, all tensors
            belonging to all the ground truth chains are concatenated. This function is needed to
            1) detect the number of chains, i.e., unique(asym_id) and 2) split the concatenated
            tensors back to individual ones that correspond to individual asym_ids.
        :return: A list of feature dictionaries with only necessary ground truth features required
            to finish multi-chain permutation. E.g., it will be a list of 5 elements if there are 5
            chains in total.
        """
        _, asym_id_counts = torch.unique(
            gt_features["asym_id"], sorted=True, return_counts=True, dim=-1
        )
        n_res = gt_features["asym_id"].shape[-1]

        def split_dim(shape):
            """Return the dimension index where the size is n_res."""
            return next(iter(i for i, size in enumerate(shape) if size == n_res), None)

        labels = list(
            map(
                dict,
                zip(
                    *[
                        [
                            (k, v)
                            for v in torch.split(
                                v_all, asym_id_counts.tolist(), dim=split_dim(v_all.shape)
                            )
                        ]
                        for k, v_all in gt_features.items()
                        if n_res in v_all.shape
                    ]
                ),
            )
        )
        return labels

    @staticmethod
    @typecheck
    def get_per_asym_token_index(features: Dict[str, Tensor], padding_value: int = -1) -> Dict[int, Int["b ..."]]:  # type: ignore
        """A function that retrieves a mapping denoting which token belong to which `asym_id`.
        
        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py
        
        :param features: A dictionary that contains input features after cropping.
        :return: A dictionary that records which region of the sequence belongs to which `asym_id`.
        """
        batch_size = features["token_index"].shape[0]

        unique_asym_ids = [i for i in torch.unique(features["asym_id"]) if i != padding_value]
        per_asym_token_index = {}
        for cur_asym_id in unique_asym_ids:
            asym_mask = (features["asym_id"] == cur_asym_id).bool()
            per_asym_token_index[int(cur_asym_id)] = rearrange(
                features["token_index"][asym_mask], "(b a) -> b a", b=batch_size
            )

        return per_asym_token_index

    @staticmethod
    @typecheck
    def get_entity_to_asym_list(
        features: Dict[str, Tensor], no_gaps: bool = False
    ) -> Dict[int, Tensor]:
        """Generate a dictionary mapping unique entity IDs to lists of unique asymmetry IDs
        (asym_id) for each entity.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param features: A dictionary containing data features, including `entity_id` and `asym_id` tensors.
        :param no_gaps: Whether to remove gaps in the `asym_id` values.
        :return: A dictionary where keys are unique entity IDs, and values are tensors of unique asymmetry IDs
            associated with each entity.
        """
        entity_to_asym_list = {}
        unique_entity_ids = torch.unique(features["entity_id"])

        # First pass: Collect all unique `cur_asym_id` values across all entities
        all_asym_ids = set()
        for cur_ent_id in unique_entity_ids:
            ent_mask = features["entity_id"] == cur_ent_id
            cur_asym_id = torch.unique(features["asym_id"][ent_mask])
            entity_to_asym_list[int(cur_ent_id)] = cur_asym_id
            all_asym_ids.update(cur_asym_id.tolist())

        # Second pass: Remap `asym_id` values to remove any gaps
        if no_gaps:
            sorted_asym_ids = sorted(all_asym_ids)
            remap_dict = {old_id: new_id for new_id, old_id in enumerate(sorted_asym_ids)}

            for cur_ent_id in entity_to_asym_list:
                cur_asym_id = entity_to_asym_list[cur_ent_id]
                remapped_asym_id = torch.tensor([remap_dict[id.item()] for id in cur_asym_id])
                entity_to_asym_list[cur_ent_id] = remapped_asym_id

        return entity_to_asym_list

    @typecheck
    def get_least_asym_entity_or_longest_length(
        self, batch: Dict[str, Tensor], input_asym_id: List[int], padding_value: int = -1
    ) -> Tuple[Tensor, List[Tensor]]:
        """Check how many subunit(s) one sequence has. Select the subunit that is less common,
        e.g., if the protein was AABBB then select one of the As as an anchor.

        If there is a tie, e.g. AABB, first check which sequence is the longest,
        then choose one of the corresponding subunits as an anchor.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: In this function, `batch` is the full ground truth features.
        :param input_asym_id: A list of `asym_ids` that are in the cropped input features.
        :param padding_value: The padding value used in the input features.
        :return: Selected ground truth `asym_ids` and a list of
            integer tensors denoting of all possible pred anchor candidates.
        """
        entity_to_asym_list = self.get_entity_to_asym_list(features=batch)
        unique_entity_ids = [i for i in torch.unique(batch["entity_id"]) if i != padding_value]
        entity_asym_count = {}
        entity_length = {}

        all_asym_ids = set()

        for entity_id in unique_entity_ids:
            asym_ids = torch.unique(batch["asym_id"][batch["entity_id"] == entity_id])

            all_asym_ids.update(asym_ids.tolist())

            # Make sure some asym IDs associated with ground truth entity ID exist in cropped prediction
            asym_ids_in_pred = [a for a in asym_ids if a in input_asym_id]
            if not asym_ids_in_pred:
                continue

            entity_asym_count[int(entity_id)] = len(asym_ids)

            # Calculate entity length
            entity_mask = batch["entity_id"] == entity_id
            entity_length[int(entity_id)] = entity_mask.sum(-1).mode().values.item()

        min_asym_count = min(entity_asym_count.values())
        least_asym_entities = [
            entity for entity, count in entity_asym_count.items() if count == min_asym_count
        ]

        # If multiple entities have the least asym_id count, return those with the longest length
        if len(least_asym_entities) > 1:
            max_length = max([entity_length[entity] for entity in least_asym_entities])
            least_asym_entities = [
                entity for entity in least_asym_entities if entity_length[entity] == max_length
            ]

        # If there are still multiple entities, return a random one
        if len(least_asym_entities) > 1:
            least_asym_entities = [random.choice(least_asym_entities)]  # nosec

        assert (
            len(least_asym_entities) == 1
        ), "There should be only one entity with the least `asym_id` count."
        least_asym_entities = least_asym_entities[0]

        anchor_gt_asym_id = random.choice(entity_to_asym_list[least_asym_entities])  # nosec
        anchor_pred_asym_ids = [
            asym_id
            for asym_id in entity_to_asym_list[least_asym_entities]
            if asym_id in input_asym_id
        ]

        # Since the entity ID to asym ID mapping is many-to-many, we need to select only
        # prediction asym IDs with equal length w.r.t. the sampled ground truth asym ID
        anchor_gt_asym_id_length = (
            (batch["asym_id"] == anchor_gt_asym_id).sum(-1).mode().values.item()
        )
        anchor_pred_asym_ids = [
            asym_id
            for asym_id in anchor_pred_asym_ids
            if (batch["asym_id"] == asym_id).sum(-1).mode().values.item()
            == anchor_gt_asym_id_length
        ]

        # Remap `asym_id` values to remove any gaps in the ground truth asym IDs,
        # but leave the prediction asym IDs as is since they are used for masking
        sorted_asym_ids = sorted(all_asym_ids)
        remap_dict = {old_id: new_id for new_id, old_id in enumerate(sorted_asym_ids)}

        remapped_anchor_gt_asym_id = torch.tensor([remap_dict[anchor_gt_asym_id.item()]])

        return remapped_anchor_gt_asym_id, anchor_pred_asym_ids

    @staticmethod
    @typecheck
    def calculate_input_mask(
        true_masks: List[Int["b ..."]],  # type: ignore
        anchor_gt_idx: int,
        asym_mask: Bool["b n"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
    ) -> Bool["b a"]:  # type: ignore
        """Calculate an input mask for downstream optimal transformation computation.

        :param true_masks: A list of masks from the ground truth chains. E.g., it will be a length
            of 5 if there are 5 chains in ground truth structure.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground
            truth anchor).
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if
            they do not belong to a specific asym ID.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the
            predicted features.
        :return: A boolean mask.
        """
        batch_size = pred_mask.shape[0]
        anchor_pred_mask = rearrange(
            pred_mask[asym_mask],
            "(b a) -> b a",
            b=batch_size,
        )
        anchor_true_mask = true_masks[anchor_gt_idx]
        input_mask = (anchor_true_mask * anchor_pred_mask).bool()
        return input_mask

    @typecheck
    def calculate_optimal_transform(
        self,
        true_poses: List[Float["b ... 3"]],  # type: ignore
        anchor_gt_idx: int,
        true_masks: List[Int["b ..."]],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        asym_mask: Bool["b n"],  # type: ignore
        pred_pos: Float["b n 3"],  # type: ignore
    ) -> Tuple[Float["b 3 3"], Float["b 1 3"]]:  # type: ignore
        """Take the selected anchor ground truth token center atom positions and the selected
        predicted anchor token center atom position and then calculate the optimal rotation matrix
        to align the ground-truth anchor and predicted anchor.

        Process:
        1) Select an anchor chain from ground truth, denoted by anchor_gt_idx, and an anchor chain from the predicted structure.
            Both anchor_gt and anchor_pred have exactly the same sequence.
        2) Obtain the token center atom positions corresponding to the selected anchor_gt,
            done be slicing the true_pose according to anchor_gt_token
        3) Calculate the optimal transformation that can best align the token center atoms of anchor_pred to those of anchor_gt
            via the Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm).

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., If there are 5 chains, this list will have a length of 5.
        :param anchor_gt_idx: A tensor with one integer in it (i.e., the index of selected ground truth anchor).
        :param true_masks: list of masks from the ground truth chains. E.g., it will be a length of 5 if there are
            5 chains in ground truth structure.
        :param pred_mask: A boolean tensor corresponding to the mask with which to mask the predicted features.
        :param asym_mask: A boolean tensor with which to mask out other elements in a tensor if they do not belong
            to a specific asym ID.
        :param pred_pos: A tensor of predicted token center atom positions.
        :return: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth as well as a matrix that records how the atoms should be shifted after applying `r`.
            N.b., Optimal alignment requires 1) a rotation and 2) a shift of the positions.
        """
        dtype = pred_pos.dtype
        batch_size = pred_pos.shape[0]

        input_mask = self.calculate_input_mask(
            true_masks=true_masks,
            anchor_gt_idx=anchor_gt_idx,
            asym_mask=asym_mask,
            pred_mask=pred_mask,
        )
        anchor_true_pos = true_poses[anchor_gt_idx]
        anchor_pred_pos = rearrange(
            pred_pos[asym_mask],
            "(b a) ... -> b a ...",
            b=batch_size,
        )
        _, r, x = self.weighted_rigid_align(
            pred_coords=anchor_pred_pos.float(),
            true_coords=anchor_true_pos.float(),
            mask=input_mask,
            return_transforms=True,
        )

        return r.type(dtype), x.type(dtype)

    @staticmethod
    @typecheck
    def apply_transform(pose: Float["b a 3"], r: Float["b 3 3"], x: Float["b 1 3"]) -> Float["b a 3"]:  # type: ignore
        """Apply the optimal transformation to the predicted token center atom positions.

        :param pose: A tensor of predicted token center atom positions.
        :param r: A rotation matrix that records the optimal rotation that will best align the selected anchor prediction to the
            selected anchor truth.
        :param x: A matrix that records how the atoms should be shifted after applying `r`.
        :return: A tensor of predicted token center atom positions after applying the optimal transformation.
        """
        aligned_pose = einsum(r, pose, "b i j, b n j -> b n i") + x
        aligned_pose.detach_()
        return aligned_pose

    @staticmethod
    @typecheck
    def batch_compute_rmsd(
        true_pos: Float["b a 3"],  # type: ignore
        pred_pos: Float["b a 3"],  # type: ignore
        mask: Bool["b a"] | None = None,  # type: ignore
        eps: float = 1e-6,
    ) -> Float["b"]:  # type: ignore
        """Calculate the root-mean-square deviation (RMSD) between predicted and ground truth
        coordinates.

        :param true_pos: The ground truth coordinates.
        :param pred_pos: The predicted coordinates.
        :param mask: The mask tensor.
        :param eps: A small value to prevent division by zero.
        :return: The RMSD.
        """
        # Apply mask if provided
        if exists(mask):
            true_coords = einx.where("b a, b a c, -> b a c", mask, true_pos, 0.0)
            pred_coords = einx.where("b a, b a c, -> b a c", mask, pred_pos, 0.0)

        # Compute squared differences across the last dimension (which is of size 3)
        sq_diff = torch.square(true_coords - pred_coords).sum(dim=-1)  # [b, m]

        # Compute mean squared deviation per batch
        msd = torch.mean(sq_diff, dim=-1)  # [b]

        # Replace NaN values with a large number to avoid issues
        msd = torch.nan_to_num(msd, nan=1e8)

        # Return the root mean square deviation per batch
        return torch.sqrt(msd + eps)  # [b]

    @typecheck
    def greedy_align(
        self,
        batch: Dict[str, Tensor],
        entity_to_asym_list: Dict[int, Tensor],
        pred_pos: Float["b n 3"],  # type: ignore
        pred_mask: Float["b n"],  # type: ignore
        true_poses: List[Float["b ... 3"]],  # type: ignore
        true_masks: List[Int["b ..."]],  # type: ignore
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """
        Implement Algorithm 4 in the Supplementary Information of the AlphaFold-Multimer paper:
            Evans, R et al., 2022 Protein complex prediction with AlphaFold-Multimer,
            bioRxiv 2021.10.04.463034; doi: https://doi.org/10.1101/2021.10.04.463034

        NOTE: The tuples in the returned list begin are zero-indexed.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param batch: A dictionary of ground truth features.
        :param entity_to_asym_list: A dictionary recording which asym ID(s) belong to which entity ID.
        :param pred_pos: Predicted positions of token center atoms from the results of model.forward().
        :param pred_mask: A boolean tensor that masks `pred_pos`.
        :param true_poses: A list of tensors, corresponding to the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param true_masks: A list of tensors, corresponding to the masks of the token center atom positions of the ground truth structure.
            E.g., if there are 5 chains, this list will have a length of 5.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that provides instructions for how the ground truth chains should be permuted.
            E.g., if 3 chains in the input structure have the same sequences, an example return value would be:
            `[(0, 2), (1, 1), (2, 0)]`, meaning the first chain in the predicted structure should be aligned
            to the third chain in the ground truth and the second chain in the predicted structure is fine
            to stay with the second chain in the ground truth.
        """
        batch_size = pred_pos.shape[0]

        used = [
            # NOTE: This is a list the keeps a record of whether a ground truth chain has been used.
            False
            for _ in range(len(true_poses))
        ]
        alignments = []

        unique_asym_ids = [i for i in torch.unique(batch["asym_id"]) if i != padding_value]

        for cur_asym_id in unique_asym_ids:
            i = int(cur_asym_id)

            asym_mask = batch["asym_id"] == cur_asym_id
            cur_entity_ids = rearrange(
                batch["entity_id"][asym_mask],
                "(b a) -> b a",
                b=batch_size,
            )

            # NOTE: Here, we assume there can be multiple unique entity IDs associated
            # with a given asym ID. This is a valid assumption when the original batch
            # contains a single unique structure that has one or more chains spread
            # across multiple entities (e.g., in the case of ligands residing in
            # a protein-majority chain).

            unique_cur_entity_ids = torch.unique(cur_entity_ids, dim=-1).unbind(dim=-1)

            for batch_cur_entity_id in unique_cur_entity_ids:
                cur_pred_pos = rearrange(
                    pred_pos[asym_mask],
                    "(b a) ... -> b a ...",
                    b=batch_size,
                )
                cur_pred_mask = rearrange(
                    pred_mask[asym_mask],
                    "(b a) -> b a",
                    b=batch_size,
                )

                best_rmsd = torch.inf
                best_idx = None

                # NOTE: Here, we assume there is only one unique entity ID per batch,
                # which is a valid assumption only when the original batch size is 1
                # (meaning only a single unique structure is represented in the batch).

                unique_cur_entity_id = torch.unique(batch_cur_entity_id)
                assert (
                    len(unique_cur_entity_id) == 1
                ), "There should be only one unique entity ID per batch."
                cur_asym_list = entity_to_asym_list[int(unique_cur_entity_id)]

                for next_asym_id in cur_asym_list:
                    j = int(next_asym_id)

                    if not used[j]:  # NOTE: This is a possible candidate.
                        cropped_pos = true_poses[j]
                        mask = true_masks[j]

                        rmsd = self.batch_compute_rmsd(
                            true_pos=cropped_pos.mean(1, keepdim=True),
                            pred_pos=cur_pred_pos.mean(1, keepdim=True),
                            mask=(
                                cur_pred_mask.any(-1, keepdim=True) * mask.any(-1, keepdim=True)
                            ),
                        ).mean()

                        if rmsd < best_rmsd:
                            # NOTE: We choose the permutation that minimizes the batch-wise
                            # average RMSD of the predicted token center atom centroid coordinates
                            # with respect to the ground truth token center atom centroid coordinates.
                            best_rmsd = rmsd
                            best_idx = j

                if exists(best_idx):
                    # NOTE: E.g., for ligands within a protein-majority chain, we may have
                    # multiple unique entity IDs associated with a given asym ID. In this case,
                    # we need to ensure that we do not reuse a chain that has already been used
                    # in the permutation alignment process.
                    used[best_idx] = True
                    alignments.append((i, best_idx))

        assert all(used), "All chains should be used in the permutation alignment process."
        return alignments

    @staticmethod
    @typecheck
    def pad_features(feature_tensor: Tensor, num_tokens_pad: int, pad_dim: int) -> Tensor:
        """Pad an input feature tensor. Padding values will be 0 and put behind the true feature
        values.

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param feature_tensor: A feature tensor to pad.
        :param num_tokens_pad: The number of tokens to pad.
        :param pad_dim: Along which dimension of `feature_tensor` to pad.
        :return: A padded feature tensor.
        """
        pad_shape = list(feature_tensor.shape)
        pad_shape[pad_dim] = num_tokens_pad
        padding_tensor = feature_tensor.new_zeros(pad_shape, device=feature_tensor.device)
        return torch.concat((feature_tensor, padding_tensor), dim=pad_dim)

    @typecheck
    def merge_labels(
        self,
        labels: List[Dict[str, Tensor]],
        alignments: List[Tuple[int, int]],
        original_num_tokens: int,
        dimension_to_merge: int = 1,
    ) -> Dict[str, Tensor]:
        """Merge ground truth labels according to permutation results.

        Adapted from:
        https://github.com/dptech-corp/Uni-Fold/blob/b1c89a2cebd4e4ee4c47b4e443f92beeb9138fbb/unifold/losses/chain_align.py#L176C1-L176C1
        and
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param labels: A list of original ground truth feats. E.g., if there are 5 chains,
            `labels` will have a length of 5.
        :param alignments: A list of tuples, each entry specifying the corresponding label of the asym ID.
        :param original_num_tokens: An integer corresponding to the number of tokens specified
            by one's (e.g., training-time) crop size.
        :param dimension_to_merge: The dimension along which to merge the labels.
        :return: A new dictionary of permuted ground truth features.
        """
        outs = {}
        for k in labels[0].keys():
            cur_out = {}
            for i, j in alignments:
                label = labels[j][k]
                cur_out[i] = label

            cur_out = [x[1] for x in sorted(cur_out.items())]
            if len(cur_out) > 0:
                new_v = torch.concat(cur_out, dim=dimension_to_merge)

                # Check whether padding is needed.
                if new_v.shape[dimension_to_merge] != original_num_tokens:
                    num_tokens_pad = original_num_tokens - new_v.shape[dimension_to_merge]
                    new_v = self.pad_features(new_v, num_tokens_pad, pad_dim=dimension_to_merge)

                outs[k] = new_v

        return outs

    @typecheck
    def compute_permutation_alignment(
        self,
        out: Dict[str, Tensor],
        features: Dict[str, Tensor],
        ground_truth: Dict[str, Tensor],
        padding_value: int = -1,
    ) -> List[Tuple[int, int]]:
        """A method that permutes chains in ground truth before calculating the loss because the
        mapping between the predicted and ground truth will become arbitrary. The model cannot be
        assumed to predict chains in the same order as the ground truth. Thus, this function picks
        the optimal permutation of predicted chains that best matches the ground truth, by
        minimising the RMSD (i.e., the best permutation of ground truth chains is selected based on
        which permutation has the lowest RMSD calculation).

        Details are described in Section 7.3 in the Supplementary of AlphaFold-Multimer paper:
        https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2

        Adapted from:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/multi_chain_permutation.py

        :param out: A dictionary of output tensors from model.forward().
        :param features: A dictionary of feature tensors that are used as input for model.forward().
        :param ground_truth: A list of dictionaries of features corresponding to chains in ground truth structure.
            E.g., it will be a length of 5 if there are 5 chains in ground truth structure.
        :param padding_value: The padding value used in the input features.
        :return: A list of tuple(int, int) that instructs how ground truth chains should be permutated.
        """
        num_tokens = features["token_index"].shape[-1]

        unique_asym_ids = set(torch.unique(features["asym_id"]).tolist())
        unique_asym_ids.discard(padding_value)  # Remove padding value
        is_monomer = len(unique_asym_ids) == 1

        per_asym_token_index = self.get_per_asym_token_index(
            features=features, padding_value=padding_value
        )

        if is_monomer:
            best_alignments = list(enumerate(range(len(per_asym_token_index))))
            return best_alignments

        best_rmsd = torch.inf
        best_alignments = None

        # 1. Choose the least ambiguous ground truth "anchor" chain.
        # For example, in an A3B2 complex an arbitrary B chain is chosen.
        # In the event of a tie e.g., A2B2 stoichiometry, the longest chain
        # is chosen, with the hope that in general the longer chains are
        # likely to have higher confidence predictions.

        # 2. Select the prediction anchor chain from the set of all prediction
        # chains with the same sequence as the ground truth anchor chain.

        anchor_gt_asym, anchor_pred_asym_ids = self.get_least_asym_entity_or_longest_length(
            batch=ground_truth,
            input_asym_id=list(unique_asym_ids),
        )
        entity_to_asym_list = self.get_entity_to_asym_list(features=ground_truth, no_gaps=True)
        labels = self.split_ground_truth_labels(gt_features=ground_truth)
        anchor_gt_idx = int(anchor_gt_asym)

        # 3. Optimally align the ground truth anchor chain to the prediction
        # anchor chain using a rigid alignment algorithm.

        pred_pos = out["pred_coords"]
        pred_mask = out["mask"].to(dtype=pred_pos.dtype)

        true_poses = [label["true_coords"] for label in labels]
        true_masks = [label["mask"].long() for label in labels]

        # Assignment Stage - Section 7.3.2 of the AlphaFold-Multimer Paper

        # 1. Greedily assign each of the predicted chains to their nearest
        # neighbour of the same sequence in the ground truth. These assignments
        # define the optimal permutation to apply to the ground truth chains.
        # Nearest neighbours are defined as the chains with the smallest distance
        # between the average of their token center atom coordinates.

        # 2. Repeat the above alignment and assignment stages for all valid choices
        # of the prediction anchor chain given the ground truth anchor chain.

        # 3. Finally, we pick the permutation that minimizes the RMSD between the
        # token center atom coordinate averages of the predicted and ground truth chains.

        for candidate_pred_anchor in anchor_pred_asym_ids:
            asym_mask = (features["asym_id"] == candidate_pred_anchor).bool()

            r, x = self.calculate_optimal_transform(
                true_poses=true_poses,
                anchor_gt_idx=anchor_gt_idx,
                true_masks=true_masks,
                pred_mask=pred_mask,
                asym_mask=asym_mask,
                pred_pos=pred_pos,
            )

            # Apply transforms.
            aligned_true_poses = [
                self.apply_transform(pose.to(r.dtype), r, x) for pose in true_poses
            ]

            alignments = self.greedy_align(
                batch=features,
                entity_to_asym_list=entity_to_asym_list,
                pred_pos=pred_pos,
                pred_mask=pred_mask,
                true_poses=aligned_true_poses,
                true_masks=true_masks,
            )

            merged_labels = self.merge_labels(
                labels=labels,
                alignments=alignments,
                original_num_tokens=num_tokens,
            )

            aligned_true_pos = self.apply_transform(merged_labels["true_coords"].to(r.dtype), r, x)

            rmsd = self.batch_compute_rmsd(
                true_pos=aligned_true_pos.mean(1, keepdim=True),
                pred_pos=pred_pos.mean(1, keepdim=True),
                mask=(
                    pred_mask.any(-1, keepdim=True) * merged_labels["mask"].any(-1, keepdim=True)
                ),
            ).mean()

            if rmsd < best_rmsd:
                # NOTE: We choose the permutation that minimizes the batch-wise
                # average RMSD of the predicted token center atom centroid coordinates
                # with respect to the ground truth token center atom centroid coordinates.
                best_rmsd = rmsd
                best_alignments = alignments

        # NOTE: The above algorithm naturally generalizes to both training and inference
        # contexts (i.e., with and without cropping) by, where applicable, pre-applying
        # cropping to the (ground truth) input coordinates and features.

        assert exists(best_alignments), "Best alignments must be found."
        return best_alignments

    @typecheck
    def forward(
        self,
        pred_coords: Float["b m 3"],  # type: ignore - predicted coordinates
        true_coords: Float["b m 3"],  # type: ignore - true coordinates
        molecule_atom_lens: Int["b n"],  # type: ignore - molecule atom lengths
        molecule_atom_indices: Int["b n"],  # type: ignore - molecule atom indices
        token_bonds: Bool["b n n"],  # type: ignore - token bonds
        additional_molecule_feats: Int[f"b n {ADDITIONAL_MOLECULE_FEATS}"] | None = None,  # type: ignore - additional molecule features
        is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"] | None = None,  # type: ignore - molecule types
        mask: Bool["b m"] | None = None,  # type: ignore - mask for variable lengths
        eps: int = int(1e6),
    ) -> Float["b m 3"]:  # type: ignore
        """Compute the multi-chain permutation alignment.

        NOTE: This function assumes that the ground truth features are batched yet only contain
        features for the same structure. This is the case after performing data augmentation
        with a batch size of 1 in the `Alphafold3` module's forward pass. If the batched
        ground truth features represent multiple different structures, this function will not
        return correct results.

        :param pred_coords: Predicted coordinates.
        :param true_coords: True coordinates.
        :param molecule_atom_lens: The molecule atom lengths.
        :param molecule_atom_indices: The molecule atom indices.
        :param token_bonds: The token bonds.
        :param is_molecule_types: Molecule type of each atom.
        :param mask: The mask for variable lengths.
        :param eps: A large integer value to server as a placeholder asym ID.
        :return: The optimally chain-permuted aligned coordinates.
        """
        num_atoms = pred_coords.shape[1]

        if not exists(additional_molecule_feats) or not exists(is_molecule_types):
            # NOTE: If no chains or no molecule types are specified,
            # we cannot perform multi-chain permutation alignment.
            true_coords.detach_()
            return true_coords

        if exists(mask):
            # Zero out all predicted and true coordinates where not an atom.
            pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
            true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)

        # Alignment Stage - Section 7.3.1 of the AlphaFold-Multimer Paper

        _, token_index, token_asym_id, token_entity_id, _ = additional_molecule_feats.unbind(
            dim=-1
        )

        # NOTE: Ligands covalently bonded to polymer chains are to be permuted
        # in sync with the corresponding chains by assigning them the same
        # asymmetric unit ID (asym_id) to group all covalently bonded
        # components together.
        polymer_indices = [IS_PROTEIN_INDEX, IS_RNA_INDEX, IS_DNA_INDEX]
        ligand_indices = [IS_LIGAND_INDEX, IS_METAL_ION_INDEX]

        is_polymer_types = is_molecule_types[..., polymer_indices].any(-1)
        is_ligand_types = is_molecule_types[..., ligand_indices].any(-1)

        polymer_ligand_pair_mask = is_polymer_types[..., None] & is_ligand_types[..., None, :]
        polymer_ligand_pair_mask = polymer_ligand_pair_mask | polymer_ligand_pair_mask.transpose(
            -1, -2
        )

        covalent_bond_mask = polymer_ligand_pair_mask & token_bonds

        is_covalent_residue_mask = covalent_bond_mask.any(-1)
        is_covalent_ligand_mask = is_ligand_types & is_covalent_residue_mask

        # NOTE: Covalent ligand-polymer bond pairs may be many-to-many, so
        # we need to group them together by assigning covalent ligands the same
        # asym IDs as the polymer chains to which they are most frequently bonded.
        covalent_bonded_asym_id = torch.where(
            covalent_bond_mask, token_asym_id[..., None], eps
        )

        covalent_bond_mode_values, _ = covalent_bonded_asym_id.mode(dim=-1, keepdim=False)
        mapped_token_asym_id = torch.where(
            is_covalent_ligand_mask, covalent_bond_mode_values, token_asym_id
        )
        mapped_atom_asym_id = batch_repeat_interleave(mapped_token_asym_id, molecule_atom_lens)

        # Move ligand coordinates to be adjacent to their covalently bonded polymer chains.
        _, mapped_atom_sorted_indices = torch.sort(mapped_atom_asym_id, dim=1)
        mapped_atom_true_coords = torch.gather(
            true_coords, dim=1, index=mapped_atom_sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Segment the ground truth coordinates into chains.
        labels = self.split_ground_truth_labels(
            dict(asym_id=mapped_atom_asym_id, true_coords=mapped_atom_true_coords)
        )

        # Pool atom-level features into token-level features.
        mol_atom_indices = repeat(molecule_atom_indices, "b m -> b m d", d=true_coords.shape[-1])

        token_pred_coords = torch.gather(pred_coords, 1, mol_atom_indices)
        token_true_coords = torch.gather(true_coords, 1, mol_atom_indices)
        token_mask = torch.gather(mask, 1, molecule_atom_indices)

        # Permute ground truth chains.
        out = {"pred_coords": token_pred_coords, "mask": token_mask}
        features = {
            "asym_id": token_asym_id,
            "entity_id": token_entity_id,
            "token_index": token_index,
        }
        ground_truth = {
            "true_coords": token_true_coords,
            "mask": token_mask,
            "asym_id": token_asym_id,
            "entity_id": token_entity_id,
        }

        alignments = self.compute_permutation_alignment(
            out=out,
            features=features,
            ground_truth=ground_truth,
        )

        # Reorder ground truth coordinates according to permutation results.
        labels = self.merge_labels(
            labels=labels,
            alignments=alignments,
            original_num_tokens=num_atoms,
        )

        permuted_true_coords = labels["true_coords"].detach()
        return permuted_true_coords

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
        true_frames: Float['b n 3 3'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b n n']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        pred_frames: predicted frames
        true_frames: true frames
        """
        # to pairs

        seq = pred_coords.shape[1]
        
        pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
        seq2pair = partial(rearrange, pattern='b (n m) ... -> b n m ...', n = seq, m = seq)
        
        pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = seq))
        pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = seq))
        pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = seq))
        pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = seq))
        
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)

        # Compute alignment errors
        alignment_errors = F.pairwise_distance(pred_coords_transformed, true_coords_transformed, eps = self.eps)

        alignment_errors = seq2pair(alignment_errors)

        # Masking
        if exists(mask):
            pair_mask = to_pairwise_mask(mask)
            alignment_errors = einx.where('b i j, b i j, -> b i j', pair_mask, alignment_errors, 0.)

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
