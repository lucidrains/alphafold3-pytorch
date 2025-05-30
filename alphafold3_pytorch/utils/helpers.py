from __future__ import annotations

from functools import wraps, partial

import torch
from torch import Tensor, is_tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from torch.nn import (
    Module,
)

from beartype.typing import (
    Callable,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    Shaped,
    typecheck,
)

from alphafold3_pytorch.attention import (
    pad_to_multiple,
)

from alphafold3_pytorch.utils.utils import get_gpu_type

from alphafold3_pytorch.utils.model_utils import (
    pack_one
)
# other external libs

from importlib.metadata import version

import einx
from einops import rearrange, repeat, reduce

# constants

# NOTE: for some types of (e.g., AMD ROCm) GPUs, this represents
# the maximum number of elements that can be processed simultaneously
# for a given tensor. For reference, see https://github.com/pytorch/pytorch/issues/136291.
MAX_CONCURRENT_TENSOR_ELEMENTS = int(2e9) if "ROCm" in get_gpu_type() else float("inf")

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

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def l2norm(t, eps = 1e-20, dim = -1):
    return F.normalize(t, p = 2, eps = eps, dim = dim)

def freeze_(m: Module):
    for p in m.parameters():
        p.requires_grad = False

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

def dict_to_device(d, device):
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, d)

def exclusive_cumsum(t, dim = -1):
    return t.cumsum(dim = dim) - t

@typecheck
def symmetrize(t: Float['b n n ...']) -> Float['b n n ...']:
    return t + rearrange(t, 'b i j ... -> b j i ...')

@typecheck
def masked_average(
    t: Shaped['...'],
    mask: Shaped['...'],
    *,
    dim: int | Tuple[int, ...],
    eps = 1.
) -> Float['...']:

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim)
    return num / den.clamp(min = eps)


@typecheck
def should_checkpoint(
    self: Module,
    inputs: Tensor | Tuple[Tensor, ...],
    check_instance_variable: str | None = 'checkpoint'
) -> bool:
    if is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training and
        any([i.requires_grad for i in inputs]) and
        (not exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )

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
def to_pairwise_mask(
    mask_i: Bool['... n'],
    mask_j: Bool['... n'] | None = None
) -> Bool['... n n']:

    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)

@typecheck
def mean_pool_with_lens(
    feats: Float['b m d'],
    lens: Int['b n']
) -> Float['b n d']:

    summed, mask = sum_pool_with_lens(feats, lens)
    avg = einx.divide('b n d, b n', summed, lens.clamp(min = 1))
    avg = einx.where('b n, b n d, -> b n d', mask, avg, 0.)
    return avg

@typecheck
def sum_pool_with_lens(
    feats: Float['b m d'],
    lens: Int['b n']
) -> tuple[Float['b n d'], Bool['b n']]:

    seq_len = feats.shape[1]

    mask = lens > 0
    assert (lens.sum(dim = -1) <= seq_len).all(), 'one of the lengths given exceeds the total sequence length of the features passed in'

    cumsum_feats = feats.cumsum(dim = 1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value = 0.)

    cumsum_indices = lens.cumsum(dim = 1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value = 0)

    # sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    cumsum_indices = repeat(cumsum_indices, 'b n -> b n d', d = cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    return summed, mask

@typecheck
def mean_pool_fixed_windows_with_mask(
    feats: Float['b m d'],
    mask: Bool['b m'],
    window_size: int,
    return_mask_and_inverse: bool = False,
) -> Float['b n d'] | Tuple[Float['b n d'], Bool['b n'], Callable[[Float['b m d']], Float['b n d']]]:

    seq_len = feats.shape[-2]
    assert divisible_by(seq_len, window_size)

    feats = einx.where('b m, b m d, -> b m d', mask, feats, 0.)

    num = reduce(feats, 'b (n w) d -> b n d', 'sum', w = window_size)
    den = reduce(mask.float(), 'b (n w) -> b n 1', 'sum', w = window_size)

    avg = num / den.clamp(min = 1.)

    if not return_mask_and_inverse:
        return avg

    pooled_mask = reduce(mask, 'b (n w) -> b n', 'any', w = window_size)

    @typecheck
    def inverse_fn(pooled: Float['b n d']) -> Float['b m d']:
        unpooled = repeat(pooled, 'b n d -> b (n w) d', w = window_size)
        unpooled = einx.where('b m, b m d, -> b m d', mask, unpooled, 0.)
        return unpooled

    return avg, pooled_mask, inverse_fn

@typecheck
def batch_repeat_interleave(
    feats: Float['b n ...'] | Bool['b n ...'] | Bool['b n'] | Int['b n'],
    lens: Int['b n'],
    output_padding_value: float | int | bool | None = None, # this value determines what the output padding value will be
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

    total_lens = lens.clamp(min = 0).sum(dim = -1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter

    seq_arange = torch.arange(seq, device = device)
    seq_arange = repeat(seq_arange, 'n -> b (n w)', b = batch, w = window_size)

    # output_indices = einx.set_at('b [m], b nw, b nw -> b [m]', output_indices, indices, seq_arange)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    # output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    feats, unpack_one = pack_one(feats, 'b n *')
    output_indices = repeat(output_indices, 'b m -> b m d', d = feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    output = einx.where(
        'b n, b n ..., -> b n ...',
        output_mask, output, output_padding_value
    )

    return output

@typecheck
def batch_repeat_interleave_pairwise(
    pairwise: Float['b n n d'],
    molecule_atom_lens: Int['b n']
) -> Float['b m m d']:

    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)

    molecule_atom_lens = repeat(molecule_atom_lens, 'b ... -> (b r) ...', r = pairwise.shape[1])
    pairwise, unpack_one = pack_one(pairwise, '* n d')
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)
    return unpack_one(pairwise)