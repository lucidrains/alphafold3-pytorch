from functools import wraps
from typing import List, Tuple, Union, Any

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import Tensor

from alphafold3_pytorch.tensor_typing import Bool, Float, Int, typecheck
from alphafold3_pytorch.utils.utils import exists

# constants

Shape = Union[Tuple[int, ...], List[int]]

# helper functions

# default scheduler used in paper w/ warmup

def default_lambda_lr_fn(steps: int) -> float:
    """Default lambda learning rate function.

    :param steps: The number of steps taken.
    :return: The learning rate.
    """
    # 1000 step warmup
    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps
    steps -= 1000
    return 0.95 ** (steps / 5e4)


def max_neg_value(t: Tensor) -> Tensor:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def log(t: Tensor, eps=1e-20) -> Tensor:
    """
    Run a safe log function that clamps the input to be above `eps` to avoid `log(0)`.

    :param t: The input tensor.
    :param eps: The epsilon value.
    :return: Tensor in the log domain.
    """
    return torch.log(t.clamp(min=eps))


def divisible_by(num: int, den: int) -> bool:
    """
    Check if a number is divisible by another number.

    :param num: The numerator.
    :param den: The denominator.
    :return: True if `num` is divisible by `den`, False otherwise.
    """
    return (num % den) == 0


def pack_one(t: Tensor, pattern: str) -> Tuple[Tensor, List[Shape]]:
    """
    Pack a single tensor into a tuple of tensors with the given pattern.

    :param t: The tensor to pack.
    :param pattern: The pattern with which to pack.
    :return: The packed tensor along with the shape(s) of the tensor.
    """
    return pack([t], pattern)


def unpack_one(t: Tensor, ps: List[Shape], pattern: str) -> List[Tensor]:
    """
    Unpack a single tensor from a tuple of tensors with the given pattern.

    :param t: The tensor to unpack.
    :param ps: The shapes of the tensors.
    :param pattern: The pattern with which to unpack.
    :return: The unpacked tensor.
    """
    return unpack(t, ps, pattern)[0]


def softclamp(t: Tensor, value: float) -> Tensor:
    """
    Perform a soft clamp on a Tensor.

    :param t: The Tensor.
    :param value: The value to clamp to.
    :return: The soft clamped Tensor
    """
    return (t / value).tanh() * value


def exclusive_cumsum(t: Tensor, dim: int = -1) -> Tensor:
    """
    Perform an exclusive cumulative summation on a Tensor.

    :param t: The Tensor.
    :param dim: The dimension to sum over.
    :return: The exclusive cumulative sum Tensor.
    """
    return t.cumsum(dim=dim) - t


# decorators


def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)

    return inner


@typecheck
def pad_at_dim(t, pad: Tuple[int, int], *, dim=-1, value=0.0) -> Tensor:
    """Pad a Tensor at a specific dimension.

    :param t: The Tensor.
    :param pad: The padding.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


# padding and slicing


@typecheck
def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    """
    Slice a Tensor at a specific dimension.

    :param t: The Tensor.
    :param dim_slice: The slice object.
    :param dim: The dimension to slice.
    :return: The sliced Tensor.
    """
    dim += t.ndim if dim < 0 else 0
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


@typecheck
def pad_to_length(t: Tensor, length: int, *, dim: int = -1, value=0) -> Tensor:
    """
    Pad a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    padding = max(length - t.shape[dim], 0)

    if padding == 0:
        return t

    return pad_at_dim(t, (0, padding), dim=dim, value=value)


@typecheck
def pad_or_slice_to(t: Tensor, length: int, *, dim: int, pad_value=0) -> Tensor:
    """
    Pad or slice a Tensor to a specific length at a specific dimension.

    :param t: The Tensor.
    :param length: The length to pad or slice to.
    :param dim: The dimension to pad or slice.
    :param pad_value: The value to pad with.
    :return: The padded or sliced Tensor.
    """
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_to_length(t, length, dim=dim, value=pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim=dim)

    return t


@typecheck
def pad_to_multiple(t: Tensor, multiple: int, *, dim=-1, value=0.0) -> Tensor:
    """
    Pad a Tensor to a multiple of a specific number at a specific dimension.

    :param t: The Tensor.
    :param multiple: The multiple to pad to.
    :param dim: The dimension to pad.
    :param value: The value to pad with.
    :return: The padded Tensor.
    """
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim=dim, value=value)


@typecheck
def concat_previous_window(t: Tensor, *, dim_seq: int, dim_window: int) -> Tensor:
    """
    Concatenate the previous window of a Tensor.

    :param t: The Tensor.
    :param dim_seq: The sequence dimension.
    :param dim_window: The window dimension.
    :return: The concatenated Tensor.
    """
    t = pad_at_dim(t, (1, 0), dim=dim_seq, value=0.0)

    t = torch.cat(
        (
            slice_at_dim(t, slice(None, -1), dim=dim_seq),
            slice_at_dim(t, slice(1, None), dim=dim_seq),
        ),
        dim=dim_window,
    )

    return t


@typecheck
def pad_and_window(t: Float["b n ..."] | Int["b n ..."], window_size: int) -> Tensor:  # type: ignore
    """
    Pad and window a Tensor.

    :param t: The Tensor.
    :param window_size: The window size.
    :return: The padded and windowed Tensor.
    """
    t = pad_to_multiple(t, window_size, dim=1)
    t = rearrange(t, "b (n w) ... -> b n w ...", w=window_size)
    return t


# to atompair input functions


@typecheck
def atom_ref_pos_to_atompair_inputs(
    atom_ref_pos: Float["... m 3"],  # type: ignore
    atom_ref_space_uid: Int["... m"],  # type: ignore
) -> Float["... m m 5"]:  # type: ignore
    """
    Convert atom reference positions and spaces to atompair inputs.

    :param atom_ref_pos: The atom reference positions.
    :param atom_ref_space_uid: The atom reference space UIDs.
    :return: The atompair inputs.
    """

    # Algorithm 5 - lines 2-6
    # allow for either batched or single

    atom_ref_pos, batch_packed_shape = pack_one(atom_ref_pos, "* m c")
    atom_ref_space_uid, _ = pack_one(atom_ref_space_uid, "* m")

    assert atom_ref_pos.shape[0] == atom_ref_space_uid.shape[0]

    # line 2

    pairwise_rel_pos = einx.subtract("b i c, b j c -> b i j c", atom_ref_pos, atom_ref_pos)

    # line 3

    same_ref_space_mask = einx.equal("b i, b j -> b i j", atom_ref_space_uid, atom_ref_space_uid)

    # line 5 - pairwise inverse squared distance

    atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim=-1, p=2) ** 2) ** -1

    # concat all into atompair_inputs for projection into atompair_feats within AlphaFold3

    atompair_inputs, _ = pack(
        (
            pairwise_rel_pos,
            atom_inv_square_dist,
            same_ref_space_mask.float(),
        ),
        "b i j *",
    )

    # mask out

    atompair_inputs = einx.where(
        "b i j, b i j dapi, -> b i j dapi", same_ref_space_mask, atompair_inputs, 0.0
    )

    # reconstitute optional batch dimension

    atompair_inputs = unpack_one(atompair_inputs, batch_packed_shape, "* i j dapi")

    # return

    return atompair_inputs


# packed atom representation functions


@typecheck
def lens_to_mask(
    lens: Int["b ..."], max_len: int | None = None  # type: ignore
) -> Bool["... m"]:  # type: ignore
    """
    Convert a Tensor of lengths to a mask Tensor.

    :param lens: The lengths Tensor.
    :param max_len: The maximum length.
    :return: The mask Tensor.
    """
    device = lens.device
    if not exists(max_len):
        max_len = lens.amax()
    arange = torch.arange(max_len, device=device)
    return einx.less("m, ... -> ... m", arange, lens)


@typecheck
def mean_pool_with_lens(
    feats: Float["b m d"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> Float["b n d"]:  # type: ignore
    """
    Perform mean pooling on a Tensor with the given lengths.

    :param feats: The features Tensor.
    :param lens: The lengths Tensor.
    :return: The mean pooled Tensor.
    """
    seq_len = feats.shape[1]

    mask = lens > 0
    assert (
        lens.sum(dim=-1) <= seq_len
    ).all(), (
        "One of the lengths given exceeds the total sequence length of the features passed in."
    )

    cumsum_feats = feats.cumsum(dim=1)
    cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value=0.0)

    cumsum_indices = lens.cumsum(dim=1)
    cumsum_indices = F.pad(cumsum_indices, (1, 0), value=0)

    sel_cumsum = einx.get_at("b [m] d, b n -> b n d", cumsum_feats, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    avg = einx.divide("b n d, b n", summed, lens.clamp(min=1))
    avg = einx.where("b n, b n d, -> b n d", mask, avg, 0.0)
    return avg


@typecheck
def repeat_consecutive_with_lens(
    feats: Float["b n ..."] | Bool["b n"] | Int["b n"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> Float["b m ..."] | Bool["b m"] | Int["b m"]:  # type: ignore
    """
    Repeat a Tensor's values consecutively with the given lengths.

    :param feats: The features Tensor.
    :param lens: The lengths Tensor.
    :return: The repeated Tensor.
    """

    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device=device)

    offsets = exclusive_cumsum(lens)
    indices = einx.add("w, b n -> b n w", arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.sum(dim=-1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

    indices = indices.masked_fill(~mask, max_len)  # scatter to sink position for padding
    indices = rearrange(indices, "b n w -> b (n w)")

    # scatter

    seq_arange = torch.arange(seq, device=device)
    seq_arange = repeat(seq_arange, "n -> (n w)", w=window_size)

    output_indices = einx.set_at("b [m],  b nw, nw -> b [m]", output_indices, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    output = einx.get_at("b [n] ..., b m -> b m ...", feats, output_indices)

    # final mask

    mask_value = False if dtype == torch.bool else 0

    output = einx.where("b n, b n ..., -> b n ...", output_mask, output, mask_value)

    return output
