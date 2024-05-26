from functools import wraps
from typing import List, Tuple, Union

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import Tensor

from alphafold3_pytorch.utils.typing import Bool, Float, Int, typecheck
from alphafold3_pytorch.utils.utils import exists

# constants

Shape = Union[Tuple[int, ...], List[int]]

# helper functions


def default_lambda_lr_fn(steps: int) -> float:
    """Default lambda learning rate function.

    :param steps: The number of steps.
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


def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)

    return inner


@typecheck
def pad_at_dim(t, pad: Tuple[int, int], *, dim=-1, value=0.0):
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
    assert (lens.sum(dim=-1) <= seq_len).all(), "One of the lengths given exceeds the total sequence length of the features passed in."

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
    feats: Float["b n ..."] | Bool["b n"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> Float["b m ..."] | Bool["b m"]:  # type: ignore
    """
    Repeat a Tensor's values consecutively with the given lengths.

    :param feats: The features Tensor.
    :param lens: The lengths Tensor.
    :return: The repeated Tensor.
    """

    is_bool = feats.dtype == torch.bool
    feats = feats.float()

    device = feats.device

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens)

    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device=device)

    cumsum_len = lens.cumsum(dim=-1)
    offsets = F.pad(cumsum_len, (1, -1), value=0)
    indices = einx.add("w, b n -> b n w", arange, offsets)

    # create output tensor + a sink position on the very right (index max_len)

    total_lens = lens.sum(dim=-1)
    max_len = total_lens.amax()

    output = torch.zeros((batch, max_len + 1, *dims), device=device)

    indices.masked_fill_(~mask, max_len)  # scatter to sink position for padding
    indices = rearrange(indices, "b n w -> b (n w)")

    feats = repeat(feats, "b n ... -> b (n w) ...", w=window_size)

    # scatter

    output = einx.set_at("b [m] ...,  b nw, b nw ... -> b [m] ...", output, indices, feats)

    # remove sink

    output = output[:, :-1]

    if is_bool:
        output = output.bool()

    return output


def repeat_pairwise_consecutive_with_lens(
    feats: Float["b n n dp"],  # type: ignore
    lens: Int["b n"],  # type: ignore
) -> Float["b m m dp"]:  # type: ignore
    """
    Repeat a Tensor's pairwise values consecutively with the given lengths.

    :param feats: The features Tensor.
    :param lens: The lengths Tensor.
    :return: The repeated Tensor.
    """

    repeated_lens = repeat(lens, "b ... -> (b repeat) ...", repeat=feats.shape[1])
    feats, ps = pack_one(feats, "* n dp")
    feats = repeat_consecutive_with_lens(feats, repeated_lens)
    feats = unpack_one(feats, ps, "* n dp")

    feats = rearrange(feats, "b i j dp -> b j i dp")
    repeated_lens = repeat(lens, "b ... -> (b repeat) ...", repeat=feats.shape[1])
    feats, ps = pack_one(feats, "* n dp")
    feats = repeat_consecutive_with_lens(feats, repeated_lens)
    feats = unpack_one(feats, ps, "* n dp")
    feats = rearrange(feats, "b j i dp -> b i j dp")
    return feats
