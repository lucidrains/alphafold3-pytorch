from functools import wraps
from typing import Callable, List, Tuple, Union

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from torch import Tensor
from torch.nn import Module

from alphafold3_pytorch.tensor_typing import Bool, Float, Int, Shaped, typecheck
from alphafold3_pytorch.utils.utils import default, exists

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


@typecheck
def distance_to_bins(
    distance: Float["... dist"],  # type: ignore
    bins: Float[" bins"],  # type: ignore
) -> Int["... dist"]:  # type: ignore
    """Convert from distance to discrete bins, e.g., for `distance_labels`.

    :param distance: The distance tensor.
    :param bins: The bins tensor.
    :return: The discrete bins.
    """
    dist_from_dist_bins = einx.subtract(
        "... dist, dist_bins -> ... dist dist_bins", distance, bins
    ).abs()
    return dist_from_dist_bins.argmin(dim=-1)


@typecheck
def offset_only_positive(t: Tensor, offset: Tensor) -> Tensor:
    """Offset a Tensor only if it is positive."""
    is_positive = t >= 0
    t_offsetted = t + offset
    return torch.where(is_positive, t_offsetted, t)


def l2norm(t: Tensor, eps: float = 1e-20, dim: int = -1) -> Tensor:
    """Perform an L2 normalization on a Tensor.

    :param t: The Tensor.
    :param eps: The epsilon value.
    :param dim: The dimension to normalize over.
    :return: The L2 normalized Tensor.
    """
    return F.normalize(t, p=2, eps=eps, dim=dim)


def max_neg_value(t: Tensor) -> Tensor:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def log(t: Tensor, eps=1e-20) -> Tensor:
    """Run a safe log function that clamps the input to be above `eps` to avoid `log(0)`.

    :param t: The input tensor.
    :param eps: The epsilon value.
    :return: Tensor in the log domain.
    """
    return torch.log(t.clamp(min=eps))


def divisible_by(num: int, den: int) -> bool:
    """Check if a number is divisible by another number.

    :param num: The numerator.
    :param den: The denominator.
    :return: True if `num` is divisible by `den`, False otherwise.
    """
    return (num % den) == 0


def compact(*args):
    """Compact a tuple of objects by removing any `None` values.

    :param args: The objects to compact.
    :return: The compacted objects.
    """
    return tuple(filter(exists, args))


def pack_one(t: Tensor, pattern: str) -> Tuple[Tensor, List[Shape]]:
    """Pack a single tensor into a tuple of tensors with the given pattern.

    :param t: The tensor to pack.
    :param pattern: The pattern with which to pack.
    :return: The packed tensor along with the shape(s) of the tensor.
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        """Unpack a single tensor.

        :param to_unpack: The tensor to unpack.
        :param pattern: The pattern with which to unpack.
        :return: The unpacked tensor.
        """
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def softclamp(t: Tensor, value: float) -> Tensor:
    """Perform a soft clamp on a Tensor.

    :param t: The Tensor.
    :param value: The value to clamp to.
    :return: The soft clamped Tensor
    """
    return (t / value).tanh() * value


def exclusive_cumsum(t: Tensor, dim: int = -1) -> Tensor:
    """Perform an exclusive cumulative summation on a Tensor.

    :param t: The Tensor.
    :param dim: The dimension to sum over.
    :return: The exclusive cumulative sum Tensor.
    """
    return t.cumsum(dim=dim) - t


@typecheck
def symmetrize(t: Float["b n n ..."]) -> Float["b n n ..."]:  # type: ignore
    """Symmetrize a Tensor.

    :param t: The Tensor.
    :return: The symmetrized Tensor.
    """
    return t + rearrange(t, "b i j ... -> b j i ...")


# decorators


def maybe(fn):
    """Decorator to check if a Tensor exists before running a function on it."""

    @wraps(fn)
    def inner(t, *args, **kwargs):
        """Inner function to check if a Tensor exists before running a function on it."""
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
    """Slice a Tensor at a specific dimension.

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
    """Pad a Tensor to a specific length at a specific dimension.

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
    """Pad or slice a Tensor to a specific length at a specific dimension.

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
    """Pad a Tensor to a multiple of a specific number at a specific dimension.

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
    """Concatenate the previous window of a Tensor.

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
    """Pad and window a Tensor.

    :param t: The Tensor.
    :param window_size: The window size.
    :return: The padded and windowed Tensor.
    """
    t = pad_to_multiple(t, window_size, dim=1)
    t = rearrange(t, "b (n w) ... -> b n w ...", w=window_size)
    return t


# packed atom representation functions


@typecheck
def lens_to_mask(
    lens: Int["b ..."], max_len: int | None = None  # type: ignore
) -> Bool["... m"]:  # type: ignore
    """Convert a Tensor of lengths to a mask Tensor.

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
    """Perform mean pooling on a Tensor with the given lengths.

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

    # sel_cumsum = einx.get_at('b [m] d, b n -> b n d', cumsum_feats, cumsum_indices)

    cumsum_indices = repeat(cumsum_indices, "b n -> b n d", d=cumsum_feats.shape[-1])
    sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

    # subtract cumsum at one index from the previous one
    summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

    avg = einx.divide("b n d, b n", summed, lens.clamp(min=1))
    avg = einx.where("b n, b n d, -> b n d", mask, avg, 0.0)
    return avg


@typecheck
def mean_pool_fixed_windows_with_mask(
    feats: Float["b m d"],  # type: ignore
    mask: Bool["b m"],  # type: ignore
    window_size: int,
    return_mask_and_inverse: bool = False,
) -> Float["b n d"] | Tuple[Float["b n d"], Bool["b n"], Callable[[Float["b m d"]], Float["b n d"]]]:  # type: ignore
    """Mean pool a sequence of features with a fixed window size.

    :param feats: The features tensor.
    :param mask: The mask tensor.
    :param window_size: The window size.
    :param return_mask_and_inverse: Whether to return the pooled mask and the pooled inverse mask.
    :return: The pooled features tensor and optionally the pooled mask and the inverse function.
    """
    seq_len = feats.shape[-2]
    assert divisible_by(seq_len, window_size)

    feats = einx.where("b m, b m d, -> b m d", mask, feats, 0.0)

    num = reduce(feats, "b (n w) d -> b n d", "sum", w=window_size)
    den = reduce(mask.float(), "b (n w) -> b n 1", "sum", w=window_size)

    avg = num / den.clamp(min=1.0)

    if not return_mask_and_inverse:
        return avg

    pooled_mask = reduce(mask, "b (n w) -> b n", "any", w=window_size)

    @typecheck
    def inverse_fn(pooled: Float["b n d"]) -> Float["b m d"]:  # type: ignore
        """An inverse function to unpool the pooled features."""
        unpooled = repeat(pooled, "b n d -> b (n w) d", w=window_size)
        unpooled = einx.where("b m, b m d, -> b m d", mask, unpooled, 0.0)
        return unpooled

    return avg, pooled_mask, inverse_fn


@typecheck
def batch_repeat_interleave(
    feats: Float["b n ..."] | Bool["b n ..."] | Bool["b n"] | Int["b n"],  # type: ignore
    lens: Int["b n"],  # type: ignore
    output_padding_value: (
        float | int | bool | None
    ) = None,  # NOTE: this value determines what the output padding value will be
) -> Float["b m ..."] | Bool["b m ..."] | Bool["b m"] | Int["b m"]:  # type: ignore
    """Batch repeat and interleave a sequence of features.

    :param feats: The features tensor.
    :param lens: The lengths tensor.
    :param output_padding_value: The output padding value.
    :return: The batch repeated and interleaved features tensor.
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

    total_lens = lens.clamp(min=0).sum(dim=-1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

    indices = indices.masked_fill(~mask, max_len)  # scatter to sink position for padding
    indices = rearrange(indices, "b n w -> b (n w)")

    # scatter

    seq_arange = torch.arange(seq, device=device)
    seq_arange = repeat(seq_arange, "n -> b (n w)", b=batch, w=window_size)

    # output_indices = einx.set_at('b [m], b nw, b nw -> b [m]', output_indices, indices, seq_arange)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink

    output_indices = output_indices[:, :-1]

    # gather

    # output = einx.get_at('b [n] ..., b m -> b m ...', feats, output_indices)

    feats, unpack_one = pack_one(feats, "b n *")
    output_indices = repeat(output_indices, "b m -> b m d", d=feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value

    output_padding_value = default(output_padding_value, False if dtype == torch.bool else 0)

    output = einx.where("b n, b n ..., -> b n ...", output_mask, output, output_padding_value)

    return output


@typecheck
def batch_repeat_interleave_pairwise(
    pairwise: Float["b n n d"],  # type: ignore
    molecule_atom_lens: Int["b n"],  # type: ignore
) -> Float["b m m d"]:  # type: ignore
    """Batch repeat and interleave a sequence of pairwise features."""
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)

    molecule_atom_lens = repeat(molecule_atom_lens, "b ... -> (b r) ...", r=pairwise.shape[1])
    pairwise, unpack_one = pack_one(pairwise, "* n d")
    pairwise = batch_repeat_interleave(pairwise, molecule_atom_lens)
    return unpack_one(pairwise)


@typecheck
def to_pairwise_mask(
    mask_i: Bool["... n"],  # type: ignore
    mask_j: Bool["... n"] | None = None,  # type: ignore
) -> Bool["... n n"]:  # type: ignore
    """Convert two masks into a pairwise mask.

    :param mask_i: The first mask.
    :param mask_j: The second mask.
    :return: The pairwise mask.
    """
    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and("... i, ... j -> ... i j", mask_i, mask_j)


@typecheck
def masked_average(
    t: Shaped["..."],  # type: ignore
    mask: Shaped["..."],  # type: ignore
    *,
    dim: int | Tuple[int, ...],
    eps=1.0,
) -> Float["..."]:  # type: ignore
    """Compute the masked average of a Tensor.

    :param t: The Tensor.
    :param mask: The mask.
    :param dim: The dimension(s) to average over.
    :param eps: The epsilon value.
    :return: The masked average.
    """
    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim)
    return num / den.clamp(min=eps)


@typecheck
def calculate_weighted_rigid_align_weights(
    atom_pos_ground_truth: Float["b m 3"],  # type: ignore
    molecule_atom_lens: Int["b n"],  # type: ignore
    is_molecule_types: Bool["b n ..."] | None = None,  # type: ignore
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
) -> Float["b m"]:  # type: ignore
    """Calculate the weighted rigid alignment weights.

    :param atom_pos_ground_truth: The ground truth atom positions.
    :param molecule_atom_lens: The molecule atom lengths.
    :param is_molecule_types: The molecule types.
    :param nucleotide_loss_weight: The nucleotide loss weight.
    :param ligand_loss_weight: The ligand loss weight.
    :return: The weighted rigid alignment weights.
    """

    # if additional molecule feats is provided
    # calculate the weights for mse loss (wl)

    align_weights = atom_pos_ground_truth.new_ones(atom_pos_ground_truth.shape[:2])

    if exists(is_molecule_types):
        is_nucleotide_or_ligand_fields = is_molecule_types.unbind(dim=-1)

        is_nucleotide_or_ligand_fields = tuple(
            batch_repeat_interleave(t, molecule_atom_lens) for t in is_nucleotide_or_ligand_fields
        )
        is_nucleotide_or_ligand_fields = tuple(
            pad_or_slice_to(t, length=align_weights.shape[-1], dim=-1)
            for t in is_nucleotide_or_ligand_fields
        )

        _, atom_is_dna, atom_is_rna, atom_is_ligand, _ = is_nucleotide_or_ligand_fields

        # section 3.7.1 equation 4

        # upweighting of nucleotide and ligand atoms is additive per equation 4

        align_weights = torch.where(
            atom_is_dna | atom_is_rna,
            1 + nucleotide_loss_weight,
            align_weights,
        )
        align_weights = torch.where(atom_is_ligand, 1 + ligand_loss_weight, align_weights)

    return align_weights


# checkpointing utils


@typecheck
def should_checkpoint(
    self: Module,
    inputs: Tensor | Tuple[Tensor, ...],
    check_instance_variable: str | None = "checkpoint",
) -> bool:
    """Determine if activation checkpointing should be used.

    :param self: The module.
    :param inputs: The inputs.
    :param check_instance_variable: The instance variable to check.
    :return: True if activation checkpointing should be used, False otherwise.
    """
    if torch.is_tensor(inputs):
        inputs = (inputs,)

    return (
        self.training
        and any([i.requires_grad for i in inputs])
        and (not exists(check_instance_variable) or getattr(self, check_instance_variable, False))
    )
