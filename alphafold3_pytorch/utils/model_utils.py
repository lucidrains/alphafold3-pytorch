from functools import wraps
from beartype.typing import Callable, List, Tuple, Union

import einx
import torch
import torch.nn.functional as F
from einops import einsum, pack, rearrange, reduce, repeat, unpack
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
def distance_to_dgram(
    distance: Float["... dist"],  # type: ignore
    bins: Float[" bins"],  # type: ignore
    return_labels: bool = False,
) -> Int["... dist"] | Int["... dist bins"]:  # type: ignore
    """Converting from distance to discrete bins, e.g., for distance_labels and pae_labels using
    the same logic as OpenFold.

    :param distance: The distance tensor.
    :param bins: The bins tensor.
    :param return_labels: Whether to return the labels.
    :return: The one-hot bins tensor or the bin labels.
    """

    distance = distance.abs()

    bins = F.pad(bins, (0, 1), value = float('inf'))
    low, high = bins[:-1], bins[1:]

    one_hot = (
        einx.greater_equal("..., bin_low -> ... bin_low", distance, low)
        & einx.less("..., bin_high -> ... bin_high", distance, high)
    ).long()

    if return_labels:
        return one_hot.argmax(dim=-1)

    return one_hot


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
def remove_consecutive_duplicate(
    t: Int["n ..."], remove_to_value: int = -1  # type: ignore
) -> Int["n ..."]:  # type: ignore
    """Remove consecutive duplicates from a Tensor.

    :param t: The Tensor.
    :param remove_to_value: The value to remove to.
    :return: The Tensor with consecutive duplicates removed.
    """
    is_duplicate = t[1:] == t[:-1]

    if is_duplicate.ndim == 2:
        is_duplicate = is_duplicate.all(dim=-1)

    is_duplicate = F.pad(is_duplicate, (1, 0), value=False)
    return einx.where("n, n ..., -> n ... ", ~is_duplicate, t, remove_to_value)


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


# functions for deriving the frames for ligands
# this follows the logic from Alphafold3 Supplementary section 4.3.2


@typecheck
def get_indices_three_closest_atom_pos(
    atom_pos: Float["... n d"],  # type: ignore
    mask: Bool["... n"] | None = None,  # type: ignore
) -> Int["... n 3"]:  # type: ignore
    """Get the indices of the three closest atoms to each atom.

    :param atom_pos: The atom positions.
    :param mask: The mask to apply.
    :return: The indices of the three closest atoms to each atom.
    """
    atom_dims, device = atom_pos.shape[-3:-1], atom_pos.device
    num_atoms, has_batch = atom_pos.shape[-2], atom_pos.ndim == 3
    batch_size = 1 if not has_batch else atom_pos.shape[0]

    if num_atoms < 3:
        return atom_pos.new_full((*atom_dims, 3), -1).long()

    if not has_batch:
        atom_pos = rearrange(atom_pos, "... -> 1 ...")

        if exists(mask):
            mask = rearrange(mask, "... -> 1 ...")

    # figure out which set of atoms are less than 3 for masking out later

    if exists(mask):
        insufficient_atom_mask = mask.sum(dim=-1, keepdim=True) < 3

    # get distances between all atoms

    atom_dist = torch.cdist(atom_pos, atom_pos)

    # mask out the distance to self

    eye = torch.eye(num_atoms, device=device, dtype=torch.bool)

    mask_value = 1e4
    atom_dist.masked_fill_(eye, mask_value)

    # take care of padding

    if exists(mask):
        pair_mask = einx.logical_and("... i, ... j -> ... i j", mask, mask)
        atom_dist.masked_fill_(~pair_mask, mask_value)

    # will use topk on the negative of the distance

    _, two_closest_atom_indices = (-atom_dist).topk(2, dim=-1)

    # place each atom at the center of its frame

    three_atom_indices, _ = pack(
        (
            two_closest_atom_indices[..., 0],
            torch.arange(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1),
            two_closest_atom_indices[..., 1],
        ),
        "b n *",
    )

    # mask out

    if exists(mask):
        three_atom_indices = torch.where(
            ~insufficient_atom_mask.unsqueeze(-1), three_atom_indices, -1
        )

    if not has_batch:
        three_atom_indices = rearrange(three_atom_indices, "1 ... -> ...")

    return three_atom_indices


@typecheck
def get_angle_between_edges(
    edge1: Float["... n 3"],  # type: ignore
    edge2: Float["... n 3"],  # type: ignore
) -> Float["... n"]:  # type: ignore
    """Get the angles between two edges for each node.

    :param edge1: The first edge.
    :param edge2: The second edge.
    :return: The angles between the two edges for each node.
    """
    cos = (l2norm(edge1) * l2norm(edge2)).sum(-1)
    return torch.acos(cos)


@typecheck
def get_frames_from_atom_pos(
    atom_pos: Float["... n d"],  # type: ignore
    mask: Bool["... n"] | None = None,  # type: ignore
    filter_colinear_pos: bool = False,
    is_colinear_angle_thres: float = 25.0,  # NOTE: DM uses 25 degrees as a way of filtering out invalid frames
) -> Int["... n 3"]:  # type: ignore
    """Get the nearest neighbor frames for all atom positions.

    :param atom_pos: The atom positions.
    :param filter_colinear_pos: Whether to filter colinear positions.
    :param is_colinear_angle_thres: The colinear angle threshold.
    :return: The frames for all atoms.
    """
    frames = get_indices_three_closest_atom_pos(atom_pos, mask=mask)

    if not filter_colinear_pos:
        return frames

    is_invalid = (frames == -1).any(dim=-1)

    # get the edges and derive angles

    three_atom_pos = torch.cat(
        [
            einx.get_at("... [m] c, ... three -> ... three c", atom_pos, frame).unsqueeze(-3)
            for frame in frames.unbind(dim=-2)
        ],
        dim=-3,
    )

    left_pos, center_pos, right_pos = three_atom_pos.unbind(dim=-2)

    edges1, edges2 = (left_pos - center_pos), (right_pos - center_pos)

    angle = get_angle_between_edges(edges1, edges2)

    degree = torch.rad2deg(angle)

    is_colinear = (degree.abs() < is_colinear_angle_thres) | (
        (180.0 - degree.abs()).abs() < is_colinear_angle_thres
    )

    # set any three atoms that are colinear to -1 indices

    three_atom_indices = einx.where(
        "..., ... three, -> ... three", ~(is_colinear | is_invalid), frames, -1
    )
    return three_atom_indices


# modules for handling frames


class ExpressCoordinatesInFrame(Module):
    """Algorithm 29."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    @typecheck
    def forward(
        self,
        coords: Float["b m 3"],  # type: ignore
        frame: Float["b m 3 3"] | Float["b 3 3"] | Float["3 3"],  # type: ignore
    ) -> Float["b m 3"]:  # type: ignore
        """Express coordinates in the given frame.

        :param coords: Coordinates to be expressed in the given frame.
        :param frame: Frames defined by three points.
        :return: The transformed coordinates.
        """

        if frame.ndim == 2:
            frame = rearrange(frame, "fr fc -> 1 1 fr fc")
        elif frame.ndim == 3:
            frame = rearrange(frame, "b fr fc -> b 1 fr fc")

        # Extract frame atoms
        a, b, c = frame.unbind(dim=-1)
        w1 = l2norm(a - b, eps=self.eps)
        w2 = l2norm(c - b, eps=self.eps)

        # Build orthonormal basis
        e1 = l2norm(w1 + w2, eps=self.eps)
        e2 = l2norm(w2 - w1, eps=self.eps)
        e3 = torch.cross(e1, e2, dim=-1)

        # Project onto frame basis
        d = coords - b

        transformed_coords = torch.stack(
            (
                einsum(d, e1, "... i, ... i -> ..."),
                einsum(d, e2, "... i, ... i -> ..."),
                einsum(d, e3, "... i, ... i -> ..."),
            ),
            dim=-1,
        )

        return transformed_coords


class RigidFrom3Points(Module):
    """An implementation of Algorithm 21 in Section 1.8.1 in AlphaFold 2 paper:

    https://www.nature.com/articles/s41586-021-03819-2
    """

    @typecheck
    def forward(
        self,
        three_points: Tuple[Float["... 3"], Float["... 3"], Float["... 3"]] | Float["3 ... 3"],  # type: ignore
    ) -> Tuple[Float["... 3 3"], Float["... 3"]]:  # type: ignore
        """Compute a rigid transformation from three points."""
        if isinstance(three_points, tuple):
            three_points = torch.stack(three_points)

        # allow for any number of leading dimensions

        (x1, x2, x3), unpack_one = pack_one(three_points, "three * d")

        # main algorithm

        v1 = x3 - x2
        v2 = x1 - x2

        e1 = l2norm(v1)
        u2 = v2 - e1 @ (e1.t() @ v2)
        e2 = l2norm(u2)

        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack((e1, e2, e3), dim=-1)
        t = x2

        # unpack

        R = unpack_one(R, "* r1 r2")
        t = unpack_one(t, "* c")

        return R, t


class RigidFromReference3Points(Module):
    """A modification of Algorithm 21 in Section 1.8.1 in AlphaFold 2 paper:

    https://www.nature.com/articles/s41586-021-03819-2

    Inpsired by the implementation in the OpenFold codebase:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/utils/feats.py#L143
    """

    @typecheck
    def forward(
        self,
        three_points: Tuple[Float["... 3"], Float["... 3"], Float["... 3"]] | Float["3 ... 3"],  # type: ignore
        eps: float = 1e-20,
    ) -> Tuple[Float["... 3 3"], Float["... 3"]]:  # type: ignore
        """Return a transformation object from reference coordinates.

        NOTE: This method does not take care of symmetries. If you
        provide the atom positions in the non-standard way,
        e.g., the N atom of amino acid residues will end up
        not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care
        of such cases in your code.

        :param three_points: Three reference points to define the transformation.
        :param eps: A small value to avoid division by zero.
        :return: A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        if isinstance(three_points, tuple):
            three_points = torch.stack(three_points)

        # allow for any number of leading dimensions

        (x1, x2, x3), unpack_one = pack_one(three_points, "three * d")

        # main algorithm

        t = -1 * x2
        x1 = x1 + t
        x3 = x3 + t

        x3_x, x3_y, x3_z = [x3[..., i] for i in range(3)]
        norm = torch.sqrt(eps + x3_x**2 + x3_y**2)
        sin_x3_1 = -x3_y / norm
        cos_x3_1 = x3_x / norm

        x3_1_R = sin_x3_1.new_zeros((*sin_x3_1.shape, 3, 3))
        x3_1_R[..., 0, 0] = cos_x3_1
        x3_1_R[..., 0, 1] = -1 * sin_x3_1
        x3_1_R[..., 1, 0] = sin_x3_1
        x3_1_R[..., 1, 1] = cos_x3_1
        x3_1_R[..., 2, 2] = 1

        norm = torch.sqrt(eps + x3_x**2 + x3_y**2 + x3_z**2)
        sin_x3_2 = x3_z / norm
        cos_x3_2 = torch.sqrt(x3_x**2 + x3_y**2) / norm

        x3_2_R = sin_x3_2.new_zeros((*sin_x3_2.shape, 3, 3))
        x3_2_R[..., 0, 0] = cos_x3_2
        x3_2_R[..., 0, 2] = sin_x3_2
        x3_2_R[..., 1, 1] = 1
        x3_2_R[..., 2, 0] = -1 * sin_x3_2
        x3_2_R[..., 2, 2] = cos_x3_2

        x3_R = einsum(x3_2_R, x3_1_R, "n i j, n j k -> n i k")
        x1 = einsum(x3_R, x1, "n i j, n j -> n i")

        _, x1_y, x1_z = [x1[..., i] for i in range(3)]
        norm = torch.sqrt(eps + x1_y**2 + x1_z**2)
        sin_x1 = -x1_z / norm
        cos_x1 = x1_y / norm

        x1_R = sin_x3_2.new_zeros((*sin_x3_2.shape, 3, 3))
        x1_R[..., 0, 0] = 1
        x1_R[..., 1, 1] = cos_x1
        x1_R[..., 1, 2] = -1 * sin_x1
        x1_R[..., 2, 1] = sin_x1
        x1_R[..., 2, 2] = cos_x1

        R = einsum(x1_R, x3_R, "n i j, n j k -> n i k")

        R = R.transpose(-1, -2)
        t = -1 * t

        # unpack

        R = unpack_one(R, "* r1 r2")
        t = unpack_one(t, "* c")

        return R, t
