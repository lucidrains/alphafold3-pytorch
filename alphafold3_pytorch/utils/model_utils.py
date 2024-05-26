from typing import Tuple

import rootutils
import torch
import torch.nn.functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from alphafold3_pytorch.utils.typing import typecheck


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


def max_neg_value(t: torch.Tensor) -> torch.Tensor:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


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
