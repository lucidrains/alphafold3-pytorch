import importlib
from typing import Any

from omegaconf import OmegaConf

from alphafold3_pytorch.models.alphafold3_module import AlphaFold3Input, AlphaFold3LitModule
from alphafold3_pytorch.models.components.alphafold3 import (
    AdaptiveLayerNorm,
    AlphaFold3,
    AttentionPairBias,
    CentreRandomAugmentation,
    ComputeAlignmentError,
    ConditionWrapper,
    ConfidenceHead,
    DiffusionModule,
    DiffusionTransformer,
    DistogramHead,
    ElucidatedAtomDiffusion,
    ExpressCoordinatesInFrame,
    InputFeatureEmbedder,
    MSAModule,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairformerStack,
    PreLayerNorm,
    RelativePositionEncoding,
    SmoothLDDTLoss,
    TemplateEmbedder,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
    WeightedRigidAlign,
)
from alphafold3_pytorch.models.components.attention import Attend, Attention

__all__ = [
    Attention,
    Attend,
    RelativePositionEncoding,
    SmoothLDDTLoss,
    WeightedRigidAlign,
    ExpressCoordinatesInFrame,
    ComputeAlignmentError,
    CentreRandomAugmentation,
    TemplateEmbedder,
    PreLayerNorm,
    AdaptiveLayerNorm,
    ConditionWrapper,
    OuterProductMean,
    MSAPairWeightedAveraging,
    TriangleMultiplication,
    AttentionPairBias,
    TriangleAttention,
    Transition,
    MSAModule,
    PairformerStack,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    InputFeatureEmbedder,
    ConfidenceHead,
    DistogramHead,
    AlphaFold3,
    AlphaFold3LitModule,
    AlphaFold3Input,
]


def resolve_omegaconf_variable(variable_path: str) -> Any:
    """Resolve an OmegaConf variable path to its value."""
    try:
        # split the string into parts using the dot separator
        parts = variable_path.rsplit(".", 1)

        # get the module name from the first part of the path
        module_name = parts[0]

        # dynamically import the module using the module name
        try:
            module = importlib.import_module(module_name)
            # use the imported module to get the requested attribute value
            attribute = getattr(module, parts[1])
        except Exception:
            module = importlib.import_module(".".join(module_name.split(".")[:-1]))
            inner_module = ".".join(module_name.split(".")[-1:])
            # use the imported module to get the requested attribute value
            attribute = getattr(getattr(module, inner_module), parts[1])

    except Exception as e:
        raise ValueError(
            f"Error: {variable_path} is not a valid path to a Python variable within the project."
        ) from e

    return attribute


def int_divide(x: int, y: int) -> int:
    """
    Perform integer division on `x` and `y`.

    :param x: The dividend.
    :param y: The divisor.
    :return: The integer division result.
    """
    if x % y == 0:
        return x // y
    else:
        raise ValueError(
            f"Error: {x} is not divisible by {y} in the call to OmegaConf's `int_divide` function."
        )


def validate_gradient_accumulation_factor(batch_size: int, world_size: int) -> int:
    """
    Validate the gradient accumulation factor. If the factor is valid, return `world_size`.

    :param batch_size: The batch size.
    :param world_size: The number of GPUs.
    :return: The validated gradient accumulation factor.
    """
    if batch_size % world_size == 0:
        return world_size
    else:
        raise ValueError(
            f"Error: For gradient accumulation, the batch size ({batch_size}) must be divisible by the distributed device world size ({world_size})."
        )


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver(
        "resolve_variable", lambda variable_path: resolve_omegaconf_variable(variable_path)
    )
    OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
    OmegaConf.register_new_resolver("subtract", lambda x, y: int(x) - int(y))
    OmegaConf.register_new_resolver("multiply", lambda x, y: int(x) * int(y))
    OmegaConf.register_new_resolver("divide", lambda x, y: int(x) / int(y))
    OmegaConf.register_new_resolver("int_divide", lambda x, y: int_divide(int(x), int(y)))
    OmegaConf.register_new_resolver(
        "validate_gradient_accumulation_factor",
        lambda batch_size, world_size: validate_gradient_accumulation_factor(
            int(batch_size), int(world_size)
        ),
    )
