from typing import NamedTuple, TypedDict

from alphafold3_pytorch.utils.typing import Bool, Float, Int, typecheck


@typecheck
class Alphafold3Input(TypedDict):
    """A collection of inputs to AlphaFold 3."""

    atom_inputs: Float["m dai"]  # type: ignore
    residue_atom_lens: Int["n 2"]  # type: ignore
    atompair_feats: Float["m m dap"]  # type: ignore
    additional_residue_feats: Float["n 10"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    residue_atom_indices: Int["n"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


class AttentionConfig(NamedTuple):
    """Configuration for an attention mechanism."""

    enable_flash: bool
    enable_math: bool
    enable_mem_efficient: bool
