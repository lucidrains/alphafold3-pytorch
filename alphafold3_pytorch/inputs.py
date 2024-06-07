from typing import TypedDict

from alphafold3_pytorch.typing import (
    typecheck,
    Int, Bool, Float
)

# constants

@typecheck
class AtomInput(TypedDict):
    atom_inputs:                Float['m dai']
    molecule_atom_lens:         Int[' n']
    atompair_inputs:            Float['m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Float['n 10']
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    template_mask:              Bool[' t'] | None
    msa_mask:                   Bool[' s'] | None
    atom_pos:                   Float['m 3'] | None
    molecule_atom_indices:      Int[' n'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int[' n'] | None
    resolved_labels:            Int[' n'] | None
