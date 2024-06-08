from typing import TypedDict

from alphafold3_pytorch.typing import (
    typecheck,
    Int, Bool, Float
)

# constants

@typecheck
class AtomInput(TypedDict):
    atom_inputs:                Float['*b m dai']
    molecule_atom_lens:         Int['*b n']
    atompair_inputs:            Float['*b m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Float['*b n 10']
    templates:                  Float['*b t n n dt']
    msa:                        Float['*b s n dm']
    atom_ids:                   Int['*b m'] | None
    atompair_ids:               Int['*b m m'] | Int['nw w (w*2)'] | None
    template_mask:              Bool['*b t'] | None
    msa_mask:                   Bool['*b s'] | None
    atom_pos:                   Float['*b m 3'] | None
    molecule_atom_indices:      Int['*b n'] | None
    distance_labels:            Int['*b n n'] | None
    pae_labels:                 Int['*b n n'] | None
    pde_labels:                 Int['*b n'] | None
    resolved_labels:            Int['*b n'] | None
