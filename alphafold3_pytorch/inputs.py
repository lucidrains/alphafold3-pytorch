from typing import Type, TypedDict, Literal, Callable

from alphafold3_pytorch.typing import (
    typecheck,
    Int, Bool, Float
)

# atom level, what Alphafold3 accepts

@typecheck
class AtomInput(TypedDict):
    atom_inputs:                Float['*b m dai']
    molecule_ids:               Int['*b n']
    molecule_atom_lens:         Int['*b n']
    atompair_inputs:            Float['*b m m dapi'] | Float['*b nw w (w*2) dapi']
    additional_molecule_feats:  Float['*b n 9']
    templates:                  Float['*b t n n dt']
    msa:                        Float['*b s n dm']
    atom_ids:                   Int['*b m'] | None
    atompair_ids:               Int['*b m m'] | Int['*b nw w (w*2)'] | None
    template_mask:              Bool['*b t'] | None
    msa_mask:                   Bool['*b s'] | None
    atom_pos:                   Float['*b m 3'] | None
    molecule_atom_indices:      Int['*b n'] | None
    distance_labels:            Int['*b n n'] | None
    pae_labels:                 Int['*b n n'] | None
    pde_labels:                 Int['*b n'] | None
    resolved_labels:            Int['*b n'] | None

# residue level - single chain proteins for starters

@typecheck
class SingleProteinInput(TypedDict):
    residue_ids:                Int['*b n']
    residue_atom_lens:          Int['*b n']
    templates:                  Float['*b t n n dt']
    msa:                        Float['*b s n dm']
    template_mask:              Bool['*b t'] | None
    msa_mask:                   Bool['*b s'] | None
    atom_pos:                   Float['*b m 3'] | None
    distance_labels:            Int['*b n n'] | None
    pae_labels:                 Int['*b n n'] | None
    pde_labels:                 Int['*b n'] | None
    resolved_labels:            Int['*b n'] | None

@typecheck
def single_protein_input_to_atom_input(
    input: SingleProteinInput
) -> AtomInput:

    raise NotImplementedError

# single chain protein with single ds nucleic acid

# o - for nucleOtide seq

@typecheck
class SingleProteinSingleNucleicAcidInput(TypedDict):
    residue_ids:                Int['*b n']
    residue_atom_lens:          Int['*b n']
    nucleotide_ids:             Int['*b o']
    nucleic_acid_type:          Literal['dna', 'rna']
    templates:                  Float['*b t n n dt']
    msa:                        Float['*b s n dm']
    template_mask:              Bool['*b t'] | None
    msa_mask:                   Bool['*b s'] | None
    atom_pos:                   Float['*b m 3'] | None
    distance_labels:            Int['*b n n'] | None
    pae_labels:                 Int['*b n n'] | None
    pde_labels:                 Int['*b n'] | None
    resolved_labels:            Int['*b n'] | None

@typecheck
def single_protein_input_and_single_nucleic_acid_to_atom_input(
    input: SingleProteinSingleNucleicAcidInput
) -> AtomInput:

    raise NotImplementedError

# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    SingleProteinInput: single_protein_input_to_atom_input,
    SingleProteinSingleNucleicAcidInput: single_protein_input_and_single_nucleic_acid_to_atom_input
}

# function for extending the config

@typecheck
def register_input_transform(
    input_type: Type,
    fn: Callable[[TypedDict], AtomInput]
):
    INPUT_TO_ATOM_TRANSFORM[input_type] = fn
