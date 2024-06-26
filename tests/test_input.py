import torch

from alphafold3_pytorch import (
    Alphafold3Input,
    AtomInput,
    maybe_transform_to_atom_input
)

from alphafold3_pytorch.life import (
    reverse_complement
)

def test_life():
    assert reverse_complement('ATCG') == 'CGAT'
    assert reverse_complement('AUCG', 'rna') == 'CGAU'
