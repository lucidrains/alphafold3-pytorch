import torch

from alphafold3_pytorch import (
    Alphafold3Input,
    AtomInput,
    maybe_transform_to_atom_input
)

from alphafold3_pytorch.life import (
    reverse_complement,
    reverse_complement_tensor
)

def test_string_reverse_complement():
    assert reverse_complement('ATCG') == 'CGAT'
    assert reverse_complement('AUCG', 'rna') == 'CGAU'

def test_tensor_reverse_complement():
    seq = torch.randint(0, 5, (100,))
    rc = reverse_complement_tensor(seq)
    assert torch.allclose(reverse_complement_tensor(rc), seq)
