import torch

import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol

from typing import Literal
from alphafold3_pytorch.tensor_typing import (
    Int,
    typecheck
)

def is_unique(arr):
    return len(arr) == len({*arr})

# human amino acids

HUMAN_AMINO_ACIDS = dict(
    A = dict(
        smile = 'CC(C(=O)O)N',
        hydroxyl_idx = 4,
        distogram_atom_idx = 0
    ),
    R = dict(
        smile = 'C(CC(C(=O)O)N)CN=C(N)N',
        hydroxyl_idx = 5,
        distogram_atom_idx = 0
    ),
    N = dict(
        smile = 'C(C(C(=O)O)N)C(=O)N',
        hydroxyl_idx = 4,
        distogram_atom_idx = 0
    ),
    D = dict(
        smile = 'C(C(C(=O)O)N)C(=O)O',
        hydroxyl_idx = 8,
        distogram_atom_idx = 0
    ),
    C = dict(
        smile = 'C(C(C(=O)O)N)S',
        hydroxyl_idx = 4,
        distogram_atom_idx = 0
    ),
    Q = dict(
        smile = 'C(CC(=O)N)C(C(=O)O)N',
        hydroxyl_idx = 8,
        distogram_atom_idx = 0
    ),
    E = dict(
        smile = 'C(CC(=O)O)C(C(=O)O)N',
        hydroxyl_idx = 8,
        distogram_atom_idx = 0
    ),
    G = dict(
        smile = 'C(C(=O)O)N',
        hydroxyl_idx = 3,
        distogram_atom_idx = 0
    ),
    H = dict(
        smile = 'NC(C(=O)O)Cc1c[nH]cn1',
        hydroxyl_idx = 4,
        distogram_atom_idx = 0
    ),
    I = dict(
        smile = 'CCC(C)C(C(=O)O)N',
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    L = dict(
        smile = 'CC(C)CC(C(=O)O)N',
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    K = dict(
        smile = 'C(CCN)CC(C(=O)O)N',
        hydroxyl_idx = 8,
        distogram_atom_idx = 0
    ),
    M = dict(
        smile = 'CSCCC(C(=O)O)N',
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    F = dict(
        smile = 'C1=CC=C(C=C1)CC(C(=O)O)N',
        hydroxyl_idx = 10,
        distogram_atom_idx = 0
    ),
    P = dict(
        smile = 'C1CC(NC1)C(=O)O',
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    S = dict(
        smile = 'C(C(C(=O)O)N)O',
        hydroxyl_idx = 4,
        distogram_atom_idx = 0
    ),
    T = dict(
        smile = 'CC(C(C(=O)O)N)O',
        hydroxyl_idx = 5,
        distogram_atom_idx = 0
    ),
    W = dict(
        smile = 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
        hydroxyl_idx = 13,
        distogram_atom_idx = 0
    ),
    Y = dict(
        smile = 'C1=CC(=CC=C1CC(C(=O)O)N)O',
        hydroxyl_idx = 10,
        distogram_atom_idx = 0
    ),
    V = dict(
        smile = 'CC(C)C(C(=O)O)N',
        hydroxyl_idx = 6,
        distogram_atom_idx = 0
    )
)

# nucleotides

DNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1C(C(OC1N2C=NC3=C(N=CN=C32)N)COP(=O)(O)O)O',
        complement = 'T',
        hydroxyl_idx = 20,
    ),
    C = dict(
        smile = 'C1C(C(OC1N2C=CC(=NC2=O)N)COP(=O)(O)O)O',
        complement = 'G',
        hydroxyl_idx = 17
    ),
    G = dict(
        smile = 'C1C(C(OC1N2C=NC3=C2N=C(NC3=O)N)COP(=O)(O)O)O',
        complement = 'C',
        hydroxyl_idx= 21
    ),
    T = dict(
        smile = 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)O)O',
        complement = 'A',
        hydroxyl_idx = 19
    )
)

RNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O)N',
        complement = 'U',
        hydroxyl_idx = 19
    ),
    C = dict(
        smile = 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)COP(=O)([O-])[O-])O)O',
        complement = 'G',
        hydroxyl_idx = 17
    ),
    G = dict(
        smile = 'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)O)O)O)N=C(NC2=O)N',
        complement = 'C',
        hydroxyl_idx = 14
    ),
    U = dict(
        smile = 'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)(O)O)O)O',
        complement = 'A',
        hydroxyl_idx = 18
    )
)

# complements in tensor form, following the ordering ACG(T|U)N

NUCLEIC_ACID_COMPLEMENT_TENSOR = torch.tensor([3, 2, 1, 0, 4], dtype = torch.long)

# some functions for nucleic acids

@typecheck
def reverse_complement(
    seq: str,
    nucleic_acid_type: Literal['dna', 'rna'] = 'dna'
):
    if nucleic_acid_type == 'dna':
        nucleic_acid_entries = DNA_NUCLEOTIDES
    elif nucleic_acid_type == 'rna':
        nucleic_acid_entries = RNA_NUCLEOTIDES

    assert all([nuc in nucleic_acid_entries for nuc in seq]), 'unknown nucleotide for given nucleic acid type'

    complement = [nucleic_acid_entries[nuc]['complement'] for nuc in seq]
    return ''.join(complement[::-1])

@typecheck
def reverse_complement_tensor(t: Int['n']):
    complement = NUCLEIC_ACID_COMPLEMENT_TENSOR[t]
    reverse_complement = t.flip(dims = (-1,))
    return reverse_complement

# metal ions

METALS = dict(
    Mg = dict(
        smile = '[Mg]'
    ),
    Mn = dict(
        smile = '[Mn]'
    ),
    Fe = dict(
        smile = '[Fe]'
    ),
    Co = dict(
        smile = '[Co]'
    ),
    Ni = dict(
        smile = '[Ni]'
    ),
    Cu = dict(
        smile = '[Cu]'
    ),
    Zn = dict(
        smile = '[Zn]'
    ),
    Na = dict(
        smile = '[Na]'
    ),
    Cl = dict(
        smile = '[Cl]'
    )
)

# miscellaneous

MISC = dict(
    Phospholipid = dict(
        smile = 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)O)OC(=O)CCCCCCCC1CC1CCCCCC'
    )
)

# atoms - for atom embeddings

ATOMS = [
    'C',
    'O',
    'N',
    'S',
    'P',
    *METALS
]

assert is_unique(ATOMS)

# bonds for atom bond embeddings

ATOM_BONDS = [
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC'
]

assert is_unique(ATOM_BONDS)

# some rdkit helper function

@typecheck
def generate_conformation(mol: Mol) -> Mol:
    mol = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs = 1)
    mol = Chem.RemoveHs(mol)
    return mol

def mol_from_smile(smile: str) -> Mol:
    mol = Chem.MolFromSmiles(smile)
    return generate_conformation(mol)

def remove_atom_from_mol(mol: Mol, atom_idx: int) -> Mol:
    edit_mol = Chem.EditableMol(mol)
    edit_mol.RemoveAtom(atom_idx)
    return mol

# initialize rdkit.Chem with canonical SMILES

ALL_ENTRIES = [
    *HUMAN_AMINO_ACIDS.values(),
    *DNA_NUCLEOTIDES.values(),
    *RNA_NUCLEOTIDES.values(),
    *METALS.values(),
    *MISC.values(),
]

for entry in ALL_ENTRIES:
    mol = mol_from_smile(entry['smile'])
    entry['rdchem_mol'] = mol
