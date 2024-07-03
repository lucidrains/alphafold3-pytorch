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
# reordered so [N][...][C][OH] - [OH] is removed for all peptides except last

HUMAN_AMINO_ACIDS = dict(
    A = dict(
        smile = 'CC(C(=O)O)N',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    R = dict(
        smile = 'C(CC(C(=O)O)N)CN=C(N)N',
        first_atom_idx = 6,
        last_atom_idx = 3,
        hydroxyl_idx = 5,
        distogram_atom_idx = 2
    ),
    N = dict(
        smile = 'C(C(C(=O)O)N)C(=O)N',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    D = dict(
        smile = 'C(C(C(=O)O)N)C(=O)O',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 8,
        distogram_atom_idx = 1
    ),
    C = dict(
        smile = 'C(C(C(=O)O)N)S',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    Q = dict(
        smile = 'C(CC(=O)N)C(C(=O)O)N',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    E = dict(
        smile = 'C(CC(=O)O)C(C(=O)O)N',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    G = dict(
        smile = 'C(C(=O)O)N',
        first_atom_idx = 4,
        last_atom_idx = 1,
        hydroxyl_idx = 3,
        distogram_atom_idx = 0
    ),
    H = dict(
        smile = 'C1=C(NC=N1)CC(C(=O)O)N',
        first_atom_idx = 10,
        last_atom_idx = 7,
        hydroxyl_idx = 9,
        distogram_atom_idx = 0
    ),
    I = dict(
        smile = 'CCC(C)C(C(=O)O)N',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    L = dict(
        smile = 'CC(C)CC(C(=O)O)N',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 4
    ),
    K = dict(
        smile = 'C(CCN)CC(C(=O)O)N',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    M = dict(
        smile = 'CSCCC(C(=O)O)N',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 4
    ),
    F = dict(
        smile = 'C1=CC=C(C=C1)CC(C(=O)O)N',
        first_atom_idx = 11,
        last_atom_idx = 8,
        hydroxyl_idx = 10,
        distogram_atom_idx = 7
    ),
    P = dict(
        smile = 'C1CC(NC1)C(=O)O',
        first_atom_idx = 3,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 2
    ),
    S = dict(
        smile = 'C(C(C(=O)O)N)O',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    T = dict(
        smile = 'CC(C(C(=O)O)N)O',
        first_atom_idx = 6,
        last_atom_idx = 3,
        hydroxyl_idx = 5,
        distogram_atom_idx = 2
    ),
    W = dict(
        smile = 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
        first_atom_idx = 14,
        last_atom_idx = 11,
        hydroxyl_idx = 13,
        distogram_atom_idx = 10
    ),
    Y = dict(
        smile = 'C1=CC(=CC=C1CC(C(=O)O)N)O',
        first_atom_idx = 11,
        last_atom_idx = 8,
        hydroxyl_idx = 10,
        distogram_atom_idx = 7
    ),
    V = dict(
        smile = 'CC(C)C(C(=O)O)N',
        first_atom_idx = 7,
        last_atom_idx = 4,
        hydroxyl_idx = 6,
        distogram_atom_idx = 3
    )
)

# nucleotides
# reordered from 5' to 3', so [O][P][...][C(3')][OH] - hydroxyl group removed when chaining into a nucleic acid chain

DNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1C(C(OC1N2C=NC3=C(N=CN=C32)N)COP(=O)(O)O)O',
        complement = 'T',
        first_atom_idx = 20,
        last_atom_idx = 1,
        hydroxyl_idx = 21,
    ),
    C = dict(
        smile = 'C1C(C(OC1N2C=CC(=NC2=O)N)COP(=O)(O)O)O',
        complement = 'G',
        first_atom_idx = 17,
        last_atom_idx = 1,
        hydroxyl_idx = 19
    ),
    G = dict(
        smile = 'C1C(C(OC1N2C=NC3=C2N=C(NC3=O)N)COP(=O)(O)O)O',
        complement = 'C',
        first_atom_idx = 21,
        last_atom_idx = 1,
        hydroxyl_idx = 22
    ),
    T = dict(
        smile = 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)O)O',
        complement = 'A',
        first_atom_idx = 19,
        last_atom_idx = 11,
        hydroxyl_idx = 20
    )
)

RNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O)N',
        complement = 'U',
        first_atom_idx = 19,
        last_atom_idx = 11,
        hydroxyl_idx = 20
    ),
    C = dict(
        smile = 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)COP(=O)([O-])[O-])O)O',
        complement = 'G',
        first_atom_idx = 17,
        last_atom_idx = 10,
        hydroxyl_idx = 19
    ),
    G = dict(
        smile = 'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)O)O)O)N=C(NC2=O)N',
        complement = 'C',
        first_atom_idx = 14,
        last_atom_idx = 7,
        hydroxyl_idx = 16
    ),
    U = dict(
        smile = 'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)(O)O)O)O',
        complement = 'A',
        first_atom_idx = 18,
        last_atom_idx = 10,
        hydroxyl_idx = 19
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
def reverse_complement_tensor(t: Int[' n']):
    complement = NUCLEIC_ACID_COMPLEMENT_TENSOR[t]
    reverse_complement = complement.flip(dims = (-1,))
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

CHAINABLE_BIOMOLECULES = [
    *HUMAN_AMINO_ACIDS.values(),
    *DNA_NUCLEOTIDES.values(),
    *RNA_NUCLEOTIDES.values()
]

METALS_AND_MISC = [
    *METALS.values(),
    *MISC.values(),
]

for entry in [*CHAINABLE_BIOMOLECULES, *METALS_AND_MISC]:
    mol = mol_from_smile(entry['smile'])
    entry['rdchem_mol'] = mol

# reorder all the chainable biomolecules
# to simplify chaining them up and specifying the peptide or phosphodiesterase bonds

for entry in CHAINABLE_BIOMOLECULES:
    mol = entry['rdchem_mol']

    atom_order = torch.arange(mol.GetNumAtoms())

    atom_order[entry['first_atom_idx']] = -1
    atom_order[entry['last_atom_idx']] = 1e4
    atom_order[entry['hydroxyl_idx']] = 1e4 + 1

    atom_reorder = atom_order.argsort()

    mol = Chem.RenumberAtoms(mol, atom_reorder.tolist())

    entry.update(
        atom_reorder_indices = atom_reorder,
        rdchem_mol = mol
    )
