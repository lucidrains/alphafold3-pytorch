import rdkit
from rdkit import Chem

# human amino acids

HUMAN_AMINO_ACIDS = dict(
    A = dict(
        smile = 'CC(C(=O)O)N'
    ),
    R = dict(
        smile = 'C(CC(C(=O)O)N)CN=C(N)N'
    ),
    N = dict(
        smile = 'C(C(C(=O)O)N)C(=O)N'
    ),
    D = dict(
        smile = 'C(C(C(=O)O)N)C(=O)O'
    ),
    C = dict(
        smile = 'C(C(C(=O)O)N)S'
    ),
    Q = dict(
        smile = 'C(CC(=O)N)C(C(=O)O)N'
    ),
    E = dict(
        smile = 'C(CC(=O)O)C(C(=O)O)N'
    ),
    G = dict(
        smile = 'C(C(=O)O)N'
    ),
    H = dict(
        smile = 'C1=C(NC=N1)CC(C(=O)O)N'
    ),
    I = dict(
        smile = 'CCC(C)C(C(=O)O)N'
    ),
    L = dict(
        smile = 'CC(C)CC(C(=O)O)N'
    ),
    K = dict(
        smile = 'C(CCN)CC(C(=O)O)N'
    ),
    M = dict(
        smile = 'CSCCC(C(=O)O)N'
    ),
    F = dict(
        smile = 'C1=CC=C(C=C1)CC(C(=O)O)N'
    ),
    P = dict(
        smile = 'C1CC(NC1)C(=O)O'
    ),
    S = dict(
        smile = 'C(C(C(=O)O)N)O'
    ),
    T = dict(
        smile = 'CC(C(C(=O)O)N)O'
    ),
    W = dict(
        smile = 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N'
    ),
    Y = dict(
        smile = 'C1=CC(=CC=C1CC(C(=O)O)N)O'
    ),
    V = dict(
        smile = 'CC(C)C(C(=O)O)N'
    )
)

# nucleotides

NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1=NC2=NC=NC(=C2N1)N'
    ),
    G = dict(
        smile = 'C1=NC2=C(N1)C(=O)NC(=N2)N'
    ),
    C = dict(
        smile = 'C1=C(NC(=O)N=C1)N'
    ),
    T = dict(
        smile = 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O'
    ),
    U = dict(
        smile = 'C1=CNC(=O)NC1=O'
    )
)

# initialize rdkit.Chem with canonical SMILES

for aa_dict in HUMAN_AMINO_ACIDS.values():
    aa_dict['rdkit_chem'] = Chem.MolFromSmiles(aa_dict['smile'])


for nuc_dict in NUCLEOTIDES.values():
    nuc_dict['rdkit_chem'] = Chem.MolFromSmiles(nuc_dict['smile'])
