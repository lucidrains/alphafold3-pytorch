# %% [markdown]
# # Clustering AlphaFold 3 PDB Dataset
#
# For clustering AlphaFold 3's PDB dataset, we follow the clustering procedure outlined in Abramson et al (2024).
#
# In order to reduce bias in the training and evaluation sets, clustering was performed on PDB chains and interfaces, as
# follows.
# • Chain-based clustering occur at 40% sequence homology for proteins, 100% homology for nucleic acids, 100%
# homology for peptides (<10 residues) and according to CCD identity for small molecules (i.e. only identical
# molecules share a cluster).
# • Chain-based clustering of polymers with modified residues is first done by mapping the modified residues to
# a standard residue using SCOP [23, 24] convention (https://github.com/biopython/biopython/
# blob/5ee5e69e649dbe17baefe3919e56e60b54f8e08f/Bio/Data/SCOPData.py). If the mod-
# ified residue could not be found as a mapping key or was mapped to a value longer than a single character, it was
# mapped to type unknown.
# • Interface-based clustering is a join on the cluster IDs of the constituent chains, such that interfaces I and J are
# in the same interface cluster C^interface only if their constituent chain pairs {I_1,I_2},{J_1,J_2} have the same chain
# cluster pairs {C_1^chain ,C_2^chain}.

# %%

import argparse
import glob
import os
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB.NeighborSearch import NeighborSearch
from gemmi import cif
from loguru import logger
from pdbeccdutils.core.exceptions import CCDUtilsError
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from alphafold3_pytorch.tensor_typing import typecheck
from scripts.filter_pdb_mmcifs import parse_mmcif_object

# Constants

CHAIN_SEQUENCES = List[Dict[str, Dict[str, str]]]
CLUSTERING_MOLECULE_TYPE = Literal["protein", "nucleic_acid", "peptide", "ligand", "unknown"]

PROTEIN_CODES_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}
DNA_CODES_3TO1 = {
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "T",
    "DN": "X",
}
RNA_CODES_3TO1 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "N": "X",
}

PROTEIN_CODES_1TO3 = {v: k for k, v in PROTEIN_CODES_3TO1.items()}
DNA_CODES_1TO3 = {v: k for k, v in DNA_CODES_3TO1.items()}
RNA_CODES_1TO3 = {v: k for k, v in RNA_CODES_3TO1.items()}

SCOP_CODES_3TO1 = {
    # From: https://github.com/biopython/biopython/blob/5ee5e69e649dbe17baefe3919e56e60b54f8e08f/Bio/Data/SCOPData.py
    "00C": "C",
    "01W": "X",
    "02K": "A",
    "03Y": "C",
    "07O": "C",
    "08P": "C",
    "0A0": "D",
    "0A1": "Y",
    "0A2": "K",
    "0A8": "C",
    "0AA": "V",
    "0AB": "V",
    "0AC": "G",
    "0AD": "G",
    "0AF": "W",
    "0AG": "L",
    "0AH": "S",
    "0AK": "D",
    "0AM": "A",
    "0AP": "C",
    "0AU": "U",
    "0AV": "A",
    "0AZ": "P",
    "0BN": "F",
    "0C ": "C",
    "0CS": "A",
    "0DC": "C",
    "0DG": "G",
    "0DT": "T",
    "0FL": "A",
    "0G ": "G",
    "0NC": "A",
    "0SP": "A",
    "0U ": "U",
    "0YG": "YG",
    "10C": "C",
    "125": "U",
    "126": "U",
    "127": "U",
    "128": "N",
    "12A": "A",
    "143": "C",
    "175": "ASG",
    "193": "X",
    "1AP": "A",
    "1MA": "A",
    "1MG": "G",
    "1PA": "F",
    "1PI": "A",
    "1PR": "N",
    "1SC": "C",
    "1TQ": "W",
    "1TY": "Y",
    "1X6": "S",
    "200": "F",
    "23F": "F",
    "23S": "X",
    "26B": "T",
    "2AD": "X",
    "2AG": "A",
    "2AO": "X",
    "2AR": "A",
    "2AS": "X",
    "2AT": "T",
    "2AU": "U",
    "2BD": "I",
    "2BT": "T",
    "2BU": "A",
    "2CO": "C",
    "2DA": "A",
    "2DF": "N",
    "2DM": "N",
    "2DO": "X",
    "2DT": "T",
    "2EG": "G",
    "2FE": "N",
    "2FI": "N",
    "2FM": "M",
    "2GT": "T",
    "2HF": "H",
    "2LU": "L",
    "2MA": "A",
    "2MG": "G",
    "2ML": "L",
    "2MR": "R",
    "2MT": "P",
    "2MU": "U",
    "2NT": "T",
    "2OM": "U",
    "2OT": "T",
    "2PI": "X",
    "2PR": "G",
    "2SA": "N",
    "2SI": "X",
    "2ST": "T",
    "2TL": "T",
    "2TY": "Y",
    "2VA": "V",
    "2XA": "C",
    "32S": "X",
    "32T": "X",
    "3AH": "H",
    "3AR": "X",
    "3CF": "F",
    "3DA": "A",
    "3DR": "N",
    "3GA": "A",
    "3MD": "D",
    "3ME": "U",
    "3NF": "Y",
    "3QN": "K",
    "3TY": "X",
    "3XH": "G",
    "4AC": "N",
    "4BF": "Y",
    "4CF": "F",
    "4CY": "M",
    "4DP": "W",
    "4F3": "GYG",
    "4FB": "P",
    "4FW": "W",
    "4HT": "W",
    "4IN": "W",
    "4MF": "N",
    "4MM": "X",
    "4OC": "C",
    "4PC": "C",
    "4PD": "C",
    "4PE": "C",
    "4PH": "F",
    "4SC": "C",
    "4SU": "U",
    "4TA": "N",
    "4U7": "A",
    "56A": "H",
    "5AA": "A",
    "5AB": "A",
    "5AT": "T",
    "5BU": "U",
    "5CG": "G",
    "5CM": "C",
    "5CS": "C",
    "5FA": "A",
    "5FC": "C",
    "5FU": "U",
    "5HP": "E",
    "5HT": "T",
    "5HU": "U",
    "5IC": "C",
    "5IT": "T",
    "5IU": "U",
    "5MC": "C",
    "5MD": "N",
    "5MU": "U",
    "5NC": "C",
    "5PC": "C",
    "5PY": "T",
    "5SE": "U",
    "5ZA": "TWG",
    "64T": "T",
    "6CL": "K",
    "6CT": "T",
    "6CW": "W",
    "6HA": "A",
    "6HC": "C",
    "6HG": "G",
    "6HN": "K",
    "6HT": "T",
    "6IA": "A",
    "6MA": "A",
    "6MC": "A",
    "6MI": "N",
    "6MT": "A",
    "6MZ": "N",
    "6OG": "G",
    "70U": "U",
    "7DA": "A",
    "7GU": "G",
    "7JA": "I",
    "7MG": "G",
    "8AN": "A",
    "8FG": "G",
    "8MG": "G",
    "8OG": "G",
    "9NE": "E",
    "9NF": "F",
    "9NR": "R",
    "9NV": "V",
    "A  ": "A",
    "A1P": "N",
    "A23": "A",
    "A2L": "A",
    "A2M": "A",
    "A34": "A",
    "A35": "A",
    "A38": "A",
    "A39": "A",
    "A3A": "A",
    "A3P": "A",
    "A40": "A",
    "A43": "A",
    "A44": "A",
    "A47": "A",
    "A5L": "A",
    "A5M": "C",
    "A5N": "N",
    "A5O": "A",
    "A66": "X",
    "AA3": "A",
    "AA4": "A",
    "AAR": "R",
    "AB7": "X",
    "ABA": "A",
    "ABR": "A",
    "ABS": "A",
    "ABT": "N",
    "ACB": "D",
    "ACL": "R",
    "AD2": "A",
    "ADD": "X",
    "ADX": "N",
    "AEA": "X",
    "AEI": "D",
    "AET": "A",
    "AFA": "N",
    "AFF": "N",
    "AFG": "G",
    "AGM": "R",
    "AGT": "C",
    "AHB": "N",
    "AHH": "X",
    "AHO": "A",
    "AHP": "A",
    "AHS": "X",
    "AHT": "X",
    "AIB": "A",
    "AKL": "D",
    "AKZ": "D",
    "ALA": "A",
    "ALC": "A",
    "ALM": "A",
    "ALN": "A",
    "ALO": "T",
    "ALQ": "X",
    "ALS": "A",
    "ALT": "A",
    "ALV": "A",
    "ALY": "K",
    "AN8": "A",
    "AP7": "A",
    "APE": "X",
    "APH": "A",
    "API": "K",
    "APK": "K",
    "APM": "X",
    "APP": "X",
    "AR2": "R",
    "AR4": "E",
    "AR7": "R",
    "ARG": "R",
    "ARM": "R",
    "ARO": "R",
    "ARV": "X",
    "AS ": "A",
    "AS2": "D",
    "AS9": "X",
    "ASA": "D",
    "ASB": "D",
    "ASI": "D",
    "ASK": "D",
    "ASL": "D",
    "ASM": "X",
    "ASN": "N",
    "ASP": "D",
    "ASQ": "D",
    "ASU": "N",
    "ASX": "B",
    "ATD": "T",
    "ATL": "T",
    "ATM": "T",
    "AVC": "A",
    "AVN": "X",
    "AYA": "A",
    "AYG": "AYG",
    "AZK": "K",
    "AZS": "S",
    "AZY": "Y",
    "B1F": "F",
    "B1P": "N",
    "B2A": "A",
    "B2F": "F",
    "B2I": "I",
    "B2V": "V",
    "B3A": "A",
    "B3D": "D",
    "B3E": "E",
    "B3K": "K",
    "B3L": "X",
    "B3M": "X",
    "B3Q": "X",
    "B3S": "S",
    "B3T": "X",
    "B3U": "H",
    "B3X": "N",
    "B3Y": "Y",
    "BB6": "C",
    "BB7": "C",
    "BB8": "F",
    "BB9": "C",
    "BBC": "C",
    "BCS": "C",
    "BE2": "X",
    "BFD": "D",
    "BG1": "S",
    "BGM": "G",
    "BH2": "D",
    "BHD": "D",
    "BIF": "F",
    "BIL": "X",
    "BIU": "I",
    "BJH": "X",
    "BLE": "L",
    "BLY": "K",
    "BMP": "N",
    "BMT": "T",
    "BNN": "F",
    "BNO": "X",
    "BOE": "T",
    "BOR": "R",
    "BPE": "C",
    "BRU": "U",
    "BSE": "S",
    "BT5": "N",
    "BTA": "L",
    "BTC": "C",
    "BTR": "W",
    "BUC": "C",
    "BUG": "V",
    "BVP": "U",
    "BZG": "N",
    "C  ": "C",
    "C12": "TYG",
    "C1X": "K",
    "C25": "C",
    "C2L": "C",
    "C2S": "C",
    "C31": "C",
    "C32": "C",
    "C34": "C",
    "C36": "C",
    "C37": "C",
    "C38": "C",
    "C3Y": "C",
    "C42": "C",
    "C43": "C",
    "C45": "C",
    "C46": "C",
    "C49": "C",
    "C4R": "C",
    "C4S": "C",
    "C5C": "C",
    "C66": "X",
    "C6C": "C",
    "C99": "TFG",
    "CAF": "C",
    "CAL": "X",
    "CAR": "C",
    "CAS": "C",
    "CAV": "X",
    "CAY": "C",
    "CB2": "C",
    "CBR": "C",
    "CBV": "C",
    "CCC": "C",
    "CCL": "K",
    "CCS": "C",
    "CCY": "CYG",
    "CDE": "X",
    "CDV": "X",
    "CDW": "C",
    "CEA": "C",
    "CFL": "C",
    "CFY": "FCYG",
    "CG1": "G",
    "CGA": "E",
    "CGU": "E",
    "CH ": "C",
    "CH6": "MYG",
    "CH7": "KYG",
    "CHF": "X",
    "CHG": "X",
    "CHP": "G",
    "CHS": "X",
    "CIR": "R",
    "CJO": "GYG",
    "CLE": "L",
    "CLG": "K",
    "CLH": "K",
    "CLV": "AFG",
    "CM0": "N",
    "CME": "C",
    "CMH": "C",
    "CML": "C",
    "CMR": "C",
    "CMT": "C",
    "CNU": "U",
    "CP1": "C",
    "CPC": "X",
    "CPI": "X",
    "CQR": "GYG",
    "CR0": "TLG",
    "CR2": "GYG",
    "CR5": "G",
    "CR7": "KYG",
    "CR8": "HYG",
    "CRF": "TWG",
    "CRG": "THG",
    "CRK": "MYG",
    "CRO": "GYG",
    "CRQ": "QYG",
    "CRU": "EYG",
    "CRW": "ASG",
    "CRX": "ASG",
    "CS0": "C",
    "CS1": "C",
    "CS3": "C",
    "CS4": "C",
    "CS8": "N",
    "CSA": "C",
    "CSB": "C",
    "CSD": "C",
    "CSE": "C",
    "CSF": "C",
    "CSH": "SHG",
    "CSI": "G",
    "CSJ": "C",
    "CSL": "C",
    "CSO": "C",
    "CSP": "C",
    "CSR": "C",
    "CSS": "C",
    "CSU": "C",
    "CSW": "C",
    "CSX": "C",
    "CSY": "SYG",
    "CSZ": "C",
    "CTE": "W",
    "CTG": "T",
    "CTH": "T",
    "CUC": "X",
    "CWR": "S",
    "CXM": "M",
    "CY0": "C",
    "CY1": "C",
    "CY3": "C",
    "CY4": "C",
    "CYA": "C",
    "CYD": "C",
    "CYF": "C",
    "CYG": "C",
    "CYJ": "X",
    "CYM": "C",
    "CYQ": "C",
    "CYR": "C",
    "CYS": "C",
    "CZ2": "C",
    "CZO": "GYG",
    "CZZ": "C",
    "D11": "T",
    "D1P": "N",
    "D3 ": "N",
    "D33": "N",
    "D3P": "G",
    "D3T": "T",
    "D4M": "T",
    "D4P": "X",
    "DA ": "A",
    "DA2": "X",
    "DAB": "A",
    "DAH": "F",
    "DAL": "A",
    "DAR": "R",
    "DAS": "D",
    "DBB": "T",
    "DBM": "N",
    "DBS": "S",
    "DBU": "T",
    "DBY": "Y",
    "DBZ": "A",
    "DC ": "C",
    "DC2": "C",
    "DCG": "G",
    "DCI": "X",
    "DCL": "X",
    "DCT": "C",
    "DCY": "C",
    "DDE": "H",
    "DDG": "G",
    "DDN": "U",
    "DDX": "N",
    "DFC": "C",
    "DFG": "G",
    "DFI": "X",
    "DFO": "X",
    "DFT": "N",
    "DG ": "G",
    "DGH": "G",
    "DGI": "G",
    "DGL": "E",
    "DGN": "Q",
    "DHA": "S",
    "DHI": "H",
    "DHL": "X",
    "DHN": "V",
    "DHP": "X",
    "DHU": "U",
    "DHV": "V",
    "DI ": "I",
    "DIL": "I",
    "DIR": "R",
    "DIV": "V",
    "DLE": "L",
    "DLS": "K",
    "DLY": "K",
    "DM0": "K",
    "DMH": "N",
    "DMK": "D",
    "DMT": "X",
    "DN ": "N",
    "DNE": "L",
    "DNG": "L",
    "DNL": "K",
    "DNM": "L",
    "DNP": "A",
    "DNR": "C",
    "DNS": "K",
    "DOA": "X",
    "DOC": "C",
    "DOH": "D",
    "DON": "L",
    "DPB": "T",
    "DPH": "F",
    "DPL": "P",
    "DPP": "A",
    "DPQ": "Y",
    "DPR": "P",
    "DPY": "N",
    "DRM": "U",
    "DRP": "N",
    "DRT": "T",
    "DRZ": "N",
    "DSE": "S",
    "DSG": "N",
    "DSN": "S",
    "DSP": "D",
    "DT ": "T",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DU ": "U",
    "DVA": "V",
    "DXD": "N",
    "DXN": "N",
    "DYG": "DYG",
    "DYS": "C",
    "DZM": "A",
    "E  ": "A",
    "E1X": "A",
    "ECC": "Q",
    "EDA": "A",
    "EFC": "C",
    "EHP": "F",
    "EIT": "T",
    "ENP": "N",
    "ESB": "Y",
    "ESC": "M",
    "EXB": "X",
    "EXY": "L",
    "EY5": "N",
    "EYS": "X",
    "F2F": "F",
    "FA2": "A",
    "FA5": "N",
    "FAG": "N",
    "FAI": "N",
    "FB5": "A",
    "FB6": "A",
    "FCL": "F",
    "FFD": "N",
    "FGA": "E",
    "FGL": "G",
    "FGP": "S",
    "FHL": "X",
    "FHO": "K",
    "FHU": "U",
    "FLA": "A",
    "FLE": "L",
    "FLT": "Y",
    "FME": "M",
    "FMG": "G",
    "FMU": "N",
    "FOE": "C",
    "FOX": "G",
    "FP9": "P",
    "FPA": "F",
    "FRD": "X",
    "FT6": "W",
    "FTR": "W",
    "FTY": "Y",
    "FVA": "V",
    "FZN": "K",
    "G  ": "G",
    "G25": "G",
    "G2L": "G",
    "G2S": "G",
    "G31": "G",
    "G32": "G",
    "G33": "G",
    "G36": "G",
    "G38": "G",
    "G42": "G",
    "G46": "G",
    "G47": "G",
    "G48": "G",
    "G49": "G",
    "G4P": "N",
    "G7M": "G",
    "GAO": "G",
    "GAU": "E",
    "GCK": "C",
    "GCM": "X",
    "GDP": "G",
    "GDR": "G",
    "GFL": "G",
    "GGL": "E",
    "GH3": "G",
    "GHG": "Q",
    "GHP": "G",
    "GL3": "G",
    "GLH": "Q",
    "GLJ": "E",
    "GLK": "E",
    "GLM": "X",
    "GLN": "Q",
    "GLQ": "E",
    "GLU": "E",
    "GLX": "Z",
    "GLY": "G",
    "GLZ": "G",
    "GMA": "E",
    "GMS": "G",
    "GMU": "U",
    "GN7": "G",
    "GND": "X",
    "GNE": "N",
    "GOM": "G",
    "GPL": "K",
    "GS ": "G",
    "GSC": "G",
    "GSR": "G",
    "GSS": "G",
    "GSU": "E",
    "GT9": "C",
    "GTP": "G",
    "GVL": "X",
    "GYC": "CYG",
    "GYS": "SYG",
    "H2U": "U",
    "H5M": "P",
    "HAC": "A",
    "HAR": "R",
    "HBN": "H",
    "HCS": "X",
    "HDP": "U",
    "HEU": "U",
    "HFA": "X",
    "HGL": "X",
    "HHI": "H",
    "HHK": "AK",
    "HIA": "H",
    "HIC": "H",
    "HIP": "H",
    "HIQ": "H",
    "HIS": "H",
    "HL2": "L",
    "HLU": "L",
    "HMR": "R",
    "HOL": "N",
    "HPC": "F",
    "HPE": "F",
    "HPH": "F",
    "HPQ": "F",
    "HQA": "A",
    "HRG": "R",
    "HRP": "W",
    "HS8": "H",
    "HS9": "H",
    "HSE": "S",
    "HSL": "S",
    "HSO": "H",
    "HTI": "C",
    "HTN": "N",
    "HTR": "W",
    "HV5": "A",
    "HVA": "V",
    "HY3": "P",
    "HYP": "P",
    "HZP": "P",
    "I  ": "I",
    "I2M": "I",
    "I58": "K",
    "I5C": "C",
    "IAM": "A",
    "IAR": "R",
    "IAS": "D",
    "IC ": "C",
    "IEL": "K",
    "IEY": "HYG",
    "IG ": "G",
    "IGL": "G",
    "IGU": "G",
    "IIC": "SHG",
    "IIL": "I",
    "ILE": "I",
    "ILG": "E",
    "ILX": "I",
    "IMC": "C",
    "IML": "I",
    "IOY": "F",
    "IPG": "G",
    "IPN": "N",
    "IRN": "N",
    "IT1": "K",
    "IU ": "U",
    "IYR": "Y",
    "IYT": "T",
    "IZO": "M",
    "JJJ": "C",
    "JJK": "C",
    "JJL": "C",
    "JW5": "N",
    "K1R": "C",
    "KAG": "G",
    "KCX": "K",
    "KGC": "K",
    "KNB": "A",
    "KOR": "M",
    "KPI": "K",
    "KST": "K",
    "KYQ": "K",
    "L2A": "X",
    "LA2": "K",
    "LAA": "D",
    "LAL": "A",
    "LBY": "K",
    "LC ": "C",
    "LCA": "A",
    "LCC": "N",
    "LCG": "G",
    "LCH": "N",
    "LCK": "K",
    "LCX": "K",
    "LDH": "K",
    "LED": "L",
    "LEF": "L",
    "LEH": "L",
    "LEI": "V",
    "LEM": "L",
    "LEN": "L",
    "LET": "X",
    "LEU": "L",
    "LEX": "L",
    "LG ": "G",
    "LGP": "G",
    "LHC": "X",
    "LHU": "U",
    "LKC": "N",
    "LLP": "K",
    "LLY": "K",
    "LME": "E",
    "LMF": "K",
    "LMQ": "Q",
    "LMS": "N",
    "LP6": "K",
    "LPD": "P",
    "LPG": "G",
    "LPL": "X",
    "LPS": "S",
    "LSO": "X",
    "LTA": "X",
    "LTR": "W",
    "LVG": "G",
    "LVN": "V",
    "LYF": "K",
    "LYK": "K",
    "LYM": "K",
    "LYN": "K",
    "LYR": "K",
    "LYS": "K",
    "LYX": "K",
    "LYZ": "K",
    "M0H": "C",
    "M1G": "G",
    "M2G": "G",
    "M2L": "K",
    "M2S": "M",
    "M30": "G",
    "M3L": "K",
    "M5M": "C",
    "MA ": "A",
    "MA6": "A",
    "MA7": "A",
    "MAA": "A",
    "MAD": "A",
    "MAI": "R",
    "MBQ": "Y",
    "MBZ": "N",
    "MC1": "S",
    "MCG": "X",
    "MCL": "K",
    "MCS": "C",
    "MCY": "C",
    "MD3": "C",
    "MD6": "G",
    "MDH": "X",
    "MDO": "ASG",
    "MDR": "N",
    "MEA": "F",
    "MED": "M",
    "MEG": "E",
    "MEN": "N",
    "MEP": "U",
    "MEQ": "Q",
    "MET": "M",
    "MEU": "G",
    "MF3": "X",
    "MFC": "GYG",
    "MG1": "G",
    "MGG": "R",
    "MGN": "Q",
    "MGQ": "A",
    "MGV": "G",
    "MGY": "G",
    "MHL": "L",
    "MHO": "M",
    "MHS": "H",
    "MIA": "A",
    "MIS": "S",
    "MK8": "L",
    "ML3": "K",
    "MLE": "L",
    "MLL": "L",
    "MLY": "K",
    "MLZ": "K",
    "MME": "M",
    "MMO": "R",
    "MMT": "T",
    "MND": "N",
    "MNL": "L",
    "MNU": "U",
    "MNV": "V",
    "MOD": "X",
    "MP8": "P",
    "MPH": "X",
    "MPJ": "X",
    "MPQ": "G",
    "MRG": "G",
    "MSA": "G",
    "MSE": "M",
    "MSL": "M",
    "MSO": "M",
    "MSP": "X",
    "MT2": "M",
    "MTR": "T",
    "MTU": "A",
    "MTY": "Y",
    "MVA": "V",
    "N  ": "N",
    "N10": "S",
    "N2C": "X",
    "N5I": "N",
    "N5M": "C",
    "N6G": "G",
    "N7P": "P",
    "NA8": "A",
    "NAL": "A",
    "NAM": "A",
    "NB8": "N",
    "NBQ": "Y",
    "NC1": "S",
    "NCB": "A",
    "NCX": "N",
    "NCY": "X",
    "NDF": "F",
    "NDN": "U",
    "NEM": "H",
    "NEP": "H",
    "NF2": "N",
    "NFA": "F",
    "NHL": "E",
    "NIT": "X",
    "NIY": "Y",
    "NLE": "L",
    "NLN": "L",
    "NLO": "L",
    "NLP": "L",
    "NLQ": "Q",
    "NMC": "G",
    "NMM": "R",
    "NMS": "T",
    "NMT": "T",
    "NNH": "R",
    "NP3": "N",
    "NPH": "C",
    "NPI": "A",
    "NRP": "LYG",
    "NRQ": "MYG",
    "NSK": "X",
    "NTY": "Y",
    "NVA": "V",
    "NYC": "TWG",
    "NYG": "NYG",
    "NYM": "N",
    "NYS": "C",
    "NZH": "H",
    "O12": "X",
    "O2C": "N",
    "O2G": "G",
    "OAD": "N",
    "OAS": "S",
    "OBF": "X",
    "OBS": "X",
    "OCS": "C",
    "OCY": "C",
    "ODP": "N",
    "OHI": "H",
    "OHS": "D",
    "OIC": "X",
    "OIP": "I",
    "OLE": "X",
    "OLT": "T",
    "OLZ": "S",
    "OMC": "C",
    "OMG": "G",
    "OMT": "M",
    "OMU": "U",
    "ONE": "U",
    "ONH": "A",
    "ONL": "X",
    "OPR": "R",
    "ORN": "A",
    "ORQ": "R",
    "OSE": "S",
    "OTB": "X",
    "OTH": "T",
    "OTY": "Y",
    "OXX": "D",
    "P  ": "G",
    "P1L": "C",
    "P1P": "N",
    "P2T": "T",
    "P2U": "U",
    "P2Y": "P",
    "P5P": "A",
    "PAQ": "Y",
    "PAS": "D",
    "PAT": "W",
    "PAU": "A",
    "PBB": "C",
    "PBF": "F",
    "PBT": "N",
    "PCA": "E",
    "PCC": "P",
    "PCE": "X",
    "PCS": "F",
    "PDL": "X",
    "PDU": "U",
    "PEC": "C",
    "PF5": "F",
    "PFF": "F",
    "PFX": "X",
    "PG1": "S",
    "PG7": "G",
    "PG9": "G",
    "PGL": "X",
    "PGN": "G",
    "PGP": "G",
    "PGY": "G",
    "PHA": "F",
    "PHD": "D",
    "PHE": "F",
    "PHI": "F",
    "PHL": "F",
    "PHM": "F",
    "PIA": "AYG",
    "PIV": "X",
    "PLE": "L",
    "PM3": "F",
    "PMT": "C",
    "POM": "P",
    "PPN": "F",
    "PPU": "A",
    "PPW": "G",
    "PQ1": "N",
    "PR3": "C",
    "PR5": "A",
    "PR9": "P",
    "PRN": "A",
    "PRO": "P",
    "PRS": "P",
    "PSA": "F",
    "PSH": "H",
    "PST": "T",
    "PSU": "U",
    "PSW": "C",
    "PTA": "X",
    "PTH": "Y",
    "PTM": "Y",
    "PTR": "Y",
    "PU ": "A",
    "PUY": "N",
    "PVH": "H",
    "PVL": "X",
    "PYA": "A",
    "PYO": "U",
    "PYX": "C",
    "PYY": "N",
    "QLG": "QLG",
    "QMM": "Q",
    "QPA": "C",
    "QPH": "F",
    "QUO": "G",
    "R  ": "A",
    "R1A": "C",
    "R4K": "W",
    "RC7": "HYG",
    "RE0": "W",
    "RE3": "W",
    "RIA": "A",
    "RMP": "A",
    "RON": "X",
    "RT ": "T",
    "RTP": "N",
    "S1H": "S",
    "S2C": "C",
    "S2D": "A",
    "S2M": "T",
    "S2P": "A",
    "S4A": "A",
    "S4C": "C",
    "S4G": "G",
    "S4U": "U",
    "S6G": "G",
    "SAC": "S",
    "SAH": "C",
    "SAR": "G",
    "SBL": "S",
    "SC ": "C",
    "SCH": "C",
    "SCS": "C",
    "SCY": "C",
    "SD2": "X",
    "SDG": "G",
    "SDP": "S",
    "SEB": "S",
    "SEC": "A",
    "SEG": "A",
    "SEL": "S",
    "SEM": "S",
    "SEN": "S",
    "SEP": "S",
    "SER": "S",
    "SET": "S",
    "SGB": "S",
    "SHC": "C",
    "SHP": "G",
    "SHR": "K",
    "SIB": "C",
    "SIC": "DC",
    "SLA": "P",
    "SLR": "P",
    "SLZ": "K",
    "SMC": "C",
    "SME": "M",
    "SMF": "F",
    "SMP": "A",
    "SMT": "T",
    "SNC": "C",
    "SNN": "N",
    "SOC": "C",
    "SOS": "N",
    "SOY": "S",
    "SPT": "T",
    "SRA": "A",
    "SSU": "U",
    "STY": "Y",
    "SUB": "X",
    "SUI": "DG",
    "SUN": "S",
    "SUR": "U",
    "SVA": "S",
    "SVV": "S",
    "SVW": "S",
    "SVX": "S",
    "SVY": "S",
    "SVZ": "X",
    "SWG": "SWG",
    "SYS": "C",
    "T  ": "T",
    "T11": "F",
    "T23": "T",
    "T2S": "T",
    "T2T": "N",
    "T31": "U",
    "T32": "T",
    "T36": "T",
    "T37": "T",
    "T38": "T",
    "T39": "T",
    "T3P": "T",
    "T41": "T",
    "T48": "T",
    "T49": "T",
    "T4S": "T",
    "T5O": "U",
    "T5S": "T",
    "T66": "X",
    "T6A": "A",
    "TA3": "T",
    "TA4": "X",
    "TAF": "T",
    "TAL": "N",
    "TAV": "D",
    "TBG": "V",
    "TBM": "T",
    "TC1": "C",
    "TCP": "T",
    "TCQ": "Y",
    "TCR": "W",
    "TCY": "A",
    "TDD": "L",
    "TDY": "T",
    "TFE": "T",
    "TFO": "A",
    "TFQ": "F",
    "TFT": "T",
    "TGP": "G",
    "TH6": "T",
    "THC": "T",
    "THO": "X",
    "THR": "T",
    "THX": "N",
    "THZ": "R",
    "TIH": "A",
    "TLB": "N",
    "TLC": "T",
    "TLN": "U",
    "TMB": "T",
    "TMD": "T",
    "TNB": "C",
    "TNR": "S",
    "TOX": "W",
    "TP1": "T",
    "TPC": "C",
    "TPG": "G",
    "TPH": "X",
    "TPL": "W",
    "TPO": "T",
    "TPQ": "Y",
    "TQI": "W",
    "TQQ": "W",
    "TRF": "W",
    "TRG": "K",
    "TRN": "W",
    "TRO": "W",
    "TRP": "W",
    "TRQ": "W",
    "TRW": "W",
    "TRX": "W",
    "TS ": "N",
    "TST": "X",
    "TT ": "N",
    "TTD": "T",
    "TTI": "U",
    "TTM": "T",
    "TTQ": "W",
    "TTS": "Y",
    "TY1": "Y",
    "TY2": "Y",
    "TY3": "Y",
    "TY5": "Y",
    "TYB": "Y",
    "TYI": "Y",
    "TYJ": "Y",
    "TYN": "Y",
    "TYO": "Y",
    "TYQ": "Y",
    "TYR": "Y",
    "TYS": "Y",
    "TYT": "Y",
    "TYU": "N",
    "TYW": "Y",
    "TYX": "X",
    "TYY": "Y",
    "TZB": "X",
    "TZO": "X",
    "U  ": "U",
    "U25": "U",
    "U2L": "U",
    "U2N": "U",
    "U2P": "U",
    "U31": "U",
    "U33": "U",
    "U34": "U",
    "U36": "U",
    "U37": "U",
    "U8U": "U",
    "UAR": "U",
    "UCL": "U",
    "UD5": "U",
    "UDP": "N",
    "UFP": "N",
    "UFR": "U",
    "UFT": "U",
    "UMA": "A",
    "UMP": "U",
    "UMS": "U",
    "UN1": "X",
    "UN2": "X",
    "UNK": "X",
    "UR3": "U",
    "URD": "U",
    "US1": "U",
    "US2": "U",
    "US3": "T",
    "US5": "U",
    "USM": "U",
    "VAD": "V",
    "VAF": "V",
    "VAL": "V",
    "VB1": "K",
    "VDL": "X",
    "VLL": "X",
    "VLM": "X",
    "VMS": "X",
    "VOL": "X",
    "WCR": "GYG",
    "X  ": "G",
    "X2W": "E",
    "X4A": "N",
    "X9Q": "AFG",
    "XAD": "A",
    "XAE": "N",
    "XAL": "A",
    "XAR": "N",
    "XCL": "C",
    "XCN": "C",
    "XCP": "X",
    "XCR": "C",
    "XCS": "N",
    "XCT": "C",
    "XCY": "C",
    "XGA": "N",
    "XGL": "G",
    "XGR": "G",
    "XGU": "G",
    "XPR": "P",
    "XSN": "N",
    "XTH": "T",
    "XTL": "T",
    "XTR": "T",
    "XTS": "G",
    "XTY": "N",
    "XUA": "A",
    "XUG": "G",
    "XX1": "K",
    "XXY": "THG",
    "XYG": "DYG",
    "Y  ": "A",
    "YCM": "C",
    "YG ": "G",
    "YOF": "Y",
    "YRR": "N",
    "YYG": "G",
    "Z  ": "C",
    "Z01": "A",
    "ZAD": "A",
    "ZAL": "A",
    "ZBC": "C",
    "ZBU": "U",
    "ZCL": "F",
    "ZCY": "C",
    "ZDU": "U",
    "ZFB": "X",
    "ZGU": "G",
    "ZHP": "N",
    "ZTH": "T",
    "ZU0": "T",
    "ZZJ": "A",
}


# Helper functions


@typecheck
def read_ccd_codes_from_pdb_components_file(path_to_cif: str) -> Set[str]:
    """
    Collect the CCD codes of multiple compounds stored in the wwPDB CCD
    `components.cif` file.

    :param path_to_cif: Path to the `components.cif` file with
        multiple ligands in it.

    :return: CCD codes of all ligands stored in the `components.cif` file.
    """
    assert path_to_cif.endswith(".cif"), "The input file must be an mmCIF file."
    if not os.path.isfile(path_to_cif):
        raise ValueError("File '{}' does not exists".format(path_to_cif))

    result_bag = set()

    for block in cif.read(path_to_cif):
        try:
            result_bag.add(block.name)
        except CCDUtilsError as e:
            logger.error(f"ERROR: Data block {block.name} not processed. Reason: ({str(e)}).")

    return result_bag


@typecheck
def convert_residue_three_to_one(
    residue: str,
    ccd_codes: Set[str],
) -> Tuple[str, CLUSTERING_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).

    NOTE: All unknown residues residues (be they protein, RNA, DNA, or ligands) are converted to 'X'.
    """
    if residue in PROTEIN_CODES_3TO1:
        return PROTEIN_CODES_3TO1[residue], "protein"
    elif residue in DNA_CODES_3TO1:
        return DNA_CODES_3TO1[residue], "nucleic_acid"
    elif residue in RNA_CODES_3TO1:
        return RNA_CODES_3TO1[residue], "nucleic_acid"
    elif residue in ccd_codes:
        return residue, "ligand"
    else:
        return "X", "unknown"


@typecheck
def convert_ambiguous_residue_three_to_one(
    residue: str,
    ccd_codes: Set[str],
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> Tuple[str, CLUSTERING_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).

    NOTE: All unknown residues or unmappable modified residues (be they protein, RNA, DNA, or ligands) are converted to 'X'.
    """
    is_modified_protein_residue = (
        molecule_type == "protein"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )
    is_modified_dna_residue = (
        molecule_type == "dna"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )
    is_modified_rna_residue = (
        molecule_type == "rna"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )

    # Map modified residues to their one-letter codes, if applicable
    if is_modified_protein_residue or is_modified_dna_residue or is_modified_rna_residue:
        one_letter_mapped_residue = SCOP_CODES_3TO1[residue]
        if is_modified_protein_residue:
            mapped_residue = PROTEIN_CODES_1TO3[one_letter_mapped_residue]
        elif is_modified_dna_residue:
            mapped_residue = DNA_CODES_1TO3[one_letter_mapped_residue]
        elif is_modified_rna_residue:
            mapped_residue = RNA_CODES_1TO3[one_letter_mapped_residue]
    else:
        mapped_residue = residue

    if mapped_residue in PROTEIN_CODES_3TO1:
        return PROTEIN_CODES_3TO1[mapped_residue], "protein"
    elif mapped_residue in DNA_CODES_3TO1:
        return DNA_CODES_3TO1[mapped_residue], "nucleic_acid"
    elif mapped_residue in RNA_CODES_3TO1:
        return RNA_CODES_3TO1[mapped_residue], "nucleic_acid"
    elif mapped_residue in ccd_codes:
        return mapped_residue, "ligand"
    else:
        return "X", "unknown"


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_file(
    filepath: str, ccd_codes: Set[str], min_num_residues_for_protein_classification: int = 10
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Parse an mmCIF file and return a dictionary mapping chain IDs
    to sequences for all molecule types (i.e., proteins, nucleic acids, peptides, ligands, etc)
    as well as a set of chain ID pairs denoting structural interfaces.
    """
    assert filepath.endswith(".cif"), "The input file must be an mmCIF file."
    mmcif_object = parse_mmcif_object(filepath)
    model = mmcif_object.structure

    # NOTE: After filtering, only heavy (non-hydrogen) atoms remain in the structure
    # all_atoms = [atom for atom in structure.get_atoms()]
    # neighbor_search = NeighborSearch(all_atoms)

    sequences = {}
    interface_chain_ids = set()
    for chain in model:
        num_ligands_in_chain = 0
        one_letter_seq_tokens = []
        token_molecule_types = set()

        # First find the most common molecule type in the chain
        molecule_type_counter = Counter(
            [convert_residue_three_to_one(res.resname, ccd_codes)[-1] for res in chain]
        )
        chain_most_common_molecule_types = molecule_type_counter.most_common(2)
        chain_most_common_molecule_type = chain_most_common_molecule_types[0][0]
        if (
            chain_most_common_molecule_type == "ligand"
            and len(chain_most_common_molecule_types) > 1
        ):
            # NOTE: Ligands may be the most common molecule type in a chain, in which case
            # the second most common molecule type is required for sequence mapping
            chain_most_common_molecule_type = chain_most_common_molecule_types[1][0]

        for res in chain:
            # Then convert each residue to a one-letter code using the most common molecule type in the chain
            one_letter_residue, molecule_type = convert_ambiguous_residue_three_to_one(
                res.resname, ccd_codes, molecule_type=chain_most_common_molecule_type
            )
            if molecule_type == "ligand":
                num_ligands_in_chain += 1
                sequences[
                    f"{chain.id}:{molecule_type}-{res.resname}-{num_ligands_in_chain}"
                ] = one_letter_residue
            else:
                assert (
                    molecule_type == chain_most_common_molecule_type
                ), f"Residue {res.resname} in chain {chain.id} has an unexpected molecule type of `{molecule_type}` (vs. the expected molecule type of `{chain_most_common_molecule_type}`)."
                one_letter_seq_tokens.append(one_letter_residue)
                token_molecule_types.add(molecule_type)

            # TODO: Efficiently compute structural interfaces by precomputing each chain's most common molecule type
            # Find all interfaces defined as pairs of chains with minimum heavy atom (i.e. non-hydrogen) separation less than 5 Å
            # for atom in res:
            #     for neighbor in neighbor_search.search(atom.coord, 5.0, "R"):
            #         neighbor_one_letter_residue, neighbor_molecule_type = convert_ambiguous_residue_three_to_one(
            #             neighbor.resname, ccd_codes, molecule_type=chain_most_common_molecule_type
            #         )
            #         molecule_index_postfix = f"-{res.resname}-{num_ligands_in_chain}" if molecule_type == "ligand" else ""
            #         interface_chain_ids.add(f"{chain.id}:{molecule_type}{molecule_index_postfix}+{neighbor.get_parent().get_id()}:{neighbor_molecule_type}-{neighbor.resname}-{neighbor_num_ligands_in_chain}")

        assert (
            len(one_letter_seq_tokens) > 0
        ), f"No residues found in chain {chain.id} within the mmCIF file {filepath}."

        token_molecule_types = list(token_molecule_types)
        if len(token_molecule_types) > 1:
            assert (
                len(token_molecule_types) == 2
            ), f"More than two molecule types found ({token_molecule_types}) in chain {chain.id} within the mmCIF file {filepath}."
            molecule_type = [
                molecule_type
                for molecule_type in token_molecule_types
                if molecule_type != "unknown"
            ][0]
        elif len(token_molecule_types) == 1 and token_molecule_types[0] == "unknown":
            molecule_type = "protein"
        else:
            molecule_type = token_molecule_types[0]

        if (
            molecule_type == "protein"
            and len(one_letter_seq_tokens) < min_num_residues_for_protein_classification
        ):
            molecule_type = "peptide"

        one_letter_seq = "".join(one_letter_seq_tokens)
        sequences[f"{chain.id}:{molecule_type}"] = one_letter_seq

    return sequences, interface_chain_ids


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_directory(
    mmcif_dir: str, ccd_codes: Set[str]
) -> CHAIN_SEQUENCES:
    """
    Parse all mmCIF files in a directory and return a dictionary for each complex mapping chain IDs to sequences
    as well as a set of chain ID pairs denoting structural interfaces for each complex."""
    all_chain_sequences = []
    all_interface_chain_ids = []

    mmcif_filepaths = list(glob.glob(os.path.join(mmcif_dir, "*", "*.cif")))
    for cif_filepath in tqdm(
        mmcif_filepaths[:100], desc="Parsing chain sequences"
    ):  # TODO: remove loop length limit after development
        structure_id = os.path.splitext(os.path.basename(cif_filepath))[0]
        (
            chain_sequences,
            interface_chain_ids,
        ) = parse_chain_sequences_and_interfaces_from_mmcif_file(cif_filepath, ccd_codes)
        all_chain_sequences.append({structure_id: chain_sequences})
        all_interface_chain_ids.append({structure_id: interface_chain_ids})

    return all_chain_sequences, all_interface_chain_ids


@typecheck
def write_sequences_to_fasta(
    all_chain_sequences: CHAIN_SEQUENCES,
    fasta_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> List[str]:
    """Write sequences of a particular molecule type to a FASTA file, and return all molecule IDs."""
    assert fasta_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    fasta_filepath = fasta_filepath.replace(".fasta", f"_{molecule_type}.fasta")

    molecule_ids = []
    with open(fasta_filepath, "w") as f:
        for structure_chain_sequences in tqdm(
            all_chain_sequences, desc=f"Writing {molecule_type} FASTA chain sequence file"
        ):
            for structure_id, chain_sequences in structure_chain_sequences.items():
                for chain_id, sequence in chain_sequences.items():
                    chain_id_, molecule_type_ = chain_id.split(":")
                    molecule_type_name_and_index = molecule_type_.split("-")
                    if molecule_type_name_and_index[0] == molecule_type:
                        molecule_index_postfix = (
                            f"-{molecule_type_name_and_index[1]}-{molecule_type_name_and_index[2]}"
                            if len(molecule_type_name_and_index) == 3
                            else ""
                        )
                        molecule_id = f"{structure_id}{chain_id_}:{molecule_type_name_and_index[0]}{molecule_index_postfix}"

                        f.write(f">{molecule_id}\n{sequence}\n")
                        molecule_ids.append(molecule_id)
    return molecule_ids


@typecheck
def run_clustalo(
    input_filepath: str,
    output_filepath: str,
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
):
    """Run Clustal Omega on the input FASTA file and write the aligned FASTA sequences and corresponding distance matrix to respective output files."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert output_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    output_filepath = output_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")

    assert os.path.isfile(input_filepath), f"Input file '{input_filepath}' does not exist."

    subprocess.run(
        [
            "clustalo",
            "-i",
            input_filepath,
            "-o",
            output_filepath,
            f"--distmat-out={distmat_filepath}",
            "--percent-id",
            "--full",
            "--force",
        ]
    )


@typecheck
def cluster_ligands_by_ccd_code(input_filepath: str, distmat_filepath: str):
    """Cluster ligands based on their CCD codes and write the resulting sequence distance matrix to a file."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", "_ligand.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", "_ligand.txt")

    # Parse the ligand FASTA input file into a dictionary
    ligands = {}
    with open(input_filepath, "r") as f:
        structure_id = None
        for line in f:
            if line.startswith(">"):
                structure_id = line[1:].strip()
                ligands[structure_id] = ""
            else:
                ligands[structure_id] += line.strip()

    # Convert ligands to a list of tuples for easier indexing
    ligand_structure_ids = list(ligands.keys())
    ligand_sequences = list(ligands.values())
    n = len(ligand_structure_ids)

    # Initialize the distance matrix efficiently
    distance_matrix = np.zeros((n, n))

    # Fill the distance matrix using only the upper triangle (symmetric)
    for i in range(n):
        for j in range(i, n):
            if ligand_sequences[i] == ligand_sequences[j]:
                distance_matrix[i, j] = 100.0
                distance_matrix[j, i] = 100.0

    # Write the ligand distance matrix to a NumPy-compatible text file
    with open(distmat_filepath, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = [ligand_structure_ids[i]] + list(map(str, distance_matrix[i]))
            f.write(" ".join(row) + "\n")


@typecheck
def read_distance_matrix(
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> np.ndarray:
    """Read a distance matrix from a file and return it as a NumPy array."""
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")
    assert os.path.isfile(
        distmat_filepath
    ), f"Distance matrix file '{distmat_filepath}' does not exist."

    # Convert sequence matching percentages to distances through complementation
    df = pd.read_csv(distmat_filepath, sep="\s+", header=None, skiprows=1)
    matrix = 100.0 - df.values[:, 1:].astype(float)

    return matrix


@typecheck
def cluster_interfaces(
    chain_cluster_mapping: Dict[str, np.int64], interface_chain_ids: Set[str]
) -> Dict[tuple, set]:
    """Cluster interfaces based on the cluster IDs of the chains involved."""
    interface_clusters = defaultdict(set)

    interface_chain_ids = list(interface_chain_ids)
    for chain_id_pair in interface_chain_ids:
        chain_ids = chain_id_pair.split("+")
        chain_clusters = [chain_cluster_mapping[chain_id] for chain_id in chain_ids]
        if (chain_clusters[0], chain_clusters[1]) not in interface_clusters:
            interface_clusters[(chain_clusters[0], chain_clusters[1])] = f"{chain_id_pair}"

    return interface_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster chains and interfaces within the AlphaFold 3 PDB dataset's filtered mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "mmcifs"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "-c",
        "--ccd_dir",
        type=str,
        default=os.path.join("data", "ccd_data"),
        help="Path to the directory containing CCD files to reference during data clustering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "clusterings"),
        help="Path to the output FASTA file.",
    )
    args = parser.parse_args()

    # Determine paths for intermediate files

    fasta_filepath = os.path.join(args.output_dir, "sequences.fasta")
    aligned_fasta_filepath = os.path.join(
        os.path.dirname(fasta_filepath), "aligned_sequences.fasta"
    )
    distmat_filepath = os.path.join(args.output_dir, "distmat.txt")

    # Load the Chemical Component Dictionary (CCD) codes into memory

    logger.info("Loading the Chemical Component Dictionary (CCD) codes into memory...")
    ccd_codes = read_ccd_codes_from_pdb_components_file(
        os.path.join(args.ccd_dir, "components.cif")
    )
    logger.info("Finished loading the Chemical Component Dictionary (CCD) codes into memory.")

    # Parse all chain sequences from mmCIF files

    (
        all_chain_sequences,
        interface_chain_ids,
    ) = parse_chain_sequences_and_interfaces_from_mmcif_directory(args.mmcif_dir, ccd_codes)

    # Align sequences separately for each molecule type and compute each respective distance matrix

    protein_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="protein"
    )
    nucleic_acid_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="nucleic_acid"
    )
    peptide_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="peptide"
    )
    ligand_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="ligand"
    )

    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="protein",
    )
    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="nucleic_acid",
    )
    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="peptide",
    )
    cluster_ligands_by_ccd_code(
        fasta_filepath,
        distmat_filepath,
    )

    protein_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="protein")
    nucleic_acid_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="nucleic_acid")
    peptide_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="peptide")
    ligand_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="ligand")

    # Cluster residues at sequence homology levels corresponding to each molecule type

    protein_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=40.0 + 1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(protein_dist_matrix)

    nucleic_acid_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(nucleic_acid_dist_matrix)

    peptide_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(peptide_dist_matrix)

    ligand_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(ligand_dist_matrix)

    # Map chain sequences to cluster IDs, and save the mappings to local (CSV) storage

    protein_chain_cluster_mapping = dict(zip(protein_molecule_ids, protein_cluster_labels))
    nucleic_acid_chain_cluster_mapping = dict(
        zip(nucleic_acid_molecule_ids, nucleic_acid_cluster_labels)
    )
    peptide_chain_cluster_mapping = dict(zip(peptide_molecule_ids, peptide_cluster_labels))
    ligand_chain_cluster_mapping = dict(zip(ligand_molecule_ids, ligand_cluster_labels))

    pd.DataFrame(
        protein_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        nucleic_acid_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        peptide_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        ligand_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv"), index=False)

    # Cluster interfaces based on the cluster IDs of the chains involved, and save the interface cluster mapping to local (CSV) storage

    interface_cluster_mapping = cluster_interfaces(
        protein_chain_cluster_mapping, interface_chain_ids
    )

    pd.DataFrame(
        interface_cluster_mapping.items(), columns=["molecule_id_pair", "interface_cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "interface_cluster_mapping.csv"), index=False)