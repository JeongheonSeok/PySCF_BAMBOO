#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:19:41 2024

@author: JeongHeonSeok
"""

import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
import glob

filedir = './'
outdir='./pkls'

if not os.path.exists(outdir):
    os.makedirs(outdir)

HARTREE_TO_KCALPMOL = 627.509474
HARTREEPBOHR_TO_KCALPMOLPA = 1185.82105
ELEMENTS = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}   # made by ChatGPT
ENERGY_ATOM = {
    # BAMBOO author; Mu ZhenLiang's information.
    # in Hartree.
    1: -0.5004938956785162,   # H
    2: -2.9012748724514967,   # He
    3: -7.478985306972282,    # Li
    4: -14.661834274696954,   # Be
    5: -24.653654806556624,   # B
    6: -37.782763286104576,   # C
    7: -54.48354226674843,    # N
    8: -74.96961419577514,    # O
    9: -99.73161510485858,    # F
    10: -128.92519289074954,  # Ne
    11: -162.23766720181578,  # Na
    12: -200.02329093220146,  # Mg
    13: -242.3123918238669,   # Al
    14: -289.26770687962454,  # Si
    15: -341.1337984084236,   # P
    16: -397.97997813506527,  # S
    17: -460.0658512283677,   # Cl
    18: -527.436350036388,    # Ar
    19: -599.8218494548689,   # K
    20: -677.4588120804531,   # Ca
    21: -760.4971316009487,   # Sc
    22: -849.1596725847471,   # Ti
    23: -943.6254146277051,   # V
    24: -1044.1113296220058,  # Cr
    25: -1150.5221511684135,  # Mn
    26: -1263.3770208466553,  # Fe
    27: -1382.4595630344677,  # Co
    28: -1508.0541977393611,  # Ni
    29: -1640.3216679894867,  # Cu
    30: -1779.2191706307078,  # Zn
    31: -1924.6219755997204,  # Ga
    32: -2076.6814293344805,  # Ge
    33: -2235.539562658915,   # As
    34: -2401.212826084795,   # Se
    35: -2573.8736990611887,  # Br
    36: -2753.5141929669835   # Kr
}
HARTREE_TO_KCALPMOL = 627.509474
ENERGY_ATOM = {key: value * HARTREE_TO_KCALPMOL for key, value in ENERGY_ATOM.items()}

def ptdict_to_pkldict(ptdir):
    ptdict = torch.load(ptdir, weights_only=False)
    pkldict = {
        'cell': [-1.,-1.,-1.],
        'total_charge': 0.,  # a.u.
        'energy': 0., # kcal/mol
        'virial': [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], # kcal/mol/Angstrom
        'dipole': [0.,0.,0.], # Debye
        'quadrupole': [0.,0.,0.,0.,0.],   # Debye-Angstrom
        'atom_types': None,     # [int * #atoms], atomic number
        'pos': None,            # [[float,float,float] * #atoms]
        'forces': None,         # [[float,float,float] * #atoms]
        'charge': None,         # [float * #atoms]
        'edge_index': None,     # np.array([[float,float] * #edges(in 5 Angstrom)])
        'all_edge_index': None  # np.array([[float,float] * #all_edges]
        # additional training dataset keys:
        #   cumsum_atom(#mol+1), mol_ids(#atom), cumsum_edge(#mol+1), cumsum_all_edge()
        }
    for key in ptdict.keys():
        pkldict[key] = ptdict[key].tolist()
    for key in ['forces', 'virial', 'edge_index', 'all_edge_index']:
        pkldict[key] = np.array(pkldict[key])
    
    pkldict['forces'] = pkldict['forces'] * -1
    pkldict['forces'] = pkldict['forces'].tolist()
    pkldict['virial'] = pkldict['virial'] * -1
    pkldict['virial'] = pkldict['virial'].tolist()
    
    energy = pkldict['energy']
    for atom in pkldict['atom_types']:
        energy -= ENERGY_ATOM[atom]
    pkldict['energy'] = energy
    
    return pkldict

def mk_pkl_from_pt(filedir='./', outdir='./pkls'):
    pt_files = glob.glob(os.path.join(filedir, '*.pt'))
    for ptdir in tqdm(pt_files):
        pkldict = ptdict_to_pkldict(ptdir)
        filename = os.path.basename(ptdir).replace('.pt','.pkl')
        pkldir = os.path.join(outdir, filename)
        with open(pkldir, 'wb') as f:
            pickle.dump(pkldict, f)

mk_pkl_from_pt(filedir, outdir)
