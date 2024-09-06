# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:37:03 2024

@author: JeongHeonSeok

from files_path's ~.pt files, make train_data.pt, val_data.pt, (test_data.pt) in out_path
"""

import torch
import os
import random
from tqdm import tqdm

###############################################################################
random.seed(42)
files_path = './DET'
out_path = './sorted_pts'
(train, val, test) = (9, 1, 0)

energy_atom = {
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
###############################################################################
HARTREE_TO_KCALPMOL = 627.509474
energy_atom = {key: value * HARTREE_TO_KCALPMOL for key, value in energy_atom.items()}

def modifications(cluster_dict):
    """
    modifications
        1. Energy: need to substract atom's energy.
        2. forces: need to multiply -1. (previous, just used gradient value)
        3. virial: force changed, virial also need to be changed. just multiply -1.
    """
    e = cluster_dict['energy'].item()
    for i in cluster_dict['atom_types'].tolist():
        e += energy_atom[i]
    cluster_dict['energy'] = torch.tensor(e)
    cluster_dict['forces'] = cluster_dict['forces'] * -1
    cluster_dict['virial'] = cluster_dict['virial'] * -1
    
    return cluster_dict

def list_pt_files(folder_path):
    """
    Finds ~.pt files in folder_path and returns a list of their paths.

    Parameters
    ----------
    folder_path : string
        folder path to find ~.pt files

    Raises
    ------
    FileNotFoundError
        if no .pt file in folder_path, raise error.

    Returns
    -------
    pt_files : list of string
        list of ~.pt files' paths.

    """
    pt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    if len(pt_files) == 0:
        raise FileNotFoundError("No .pt files in "+folder_path)
    return pt_files

def split_list_by_ratio(lst, ratios):
    """
    Randomly shuffle the order of the lst, divide it according to the ratios, and return it.

    Parameters
    ----------
    lst : list
        list to divide.
    ratios : list [number, number, ...]
        list of divide ratios. any positive numbers
    
    Raises
    ------
    ValueError
        If any of the values ​​is negative, raise error.
    
    Returns
    -------
    split_lists : [list, list, ...]
        list of divided lists.

    """
    if any(i < 0 for i in ratios): raise ValueError("ratio needs to be positive")

    random.shuffle(lst)
    total_ratio = sum(ratios)
    lengths = [int(len(lst) * (ratio / total_ratio)) for ratio in ratios]
    
    split_lists = []
    current_index = 0
    for length in lengths:
        split_lists.append(lst[current_index:current_index + length])
        current_index += length
    
    if current_index < len(lst):
        split_lists[-1].extend(lst[current_index:])
    
    return split_lists

def merge_to_one_dict(files):
    """
    from files list's .pt files, merge them to one dictionary

    Parameters
    ----------
    files : list of string
        contains ~.pt files' directory.

    Raises
    ------
    ValueError
        if files list empty, raise ValueError.

    Returns
    -------
    file_dict : dictionary
        merged one dictionary.

    """
    if len(files) == 0: raise ValueError("no files to merge")
    cluster_dict = modifications(torch.load(files.pop(0), weights_only=False))
    cluster_atom_num = len(cluster_dict['atom_types'])
    cs_atom = cluster_atom_num
    cs_edge = len(cluster_dict['edge_index'])
    cs_all_edge = len(cluster_dict['all_edge_index'])
    mol_i = 0
    
    file_dict = {
        'cell': cluster_dict['cell'].unsqueeze(0),                  # torch.tensor([[-1,-1,-1] * #clusters])
        'total_charge': cluster_dict['total_charge'].unsqueeze(0),  # torch.tensor([float * #clusters]) a.u.
        'energy': cluster_dict['energy'].unsqueeze(0),              # torch.tensor([float * #clusters]) kcal/mol
        'virial': cluster_dict['virial'].unsqueeze(0),              # torch.tensor([3*3 rank2 tensor] * #clusters) kcal/mol/Angstrom
        'dipole': cluster_dict['dipole'].unsqueeze(0),              # torch.tensor([[float,float,float] * #clusters]) Debye
        'quadrupole': cluster_dict['quadrupole'].unsqueeze(0),      # torch.tensor([[float * 5] * #clusters]) Debye-Angstrom
        'atom_types': cluster_dict['atom_types'],                   # torch.tensor([int * #atoms]), atomic number
        'pos': cluster_dict['pos'],                                 # torch.tensor([[float,float,float] * #atoms])
        'forces': cluster_dict['forces'],                           # torch.tensor([[float,float,float] * #atoms])
        'charge': cluster_dict['charge'],                           # torch.tensor([float * #atoms])
        'edge_index': cluster_dict['edge_index'],                   # torch.tensor([[float,float] * #edges(in 5 Angstrom)])
        'all_edge_index': cluster_dict['all_edge_index'],           # torch.tensor([[float,float] * #all_edges])
        'cumsum_atom': torch.tensor([0]),                           # torch.tensor([int * #clusters])
        'cumsum_edge': torch.tensor([0]),                           # torch.tensor([int * #clusters])
        'cumsum_all_edge': torch.tensor([0]),                       # torch.tensor([int * #clusters])
        'mol_ids': torch.zeros(cluster_atom_num, dtype=torch.int)   # torch.tensor([int * #atoms])
        }
    
    for file in tqdm(files):
    # Load cluster dictionary from file
        cluster_dict = modifications(torch.load(file, weights_only=False))
        cluster_atom_num = len(cluster_dict['atom_types'])

    # 'cell', 'total_charge', 'energy', 'virial', 'dipole', 'quadrupole' add
        for key in ['cell','total_charge','energy','virial','dipole','quadrupole']:
            file_dict[key] = torch.cat((file_dict[key], cluster_dict[key].unsqueeze(0)))
    # 'charge', 'atom_types', 'pos', 'forces', 'edge_index', 'all_edge_index' add
        cluster_dict['edge_index'] += cs_atom;  cluster_dict['all_edge_index'] += cs_atom
        for key in ['charge','atom_types','pos','forces','edge_index','all_edge_index']:
            file_dict[key] = torch.cat((file_dict[key], cluster_dict[key]))
    # 'cumsum_atom' add
        file_dict['cumsum_atom'] = torch.cat((file_dict['cumsum_atom'], torch.tensor([cs_atom])))
    # 'cumsum_edge' add
        file_dict['cumsum_edge'] = torch.cat((file_dict['cumsum_edge'], torch.tensor([cs_edge])))
    # 'cumsum_all_edge' add
        file_dict['cumsum_all_edge'] = torch.cat((file_dict['cumsum_all_edge'], torch.tensor([cs_all_edge])))    
    # 'mol_ids' add
        mol_i += 1
        file_dict['mol_ids'] = torch.cat((file_dict['mol_ids'], torch.full((cluster_atom_num,), mol_i)))
        
    # modify cs_atom, cs_edge, cs_all_edge
        cs_atom += cluster_atom_num
        cs_edge += len(cluster_dict['edge_index'])
        cs_all_edge += len(cluster_dict['all_edge_index'])
    
# 'cumsum_atom' add
    file_dict['cumsum_atom'] = torch.cat((file_dict['cumsum_atom'], torch.tensor([cs_atom])))
# 'cumsum_edge' add
    file_dict['cumsum_edge'] = torch.cat((file_dict['cumsum_edge'], torch.tensor([cs_edge])))
# 'cumsum_all_edge' add
    file_dict['cumsum_all_edge'] = torch.cat((file_dict['cumsum_all_edge'], torch.tensor([cs_all_edge])))    
    
    return file_dict

def make_train_val_test_pt(files_path, out_path, train=9, val=1, test=0):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    pt_files = list_pt_files(files_path)
    [train_files, val_files, test_files] = split_list_by_ratio(pt_files, [train, val, test])
    
    train_dict = merge_to_one_dict(train_files)
    torch.save(train_dict, out_path+"/train_data.pt")
    if val != 0:
        val_dict = merge_to_one_dict(val_files)
        torch.save(val_dict, out_path+"/val_data.pt")
    if test != 0:
        test_dict = merge_to_one_dict(test_files)
        torch.save(test_dict, out_path+"/test_data.pt")

make_train_val_test_pt(files_path, out_path, train, val, test)