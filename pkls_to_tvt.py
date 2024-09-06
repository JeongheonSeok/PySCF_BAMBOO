#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:47:00 2024

@author: JeongHeonSeok

from files_path's ~.pkl files, make train_data.pt, val_data.pt, (test_data.pt) in out_path
"""

import pickle
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import gc

###############################################################################
random.seed(42)
files_path = './DET'
out_path = './sorted_pts'
(train, val, test) = (9, 1, 0)
###############################################################################
GARBAGE_COLLECTION_PERIOD = 5000

def list_pkl_files(folder_path):
    """
    Finds ~.pkl files in folder_path and returns a list of their paths.

    Parameters
    ----------
    folder_path : string
        folder path to find ~.pkl files

    Raises
    ------
    FileNotFoundError
        if no .pkl file in folder_path, raise error.

    Returns
    -------
    pkl_files : list of string
        list of ~.pkl files' paths.

    """
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    if len(pkl_files) == 0:
        raise FileNotFoundError("No .pt files in "+folder_path)
    return pkl_files

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
        contains ~.pkl files' directory.

    Raises
    ------
    ValueError
        if files list empty, raise ValueError.

    Returns
    -------
    file_dict : dictionary
        merged one dictionary.

    """
    gc_idx = 0
    if len(files) == 0: raise ValueError("no files to merge")
    with open(files.pop(0), 'rb') as f:
        cluster_dict = pickle.load(f)
    cluster_atom_num = len(cluster_dict['atom_types'])
    cs_atom = cluster_atom_num
    cs_edge = len(cluster_dict['edge_index'])
    cs_all_edge = len(cluster_dict['all_edge_index'])
    mol_i = 0
    
    file_dict = {
        'cell': [cluster_dict['cell']],                             # [[-1,-1,-1] * #clusters]
        'total_charge': [cluster_dict['total_charge']],             # [float * #clusters] a.u.
        'energy': [cluster_dict['energy']],                         # [float * #clusters] kcal/mol
        'virial': [cluster_dict['virial']],                         # [[3*3 rank2 tensor] * #clusters] kcal/mol/Angstrom
        'dipole': [cluster_dict['dipole']],                         # [[float,float,float] * #clusters] Debye
        'quadrupole': [cluster_dict['quadrupole']],                 # [[float * 5] * #clusters] Debye-Angstrom
        'atom_types': cluster_dict['atom_types'],                   # [int * #atoms], atomic number
        'pos': cluster_dict['pos'],                                 # [[float,float,float] * #atoms]
        'forces': cluster_dict['forces'],                           # [[float,float,float] * #atoms]
        'charge': cluster_dict['charge'],                           # [float * #atoms]
        'edge_index': cluster_dict['edge_index'].tolist(),          # [[float,float] * #edges(in 5 Angstrom)]
        'all_edge_index': cluster_dict['all_edge_index'].tolist(),  # [[float,float] * #all_edges]
        'cumsum_atom': [0],                                         # [int * #clusters]
        'cumsum_edge': [0],                                         # [int * #clusters]
        'cumsum_all_edge': [0],                                     # [int * #clusters]
        'mol_ids': [0] * cluster_atom_num                           # [int * #atoms]
        }
    
    for file in tqdm(files):
    # Load cluster dictionary from file
        with open(file, 'rb') as f:
            cluster_dict = pickle.load(f)
        cluster_atom_num = len(cluster_dict['atom_types'])

    # 'cell', 'total_charge', 'energy', 'virial', 'dipole', 'quadrupole' add
        for key in ['cell','total_charge','energy','virial','dipole','quadrupole']:
            file_dict[key].append(cluster_dict[key])
    # 'charge', 'atom_types', 'pos', 'forces' add
        for key in ['charge','atom_types','pos','forces']:
            file_dict[key].extend(cluster_dict[key])
    # 'edge_index', 'all_edge_index' add
        for key in ['edge_index', 'all_edge_index']:
            cluster_dict[key] += cs_atom
            file_dict[key].extend(cluster_dict[key].tolist())
    # 'cumsum_atom' add
        file_dict['cumsum_atom'].append(cs_atom)
    # 'cumsum_edge' add
        file_dict['cumsum_edge'].append(cs_edge)
    # 'cumsum_all_edge' add
        file_dict['cumsum_all_edge'].append(cs_all_edge) 
    # 'mol_ids' add
        mol_i += 1
        for i in range(cluster_atom_num):
            file_dict['mol_ids'].append(mol_i)
    # modify cs_atom, cs_edge, cs_all_edge
        cs_atom += cluster_atom_num
        cs_edge += len(cluster_dict['edge_index'])
        cs_all_edge += len(cluster_dict['all_edge_index'])
        
    # delete finished custer_dict
        del cluster_dict
        gc_idx += 1
        if gc_idx % GARBAGE_COLLECTION_PERIOD == 0:
            gc.collect()
    
# 'cumsum_atom' add
    file_dict['cumsum_atom'].append(cs_atom)
# 'cumsum_edge' add
    file_dict['cumsum_edge'].append(cs_edge)
# 'cumsum_all_edge' add
    file_dict['cumsum_all_edge'].append(cs_all_edge)
    
# list dictionary -> torch.tensor dictionary
    file_dict = {key: torch.tensor(value) for key, value in file_dict.items()}
    
    return file_dict

def make_train_val_test_pt(files_path, out_path, train=9, val=1, test=0):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if train == 0:
        raise ValueError("train ratio must not be 0")
    
    pkl_files = list_pkl_files(files_path)
    [train_files, val_files, test_files] = split_list_by_ratio(pkl_files, [train, val, test])
    
    train_dict = merge_to_one_dict(train_files)
    torch.save(train_dict, out_path+"/train_data.pt")
    if val != 0:
        val_dict = merge_to_one_dict(val_files)
        torch.save(val_dict, out_path+"/val_data.pt")
    if test != 0:
        test_dict = merge_to_one_dict(test_files)
        torch.save(test_dict, out_path+"/test_data.pt")

make_train_val_test_pt(files_path, out_path, train, val, test)
