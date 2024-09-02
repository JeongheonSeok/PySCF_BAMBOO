#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  25 14:40:00 2024

@author: jeongheonseok

for all ~.xyz, ~.log files, read informations and dump .pkl file, zip to pkl.zip file.

ver 2 modifications:
    1. return as list. no torch.tensor
    2. gradint * -1
    3. energy = cluster_energy - atom_energy
"""
import pickle
import numpy as np
import os
import re
from tqdm import tqdm
import zipfile
from scipy.spatial.distance import pdist, squareform
from itertools import permutations

###############################################################################
# ION_LIST element: [ion name, charge]
ION_LIST = [['Li',1], ['FSI',-1]]
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
###############################################################################

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
ENERGY_ATOM = {key: value * HARTREE_TO_KCALPMOL for key, value in ENERGY_ATOM.items()}

def total_charge_from_name(setname):
    """
    calculate charge from set name.\n
    ION_LIST need. default: Li, FSI

    Parameters
    ----------
    set_name : string
        may contain Li and FSI(default).\n
        If other ion needed, please modify ION_LIST \n
        ex) Frame10_0-7_sub6-0_TFDMP5FSI1

    Returns
    -------
    charge : float
        ex) = #Li - #FSI

    """
    charge = 0.
    for ion, ion_charge in ION_LIST:
        ion_match = re.search(rf'{ion}(\d+)', setname)
    
        if ion_match:
            ion_count = float(ion_match.group(1))
            charge += ion_count * ion_charge
    return charge

def energy_from_logfile(log_path, atom_types, first_line):
    """
    read energy from file('s first line) and returns the value in kcal/mol

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    atom_types : list
        list contains atoms in cluster.
    first_line : String
        first line of log file that includes energy information

    Raises
    ------
    ValueError
        If there are no energy section in firstline, raise value error

    Returns
    -------
    energy : float
        [kcal/mol], returns the energy value in kcal/mol

    """
    match = re.search(r'converged SCF energy = (-?\d+\.\d+)', first_line)
    if match:
        energy = float(match.group(1)) * HARTREE_TO_KCALPMOL
        for atom in atom_types:
            energy += ENERGY_ATOM[atom]
        return energy
    else:
        raise ValueError("Energy section not found in "+log_path)

def forces_from_logfile(log_path, lines):
    """
    read log file, find force(gradient) information, return force in kcal/mol/A

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    lines : list(string), file lines
        log file lines

    Raises
    ------
    ValueError
        If there are no gradient section in file, raise value error

    Returns
    -------
    gradients_tensor : list([[float, float, float] * #atoms])
        returns the force(gradient) tensor in kcal/mol/A

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "--------------- DFRKS gradients ---------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("DFRKS gradients section not found in "+log_path)
    
    gradients = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 5:  # idx + atom name + x + y + z
            x = float(parts[2]) * HARTREEPBOHR_TO_KCALPMOLPA * -1
            y = float(parts[3]) * HARTREEPBOHR_TO_KCALPMOLPA * -1
            z = float(parts[4]) * HARTREEPBOHR_TO_KCALPMOLPA * -1
            gradients.append([x, y, z])
    
    return gradients

def dipole_from_logfile(log_path, lines):
    """
    read lines, return dipole moment tensor in Debye

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    lines : list(string), file lines
        log file lines

    Raises
    ------
    ValueError
        If no dipole moment section or not enough data, occurs error.

    Returns
    -------
    dipole_list : torch.tensor([float,float,float])
        dipole moment of cluster in Debye.

    """
    dipole_list = None
    for line in lines:
        if "Dipole moment(X, Y, Z, Debye)" in line:
            match = re.search(r"Dipole moment\(X, Y, Z, Debye\):\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)", line)
            if match:
                dipole_x = float(match.group(1))
                dipole_y = float(match.group(2))
                dipole_z = float(match.group(3))
                dipole_list = [dipole_x, dipole_y, dipole_z]
            else:
                raise ValueError("Dipole moment values not found in "+log_path)
    if dipole_list == None:
        raise ValueError("Dipole moment section not found in "+log_path)
    return dipole_list

def quadrupole_from_logfile(log_path, lines):
    """
    read lines, return traceless quadrupole moment 5 elements [XX,XY,XZ,YY,YZ] in Debye-Angstrom

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    lines : list(string), file lines
        log file lines.

    Raises
    ------
    ValueError
        If no quadrupole moment section or not enough data, occurs error.

    Returns
    -------
    quadrupole_list : [float,float,float,float,float]
        Traceless quadrupole moment values in Debye-Angstrom. [XX,XY,XZ,YY,YZ]

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "-------------- Quadrupole Moment--------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("Quadrupole Moment section not found in " + log_path)
        
    quadrupole_array = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 3:
            quadrupole_array.append([float(parts[0]), float(parts[1]), float(parts[2])])
        else:
            raise ValueError("Wrong Quadrupole Moment format in " + log_path)
    
    if len(quadrupole_array) != 3:
        raise ValueError("Wrong Quadrupole Moment format in " + log_path)
    
    quadrupole_tensor = np.array(quadrupole_array)

    trl = quadrupole_tensor - (np.trace(quadrupole_tensor) / 3.0) * np.eye(3)
    
    quadrupole_list = [trl[0, 0], trl[0, 1], trl[0, 2], trl[1, 1], trl[1, 2]]
    return quadrupole_list

def charge_from_logfile(log_path, lines):
    """
    read lines, return ChElPG charge in atomic unit

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    lines : list(string), file lines
        log file lines.

    Raises
    ------
    ValueError
        If no ChElPG charge section, occurs error.

    Returns
    -------
    charge : [float * #atoms]
        ChElPG charge in atomic unit.

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "--------------- ChElPG Charges ---------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("ChElPG Charges section not found in "+log_path)
        
    charge = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 2:
            charge.append(float(parts[1]))
    return charge

def atom_types_from_xyzfile(xyz_path, lines):
    """
    read atom types and returns atomic numbers list (tensor)

    Parameters
    ----------
    xyz_path : String
        xyz file path. information for raising error. (actually, here no error raising)
    lines : list(string), file lines
        xyz file lines.

    Returns
    -------
    atom_types : [int * #atoms]
        atomic numbers list.

    """
    atom_types= []
    for line in lines:
        atom_types.append(ELEMENTS[line.strip().split()[0]])
    return atom_types

def pos_from_xyzfile(xyz_path, lines):
    """
    read atom positions and returns position list (tensor)

    Parameters
    ----------
    xyz_path : String
        xyz file path. information for raising error. (actually, here no error raising)
    lines : list(string), file lines
        xyz file lines.

    Raises
    ------
    ValueError
        if wrong xyz format (especially for not enough coordinate informations).

    Returns
    -------
    pos : [[float,float,float] * #atoms]
        position information.

    """
    pos = []
    for line in lines:
        line = line.strip().split()
        if len(line) != 4:
            raise ValueError("Wrong xyz format in", xyz_path)
        pos.append([float(line[1]),float(line[2]),float(line[3])])
    return pos

def edge_from_pos(pos_tensor, set_atom_num, cutoff=5.0):
    """
    get edge information

    Parameters
    ----------
    pos_tensor : torch.tensor([[float,float,float] * #atoms])
        position information (obtained from xyz)
    set_atom_num : int
        # of atoms 
    cutoff : float
        cutoff radius in Angstrom. The default is 5.0.

    Returns
    -------
    pairs : [[int,int] * #edges]
        edge index information.

    """
    distances = squareform(pdist(np.array(pos_tensor)))
    pairs = []
    for i in range(set_atom_num):
        for j in range(i+1, set_atom_num):
            if distances[i, j] <= cutoff:
                pairs.append([i,j])
                pairs.append([j,i])
    return pairs

def get_all_atom_pairs(set_atom_num):
    """
    return all atom pairs : all_edge_index

    Parameters
    ----------
    set_atom_num : int
        # of atoms

    Returns
    -------
    pairs : [[int,int] * set_atom_num*(set_atom_num-1)]
        all pairs of cluster.

    """
    atom_indices = list(range(set_atom_num))
    pairs = list(permutations(atom_indices, 2))
    return pairs

def read_n_make_dict(setname, log_path, xyz_path):
    set_dict = {
        'cell': [-1.,-1.,-1.],
        'total_charge': total_charge_from_name(setname),  # a.u.
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
        
    with open(xyz_path, 'r') as file:
        set_atom_num = int(file.readline().strip());     file.readline()
        lines = file.readlines()
        if set_atom_num != len(lines):
            raise ValueError("Wrong xyz format")
    set_dict['atom_types'] = atom_types_from_xyzfile(xyz_path, lines)
    set_dict['pos'] = pos_from_xyzfile(xyz_path, lines)
    
    with open(log_path, 'r') as file:
        set_dict['energy'] = energy_from_logfile(log_path, set_dict['atom_types'], file.readline().strip())
        lines = file.readlines()
    set_dict['forces'] = forces_from_logfile(log_path, lines)
    set_dict['dipole'] = dipole_from_logfile(log_path, lines)
    set_dict['quadrupole'] = quadrupole_from_logfile(log_path, lines)
    set_dict['charge'] = charge_from_logfile(log_path, lines)
    
    set_dict['edge_index'] = np.array(edge_from_pos(set_dict['pos'], set_atom_num, cutoff=5.0))    # edges with cutoff radius 5 Angstrom
    set_dict['all_edge_index'] = np.array(get_all_atom_pairs(set_atom_num))
    
    virial = np.einsum('ij,ik->jk', np.array(set_dict['pos']), np.array(set_dict['forces']))
    set_dict['virial'] = ((virial + virial.T)/2).tolist()
    return set_dict
    
def make_pkl(setname, log_path, xyz_path, folder_path):
    set_dict = read_n_make_dict(setname, log_path, xyz_path)
    pkl_path = os.path.join(folder_path, setname + '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(set_dict, f)
    # print(setname+".pkl","Saved")
    
def zip_pkl_files(directory, output_filename='pkl.zip'):
    error_file_path = os.path.join(directory, "error_list.txt")
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in tqdm(filenames, desc="Make .zip file", unit="files"):
                if filename.endswith('.pkl'):
                    pkl_file_path = os.path.join(foldername, filename)
                    #xyz_filename = filename.replace('.pkl', '.xyz')
                    #xyz_file_path = os.path.join(foldername, xyz_filename)
                    zipf.write(pkl_file_path, os.path.relpath(pkl_file_path, directory))
                    #if os.path.exists(xyz_file_path):
                    #    zipf.write(xyz_file_path, os.path.relpath(xyz_file_path, directory))
                    os.remove(pkl_file_path)
        if os.path.isfile(error_file_path):
            zipf.write(error_file_path, os.path.relpath(error_file_path, directory))

def process_all_log_files_in_folder(folder_path):
    error_log_path = os.path.join(folder_path, 'error_list.txt')
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".log"):
            setname = filename[:-4]
            log_path = os.path.join(folder_path, filename)
            xyz_path = os.path.join(folder_path, setname+".xyz")
            
            try:
                make_pkl(setname, log_path, xyz_path, folder_path)
            except Exception as e:
                print(e)
                with open(error_log_path, 'a') as error_file:
                    error_file.write(str(e)+'\n')
    zip_pkl_files(folder_path)

folder_path = './'
process_all_log_files_in_folder(folder_path)
