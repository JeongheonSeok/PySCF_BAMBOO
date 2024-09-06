#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:54:17 2024

@author: jeongheonseok
"""

import torch
import numpy as np
import os
import re
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

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

def total_charge_from_name(setname):
    """
    calculate charge from set name.\n
    check only Li and FSI.

    Parameters
    ----------
    set_name : string
        may contain Li and FSI.\n
        ex) Frame10_0-7_sub6-0_TFDMP5FSI1

    Returns
    -------
    charge : tensor(float)
        = #Li - #FSI

    """
    li_match = re.search(r'Li(\d+)', setname)
    fsi_match = re.search(r'FSI(\d+)', setname)
    N = float(li_match.group(1)) if li_match else 0
    M = float(fsi_match.group(1)) if fsi_match else 0
    charge = torch.tensor(N-M)
    return charge

def energy_from_logfile(log_path, first_line):
    """
    read energy from file('s first line) and returns the value in kcal/mol

    Parameters
    ----------
    log_path : String
        log file path. information for raising error.
    first_line : String
        first line of log file that includes energy information

    Raises
    ------
    ValueError
        If there are no energy section in firstline, raise value error

    Returns
    -------
    energy : torch.tensor(float)
        [kcal/mol], returns the energy value in kcal/mol

    """
    match = re.search(r'converged SCF energy = (-?\d+\.\d+)', first_line)
    if match:
        return torch.tensor(float(match.group(1)) * HARTREE_TO_KCALPMOL)
    else:
        raise ValueError("Energy section not found in", log_path)

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
    gradients_tensor : torch.tensor(#atoms * 3)
        returns the force(gradient) tensor in kcal/mol/A

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "--------------- DFRKS gradients ---------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("DFRKS gradients section not found in", log_path)
    
    gradients = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 5:  # idx + atom name + x + y + z
            x = float(parts[2]) * HARTREEPBOHR_TO_KCALPMOLPA
            y = float(parts[3]) * HARTREEPBOHR_TO_KCALPMOLPA
            z = float(parts[4]) * HARTREEPBOHR_TO_KCALPMOLPA
            gradients.append([x, y, z])
    
    gradients_tensor = torch.tensor(gradients, dtype=torch.float32)
    return gradients_tensor

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
    dipole_tensor : torch.tensor([float,float,float])
        dipole moment of cluster in Debye.

    """
    dipole_tensor = None
    for line in lines:
        if "Dipole moment(X, Y, Z, Debye)" in line:
            match = re.search(r"Dipole moment\(X, Y, Z, Debye\):\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)", line)
            if match:
                dipole_x = float(match.group(1))
                dipole_y = float(match.group(2))
                dipole_z = float(match.group(3))
                dipole_tensor = torch.tensor([dipole_x, dipole_y, dipole_z])
            else:
                raise ValueError("Dipole moment values not found in", log_path)
    if dipole_tensor == None:
        raise ValueError("Dipole moment section not found in", log_path)
    return dipole_tensor

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
    quadrupole_tensor : torch.tensor([float,float,float,float,float])
        Traceless quadrupole tensor element in Debye-Angstrom. [XX,XY,XZ,YY,YZ]

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "-------------- Quadrupole Moment--------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("Quadrupole Moment section not found in", log_path)
        
    quadrupole_array = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 3:
            quadrupole_array.append([float(parts[0]),float(parts[1]),float(parts[2])])
        else:
            raise ValueError("Wrong Quadrupole Moment format in", log_path)
    if len(quadrupole_array) != 3:
        raise ValueError("Wrong Quadrupole Moment format in", log_path)
    std_quadrupole_tensor = torch.tensor(quadrupole_array)
    trl = std_quadrupole_tensor - (torch.trace(std_quadrupole_tensor) / 3.0) * torch.eye(3)
    quadrupole_tensor = torch.tensor([trl[0][0],trl[0][1],trl[0][2],trl[1][1],trl[1][2]])
    return quadrupole_tensor

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
    charge_tensor : torch.tensor([float * #atoms])
        ChElPG charge in atomic unit.

    """
    start_idx = None
    for i, line in enumerate(lines):
        if "--------------- ChElPG Charges ---------------" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("ChElPG Charges section not found in", log_path)
        
    charge = []
    for line in lines[start_idx:]:
        if line.strip() == '----------------------------------------------':
            break
        parts = line.split()
        if len(parts) == 2:
            charge.append(float(parts[1]))
    charge_tensor = torch.tensor(charge)
    return charge_tensor

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
    atom_types_tensor : torch.tensor([int * #atoms])
        atomic numbers list (tensor).

    """
    atom_types= []
    for line in lines:
        atom_types.append(ELEMENTS[line.strip().split()[0]])
    atom_types_tensor = torch.tensor(atom_types)
    return atom_types_tensor

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
    pos_tensor : torch.tensor([float,float,float] * #atoms)
        position information.

    """
    pos = []
    for line in lines:
        line = line.strip().split()
        if len(line) != 4:
            raise ValueError("Wrong xyz format in", xyz_path)
        pos.append([float(line[1]),float(line[2]),float(line[3])])
    pos_tensor = torch.tensor(pos)
    return pos_tensor

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
    pairs_tensor : torch.tensor([[int,int] * #edges])
        edge index information.

    """
    distances = squareform(pdist(pos_tensor.numpy()))
    pairs = []
    for i in range(set_atom_num):
        for j in range(i+1, set_atom_num):
            if distances[i, j] <= cutoff:
                pairs.append([i,j])
                pairs.append([j,i])
    pairs_tensor = torch.tensor(pairs)
    return pairs_tensor

def get_all_atom_pairs(set_atom_num):
    """
    return all atom pairs : all_edge_index

    Parameters
    ----------
    set_atom_num : int
        # of atoms

    Returns
    -------
    pairs_tensor : torch.tensor([[int,int] * set_atom_num(set_atom_num-1)])
        all pairs of cluster.

    """
    atom_indices = list(range(set_atom_num))
    pairs = list(combinations(atom_indices, 2))
    pairs_tensor = torch.tensor(pairs)
    return pairs_tensor

def process_all_log_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".log"):
            setname = filename[:-4]
            log_path = os.path.join(folder_path, filename)
            xyz_path = os.path.join(folder_path, setname+".xyz")
            make_pt(setname, log_path, xyz_path, folder_path)

def read_n_make_dict(setname, log_path, xyz_path):
    set_dict = {
        'cell': torch.tensor([-1.,-1.,-1.]),
        'total_charge': total_charge_from_name(setname),  # a.u.
        'energy': torch.tensor(0.), # kcal/mol
        'virial': torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]), # kcal/mol/Angstrom
        'dipole': torch.tensor([0.,0.,0.]), # Debye
        'quadrupole': torch.tensor([0.,0.,0.,0.,0.]),   # Debye-Angstrom
        'atom_types': None,     # torch.tensor([int * #atoms]), atomic number
        'pos': None,            # torch.tensor([[float,float,float] * #atoms])
        'forces': None,         # torch.tensor([[float,float,float] * #atoms])
        'charge': None,         # torch.tensor([float * #atoms])
        'edge_index': None,     # torch.tensor([[float,float] * #edges(in 5 Angstrom)])
        'all_edge_index': None  # torch.tensor([[float,float] * #all_edges])
        # additional training dataset keys:
        #   cumsum_atom(#mol+1), mol_ids(#atom), cumsum_edge(#mol+1), cumsum_all_edge()
        }
    with open(log_path, 'r') as file:
        set_dict['energy'] = energy_from_logfile(log_path, file.readline().strip())
        lines = file.readlines()
    set_dict['forces'] = forces_from_logfile(log_path, lines)
    set_dict['dipole'] = dipole_from_logfile(log_path, lines)
    set_dict['quadrupole'] = quadrupole_from_logfile(log_path, lines)
    set_dict['charge'] = charge_from_logfile(log_path, lines)
    
    with open(xyz_path, 'r') as file:
        set_atom_num = int(file.readline().strip());     file.readline()
        lines = file.readlines()
        if set_atom_num != len(lines):
            raise ValueError("Wrong xyz format")
    set_dict['atom_types'] = atom_types_from_xyzfile(xyz_path, lines)
    set_dict['pos'] = pos_from_xyzfile(xyz_path, lines)
    
    set_dict['edge_index'] = edge_from_pos(set_dict['pos'], set_atom_num, cutoff=5.0)    # edges with cutoff radius 5 Angstrom
    set_dict['all_edge_index'] = get_all_atom_pairs(set_atom_num)
    
    virial = np.einsum('ij,ik->jk', set_dict['pos'].numpy(), set_dict['forces'].numpy())
    set_dict['virial'] = torch.from_numpy((virial + virial.T)/2)
    return set_dict
    
def make_pt(setname, log_path, xyz_path, folder_path):
    set_dict = read_n_make_dict(setname, log_path, xyz_path)
    torch.save(set_dict, setname+'.pt')
    print(setname+".pt","Saved")

folder_path = './'
process_all_log_files_in_folder(folder_path)
