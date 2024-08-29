#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:50:19 2024

@author: jeongheonseok

for .xyz file (get from sys.argv[1]), DFT-rks computation performed.
B3LYP, def2-svpd. The input cluster is assumed to have no unpaired electrons.
"""

import pyscf
from gpu4pyscf import grad
from gpu4pyscf.dft import rks
from pyscf.tools import molden
from gpu4pyscf.qmmm import chelpg
import time
from datetime import datetime
import re
import sys

###############################################################################
# ION_LIST element: [ion name, charge]
ION_LIST = [['Li',1], ['FSI',-1]]
###############################################################################

def charge_from_xyz(setname):
    """
    calculate charge from set name.\n
    ION_LIST need. default: Li, FSI

    Parameters
    ----------
    set_name : string
        may contain Li and FSI(default).\n
        If other ion needed, please modify ION_LIST \n
        ex) Frame10_0-7_sub6-0_TFDMP5FSI1 -> this function check TFDMP5FSI1 part

    Returns
    -------
    charge : int
        ex) = #Li - #FSI

    """
    charge = 0
    for ion, ion_charge in ION_LIST:
        ion_match = re.search(rf'{ion}(\d+)', setname)
    
        if ion_match:
            ion_count = float(ion_match.group(1))
            charge += ion_count * ion_charge
    return charge

def calc_from_xyz(file_name):
    print("------------------------------------------------------------------")
    print("Processing ",file_name)
    print("Start: ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    charge = charge_from_xyz(file_name)
    spin = 0
    mol = pyscf.M(
        atom=file_name,
        basis='def2-svpd',
        output=file_name[:-4]+'.log',
        charge=charge,
        spin=spin,
        verbose=3)
    mf_GPU = rks.RKS(
        mol,
        xc='b3lyp').density_fit()

    mf_GPU.with_df.auxbasis = 'def2-universal-jkfit'
    
    mf_GPU.conv_tol = 1e-10
    mf_GPU.max_cycle = 100
    
    st = time.time()
    mf_GPU.kernel()
    # print(f"0 total energy = {mf_GPU.e_tot}")
    print(file_name," SP time: ",time.time()-st)

    """
    st = time.time()
    molden_fname = file_name[:-4]+'.molden'
    with open(molden_fname, 'w') as f:
        molden.header(mol, f)
        molden.orbital_coeff(mol, f, mf_GPU.mo_coeff, ene=mf_GPU.mo_energy, occ=mf_GPU.mo_occ)
    print(file_name," molden dump time: ",time.time()-st)
    """

    st = time.time()
    gradients = grad.RKS(mf_GPU).kernel()
    # print(f"0 gradients = {gradients}")
    print(file_name," grad time: ",time.time()-st)
    
    st = time.time()
    dm = mf_GPU.make_rdm1()
    dip = mf_GPU.dip_moment(unit='DEBYE', dm=dm.get())
    """
    with open(file_name[:-4]+'.log', 'a') as file:
        file.write("---------------- Dipole Moment----------------\n")
        formatted_items = " ".join(f"{item:.10g}" for item in dip)
        file.write(formatted_items + "\n")
        file.write("----------------------------------------------\n")
    """
    quad = mf_GPU.quad_moment(unit='DEBYE-ANG', dm=dm.get())
    with open(file_name[:-4]+'.log', 'a') as file:
        file.write("-------------- Quadrupole Moment--------------\n")
        for row in quad:
            formatted_row = " ".join(f"{item:.10g}" for item in row)
            file.write(formatted_row + "\n")
        file.write("----------------------------------------------\n")
    print(file_name," multipole time: ",time.time()-st)

    st = time.time()
    q = chelpg.eval_chelpg_layer_gpu(mf_GPU)
    with open(file_name[:-4]+'.log', 'a') as file:
        file.write("--------------- ChElPG Charges ---------------\n")
        for index, item in enumerate(q):
            file.write(f"{index}\t{item}\n")
        file.write("----------------------------------------------\n")
    print(file_name," ChElPG time: ",time.time()-st)
    print("Finish: ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

"""    
original_stdout = sys.stdout

with open('run_output', 'w') as f:
    sys.stdout = f

    folder_path = "./"
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xyz'):
            # print(file_name)
            calc_from_xyz(file_name)
    
    sys.stdout = original_stdout
"""

file_name = sys.argv[1]        
calc_from_xyz(file_name)
