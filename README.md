# PySCF_BAMBOO
Codes to generate dataset for training BAMBOO model (train.pt, val.pt)

BAMBOO MLFF model:
Gong, Sheng, et al. "BAMBOO: a predictive and transferable machine learning force field framework for liquid electrolyte development." arXiv preprint arXiv:2404.07181 (2024). https://arxiv.org/abs/2404.07181
https://github.com/bytedance/bamboo

The DFT dataset for training was created with gpu4pyscf and pyscf. → pyscf, gpu4pyscf, and related Python packages were installed and used in a Linux environment.
https://github.com/pyscf/pyscf
https://github.com/pyscf/gpu4pyscf
Xiaojie Wu et al. Python-Based Quantum Chemistry Calculations with GPU Acceleration. 2024.
arXiv: 2404.09452 [physics.comp-ph].
https://arxiv.org/abs/2404.09452

## 0. Generating structures for DFT calculations
Perform Classical MD, extract random clusters from some of these snapshots, and use them as data.

Extract cluster structures as .xyz files. The file name must include the names and numbers of ions and molecules that make up the cluster. (ex: Frame10_0-7_sub6-0_TFDMP5FSI1.xyz)

## 1. DFT computation
DFT computation conditions: Restricted Kohn-Sham, B3LYP, def2-svpd, Density Fitting, auxiliary basis: def2-universal_jkfit, SCF convergence threshold 1.0 10^-10 a.u., maximum iteration 100.

   The code was written on the assumption that all electrons were paired for convenience of calculation. If you want to consider the system which have non-zero total spin, It can be used by modifying "rks(RKS)" in line 14, 67 to "uks(UKS)", and "spin" value in line 59 to the spin value of the system.
   
1. Place the .xyz files, calc_from_xyzs.py and run_calc_from_xyzs.sh files in the same directory.
2. Modify calc_from_xyzs.py's ION_LIST appropriately to the system on which it will be calculated.

   Uploaded files are saved as if only Li+, FSI- are ions (charged particle).
   
   The ion name must be same as the name used in the .xyz file, and substance names not included in ION_LIST are considered as neutral molecules.
   
   If you do not input this accurately, errors may occur at DFT calculation.
   
   If you want to get wave function information, you can uncomment the commented part at line 15 and 81 to generate a .molden file.
3. Modify run_calc_from_xyzs.sh appropriately to suit your computation environment. This file contains only the basic contents.
4. If you run below script in the terminal, DFT calculations will be performed on all xyz files in that folder.
```bash
bash ./run_calc_from_xyzs.sh
```
5. After the calculation, a .log file with the same name as the .xyz files are created. That file contains information about the energy, gradient, dipole moment, quadrupole moment, and ChElPG charge of the cluster.

## 2. Convert DFT computation result to BAMBOO training data
24/09/02 Changes: In log_to_zip.py, the atom reference energy is used as the value received from Mu ZhenLiang, author of BAMBOO on GitHub.

Training and validation of the BAMBOO model requires collecting DFT calculation results (.log files) and converting them into .pt files.

log_to_zip.py: Store the DFT computation results of each cluster in a .pkl file, and collect and compress them into a zip file.

pkls_to_tvt.py: Read the .pkl files containing information about each cluster and create .pt files from them for BAMBOO training.
1. Place log_to_zip.py in the folder where the .log files are located.
2. Modify log_to_zip.py's ION_LIST appropriately to suit your system. This is same as calc_from_xyzs.py.
3. After running log_to_zip.py, a pkl.zip file is created.
```bash
python ./log_to_zip.py
```
4. Put all the pkl files created this way into one folder.
5. In the pkls_to_tvt.py file, appropriately input the random seed, the path where the pkl files are stored, the path where the .pt files for BAMBOO training will be created, and the train val test ratio.
6. When you run pkls_to_tvt.py, train.pt and val.pt(+test.pt) are created in the out_path you entered.
```bash
python ./run_pkls_to_tvt.py
```

## 3. BAMBOO training
0. Install the BAMBOO (https://github.com/bytedance/bamboo)
1. Make the 'data'folder at BAMBOO folder, Put the train.pt and val.pt files created above into that folder.
2. Modify ~/configs/train_config/config.json under the BAMBOO folder as appropriate for your desired learning.
3. Move the terminal working directory to the BAMBOO folder and proceed with model training with the command below.
```bash
python -m train.train --config configs/train_config/config.json
```
4. Learned model parameters and logs are generated in train/bamboo_community_train/ under the BAMBOO folder. Checkpoint .pts are generated in the checkpoints folder, and train logs are generated in the train_logs folder.
