#!/bin/bash

# to manage pyscf's memory, restart python kernel for every calculations.
# place run_calc_from_xyzs.sh, calc_from_xyzs.py, .xyz files in same directory
# The cluster(.xyz) file name must include the names
#	and number of chemical species that involves in the cluster.
# run this file:
#	bash ./run_calc_from_xyzs.sh

#for xyz in "$1"/*.xyz ; do
for xyz in ./*.xyz ; do
    ./calc_from_xyzs.py "$xyz"
done
