#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o validate.out

python validate.py config_validate.json -1 -1
