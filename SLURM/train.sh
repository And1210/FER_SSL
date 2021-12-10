#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train.out

python train.py config_fer_semi.json
