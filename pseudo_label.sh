#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o pseudo_label.out

python pseudo_label.py config_fer_pseudo_label.json
