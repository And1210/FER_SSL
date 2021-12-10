#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o generate_for_alt_gt.out

python generate_for_alt_gt.py config_fer_pseudo_label.json
