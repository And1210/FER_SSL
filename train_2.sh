#!/bin/sh
#SBATCH -p Aurora
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train_2.out

python train.py config_fer_semi_2.json
