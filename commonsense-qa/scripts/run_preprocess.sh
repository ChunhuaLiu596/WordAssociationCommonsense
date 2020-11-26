#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

#only run kg preprocess
# python preprocess.py swow --run common
# python preprocess.py cpnet7rel --run common

#only run grounding
# python preprocess.py swow --run csqa
# python preprocess.py swow --run csqa
python preprocess.py cpnet7rel --run csqa -p 20