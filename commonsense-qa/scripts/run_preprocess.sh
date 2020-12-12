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
# python preprocess.py cpnet --run common
# python preprocess.py cpnet7rel --run common
# python preprocess.py cpnet1rel --run common
# python preprocess.py swow --run common
# python preprocess.py swow1rel --run common -p 20

##grounding: csqa
# python preprocess.py swow --run csqa
# python preprocess.py swow --run csqa
# python preprocess.py cpnet7rel --run csqa -p 20
# python preprocess.py cpnet1rel --run csqa -p 20
# python preprocess.py swow1rel --run csqa -p 20

##grounding: obqa
# python preprocess.py cpnet --run obqa -p 20
# python preprocess.py cpnet7rel --run obqa 
# python preprocess.py cpnet1rel --run obqa 
python preprocess.py swow1rel --run obqa
# python preprocess.py swow --run obqa

# python preprocess.py swow --run obqa -p 20