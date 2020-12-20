#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn


output_file="./log/auto_extract_results.md"
# input_files=(22585606 22606219 22606616 22606655 22606658 22613565 22613564 22613561 )
input_files=(22700249 22700454)
# for input_file in input_files;do
for input_file in "${input_files[@]}";do
    echo $input_file
    python -u utils/extract_res.py --input_file $input_file --output_file $output_file
done 
