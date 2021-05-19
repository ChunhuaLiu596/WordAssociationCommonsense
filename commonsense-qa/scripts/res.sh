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
# input_files=(22700249 22700454)
# input_files=(22755409 22755432 22755433 22755492 22755553 22755738 22755761 22755765 22755736) 
# input_files=(22759730 22759851 22759652 22759654)
# input_files=(22769219 22769220 22769221 22769222 22769223 22769224 22769225 22769226)

# input_files=(22781508 22781509 22781510 22781511 22781512 22781513 22781514 22781515)
# input_files=(22804790 22804791 22804792 22804793 22804794 22804795 22804796 22804797)

# input_files=(22821159 22821160 22821161 22821162 22821163 22821164)
# input_files=(22828586 22828587 22829364 22829365)
# input_files=(22831488 22831489)
# input_files=(22842917 22842918)
# input_files=(22842799 22842800 22842801 22842802)
# input_files=(22851682 22851683 22851685 22851686)
# input_files=(22852970 22852971)
# input_files=(22891089 22891090 22891113 22891114)
# input_files=(22911905 22911906)
# input_files=(22966349)
# input_files=(23394609 23394610 23394611 23394612)
# input_files=(23515300 23509467 23515323)
# input_files=(23509490 23509491 22804793)
# input_files=(23509490)
# input_files=(23509467 23509490)
# input_files=(23607108 
# input_files=(23612156 23564790)
# input_files=(22768358 22768355 22768356 22768357)
# input_files=(23627151)
# input_files=(23638060 23638289 23638290 23637655 23637654)
# input_files=(22606658 22613561)
# input_files=(23692502 23692539)
# input_files=(23730420 23730696 23730303 23730474)
# input_files=(23761583 23761584 23761585 23761586 23761658 23761659)
input_files=(23774302)
n=3
# input_files=$1
# n=$2
# for input_file in input_files;do
for input_file in "${input_files[@]}";do
    echo $input_file
    python -u utils/extract_res.py --input_file $input_file --output_file $output_file --n $n
done 
