#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn


# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow gconattn -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow rn -J albert 

# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_global  -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full -J albert


# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel gconattn  -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel rn  -J albert

# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel pg_global  -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel pg_full -J albert

: << COMMENT
sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel gconattn -J albert 
sbatch ./scripts/run_main.sh config/csqa_17rel.config swow gconattn -J albert 

sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel pg_global -J albert 
sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_global -J albert 
COMMENT

# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel gconattn -J bert-base-uncased 
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow gconattn -J bert-base-uncased

# sbatch ./scripts/run_main.sh config/obqa.config cpnet17rel gconattn  -J bert-base-uncased
# sbatch ./scripts/run_main.sh config/obqa.config cpnet17rel rn  -J bert-base-uncased

# sbatch ./scripts/run_main.sh config/obqa.config cpnet17rel pg_global  -J bert-base-uncased
# sbatch ./scripts/run_main.sh config/obqa.config cpnet17rel pg_full -J bert-base-uncased


# sbatch ./scripts/run_main.sh config/obqa.config swow gconattn -J bert-base-uncased
# sbatch ./scripts/run_main.sh config/obqa.config swow rn -J bert-base-uncased 

# sbatch ./scripts/run_main.sh config/obqa.config swow pg_global  -J bert-base-uncased
# sbatch ./scripts/run_main.sh config/obqa.config swow pg_full -J bert-base-uncased


# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet17rel pg_full swow -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full cpnet -J albert
# 
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet1rel pg_full swow -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow1rel pg_full cpnet -J albert

# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet_swow pg_full cpnet_swow -J albert-transe19rel
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet_swow rn cpnet_swow -J albert-transe19rel

# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet None cpnet -J roberta-large-lm

# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full swow train 11010 0 -J albert

# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet_swow pg_global cpnet_swow -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet pg_full cpnet_swow -J albert



### OBQA ##### 
# sbatch ./scripts/run_main.sh config/obqa.config cpnet_swow pg_global cpnet_swow 
# sbatch ./scripts/run_main.sh config/obqa.config cpnet_swow rn cpnet_swow 
# sbatch ./scripts/run_main.sh config/obqa.config cpnet_swow pg_full cpnet_swow 


# sbatch ./scripts/run_main.sh config/obqa.config cpnet pg_full cpnet  
# sbatch ./scripts/run_main.sh config/obqa.config cpnet rn cpnet 
# sbatch ./scripts/run_main.sh config/obqa.config swow pg_full swow 

# sbatch ./scripts/run_main.sh config/obqa.config cpnet7rel pg_full cpnet
# sbatch ./scripts/run_main.sh config/obqa.config cpnet7rel rn cpnet

# sbatch ./scripts/run_main.sh config/obqa.config cpnet1rel pg_full cpnet
# sbatch ./scripts/run_main.sh config/obqa.config cpnet1rel rn cpnet

# sbatch ./scripts/run_main.sh config/obqa.config cpnet pg_full swow 
# sbatch ./scripts/run_main.sh config/obqa.config swow pg_full cpnet 

##### test ######
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet pg_full cpnet pred 0 0 -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet pg_full cpnet pred 23528 3  -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet pg_full cpnet pred 24524 2 -J albert

# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full swow pred 0 0 -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full swow pred 19286 0 -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config swow pg_full swow pred 11010 0 -J albert
# sbatch ./scripts/run_main.sh config/csqa_17rel.config cpnet pg_full cpnet pred 21527 1 -J albert

# sbatch ./scripts/run_test.sh config/obqa.config cpnet pg_full cpnet pred 8625 1 -J albert
# sbatch ./scripts/run_test.sh config/obqa.config cpnet pg_full cpnet pred 14846 5 -J albert
# sbatch ./scripts/run_test.sh config/obqa.config cpnet pg_full cpnet pred 5863 4 -J albert

# sbatch ./scripts/run_test.sh config/obqa.config swow pg_full swow pred 10806 3 -J albert
# sbatch ./scripts/run_test.sh config/obqa.config swow pg_full swow pred 17036 5 -J albert
# sbatch ./scripts/run_test.sh config/obqa.config swow pg_full swow pred 3469 0 -J albert

# ./saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/train.log
####### ensemble #####
# sbatch ./scripts/


bash ./scripts/run_main.sh config/mcscript_17rel.config cpnet None cpnet 

bash ./scripts/run_main.sh config/csqa_17rel.config cpnet None cpnet 