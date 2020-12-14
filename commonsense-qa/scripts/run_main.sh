#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

source $1 
RANDOM=$$

for ((i=0; i<${n_runs}; i++));do
	if [[ -z ${seeds} ]]; then
		[[ $i -eq 0 ]] && seed=0 || seed=$RANDOM
	else
		seed=${seeds[$i]}
	fi

	save_dir="./saved_models/${dataset}/${encoder}_elr${encoder_lr}_dlr${decoder_lr}_d${dropoutm}_b${batch_size}_s${seed}_g${gpu_device}_${kg_model}_a${ablation}_${kg_name}_ent${ent_emb}_p${subsample}_$i"
	mkdir -p ${save_dir}

	python -u main.py \
		--dataset $dataset \
		--inhouse $inhouse \
		--save_dir $save_dir \
		--encoder $encoder \
		--max_seq_len $max_seq_len \
		--encoder_lr $encoder_lr \
		--decoder_lr $decoder_lr \
		--batch_size $batch_size \
		--dropoutm $dropoutm \
		--gpu_device $gpu_device \
		--nprocs 20 \
		--save_model $save_model \
		--seed $seed \
		--use_cache $use_cache\
		--ent_emb $ent_emb\
		--ent_emb_paths $ent_emb_paths\
		--rel_emb_path $rel_emb_path\
		--kg_name $kg_name\
		--kg_model $kg_model\
		--subsample $subsample\
		--debug $debug\
		--mini_batch_size $mini_batch_size\
		--path_embedding_path $path_embedding_path\
		--ablation $ablation\
		--lm_sent_pool $lm_sent_pool\
		--decoder_hidden_dim $decoder_hidden_dim\
		--encoder_dim $encoder_dim\
		--encoder_dropoute $encoder_dropoute\
		--encoder_dropouti $encoder_dropouti\
		--encoder_dropouth $encoder_dropouth\ 
		> ${save_dir}/train.log
	 
	echo ${save_dir}/train.log 
done