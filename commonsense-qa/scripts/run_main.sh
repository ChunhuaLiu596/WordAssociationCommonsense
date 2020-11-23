#!/bin/bash

source $1 

save_dir="./saved_models/${dataset}/${encoder}_elr${encoder_lr}_dlr${decoder_lr}_d${dropoutm}_b${batch_size}_s${seed}_g${gpu_device}_${kg_model}_${kg_name}"
mkdir -p ${save_dir}

nohup python -u main.py \
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
	--cpnet_vocab_path $cpnet_vocab_path\
	--cpnet_graph_path $cpnet_graph_path\
	--ent_emb_paths $ent_emb_paths\
	--rel_emb_path $rel_emb_path\
	--kg_name $kg_name\
	--path_embedding $path_embedding\
	> ${save_dir}/train.log 2>&1 &

echo ${save_dir}/train.log 
tail -f ${save_dir}/train.log