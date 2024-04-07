#!/bin/sh
python -u main.py --dataset UrbanSound \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 140 \
			   --class_num 10 \
			   --client_num 14 \
			   --loss NT_CE \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm FedNTD \
			   --seed 0
			   
python -u main.py --dataset UrbanSound \
               --split_method distribution \
			   --split_para 0.5 \
			   --split_num 140 \
			   --class_num 10 \
			   --client_num 14 \
			   --loss NT_CE \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm FedNTD \
			   --seed 0
			   
python -u main.py --dataset UrbanSound \
               --split_method distribution \
			   --split_para 0.1 \
			   --split_num 140 \
			   --class_num 10 \
			   --client_num 14 \
			   --loss NT_CE \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm FedNTD \
			   --seed 0