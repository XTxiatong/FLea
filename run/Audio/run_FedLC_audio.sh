#!/bin/sh

python -u main.py --dataset UrbanSound \
               --split_method distribution \
			   --split_para 0.1 \
			   --split_num 140 \
			   --class_num 10 \
			   --client_num 14 \
			   --loss CE_LC \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm FedLC \
			   --seed 0 