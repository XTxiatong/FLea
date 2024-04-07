#!/bin/sh
python -u main.py --dataset UrbanSound \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 70 \
			   --client_num 7 \
			   --class_num 10 \
			   --local_epochs 15 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm FLea \
               --loss MCE_DeC_KL \
			   --layer 1 \
			   --fea_percent 0.1 \
			   --seed 0 

 

