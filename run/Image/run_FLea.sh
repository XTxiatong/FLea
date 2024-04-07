#!/bin/sh
python -u main.py --dataset Cifar10 \
               --split_method distribution \
			   --split_para 0.1 \
			   --split_num 1000 \
			   --client_num 100 \
			   --class_num 10 \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FLea \
               --loss MCE_DeC_KL \
			   --layer 0 \
			   --fea_percent 0.1 \
			   --seed 0 \
			   --beta 1
		
			   
			   

 