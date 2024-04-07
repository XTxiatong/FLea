#!/bin/sh
python -u main.py --dataset UCI_HAR \
               --split_method distribution \
			   --split_para 0.1 \
			   --split_num 150 \
			   --client_num 15 \
			   --class_num 6 \
			   --local_epochs 5 \
			   --batch_size 32 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname HARNet \
			   --algorithm FedMix_pre \
               --loss CE_MCE \
			   --layer 0 \
			   --fea_percent 0.1 \
			   --seed 0 \
			   --beta 1
		
			   
			   

 