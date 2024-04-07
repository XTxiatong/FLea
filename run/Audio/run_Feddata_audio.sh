#!/bin/sh
python -u main.py --dataset UrbanSound \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 140 \
			   --client_num 14 \
			   --class_num 10 \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm Data_pre \
               --loss CE \
			   --fea_percent 0.1 \
			   --layer -1  

python -u main.py --dataset UrbanSound \
               --split_method distribution \
			   --split_para 0.5 \
			   --split_num 140 \
			   --client_num 14 \
			   --class_num 10 \
			   --local_epochs 5 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname AudioNet \
			   --algorithm Data_pre \
               --loss CE \
			   --fea_percent 0.1 \
			   --layer -1 			   