#!/bin/sh

python -u main.py --dataset Cifar10 \
               --split_method distribution \
			   --split_para 0.5 \
			   --split_num 100 \
			   --class_num 10 \
			   --client_num 10 \
			   --loss CE \
			   --local_epochs 20 \
			   --batch_size 64 \
			   --num_global_iters 1 \
			   --personal_learning_rate 0.0001 \
			   --modelname MOBNET \
			   --algorithm CCVR \
			   --layer -1 \
			   --fea_percent 0.1 
   
       