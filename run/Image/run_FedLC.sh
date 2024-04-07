#!/bin/sh
python -u main.py --dataset Cifar10 \
               --split_method distribution \
			   --split_para 1.0 \
			   --split_num 100 \
			   --class_num 10 \
			   --client_num 10 \
			   --loss CE_Prox \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FedProx \
			   --seed 0 \
			   --beta 0.001
			   
python -u main.py --dataset Cifar10 \
               --split_method distribution \
			   --split_para 1.0 \
			   --split_num 100 \
			   --class_num 10 \
			   --client_num 10 \
			   --loss CE_LC \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FedLC \
			   --seed 0 \
			   --beta 1
