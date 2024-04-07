#!/bin/sh
python -u main.py --dataset Cifar10 \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 100 \
			   --class_num 10 \
			   --client_num 10 \
			   --loss CE_RS \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FedRS \
			   --seed 0 \
			   --beta 0.1

			   