#!/bin/bash
#

echo "===Begin==="
nohup python main.py --model=MF_Naive --batch_size 2048 --embedding_size 40 --lr 0.001 --weight_decay 0.1 >MF_Naive_50.log 2>&1
echo "finished"
nohup python main.py --model=MF_IPS --batch_size 256 --embedding_size 48 --lr 0.001 --weight_decay 1 >MF_IPS_50.log 2>&1
echo "finished"
nohup python main.py --model=CausE --batch_size 2048 --embedding_size 28 --lr 0.005 --weight_decay 0.1 --reg_c 0.1 --reg_t 0.001 --reg_tc 0.1 >CausE_50.log 2>&1
echo "finished"

echo "===end==="