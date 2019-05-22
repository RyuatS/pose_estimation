# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-05-20T06:26:42.185Z
# Description:
#
# ===============================================

python train.py --checkpoints_dir=./checkpoints \
                --tfrecord_dir=./data/cocodevkit/tfrecord \
                --data_type=val2017 \
                --logdir=./logdir   \
                --save_interval=100 \
                --eval_interval=100 \
                --steps=10000       \
                --batch_size=8     \
                --base_learning_rate=0.01 \
                --learning_rate_decay_factor=0.1 \
                --learning_rate_decay_step=1000  \
                --weight_decay=0.00005           \
                --model_type=hourglass
