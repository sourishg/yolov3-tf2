#!/bin/bash
export TF_FORCE_GPU_ALLOW_GROWTH=true

TRAIN_DATASET='/mnt/datasets/xplane/train/*.tfrecord'
VAL_DATASET='/mnt/datasets/xplane/test/*.tfrecord'
WEIGHTS='./checkpoints/yolov3.tf'
IMG_SIZE=1280
EPOCHS=30
BATCH_SIZE=16
NUM_CLASSES=11
LEARNING_RATE=1e-3
MODE='fit'
TRANSFER='none'


python train.py \
--dataset=$TRAIN_DATASET \
--val_dataset=$VAL_DATASET \
--epochs=$EPOCHS \
--batch_size=$BATCH_SIZE \
--num_classes=$NUM_CLASSES \
--learning_rate=$LEARNING_RATE \
--size=$IMG_SIZE \
--mode=$MODE \
--transfer=$TRANSFER \
--tiny
