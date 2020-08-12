#!/bin/bash
export TF_FORCE_GPU_ALLOW_GROWTH=true

TRAIN_DATASET='/home/sourish/dev/flying_object_detection/fod/dataengine/data/xplane/train/mount_morning_cirrus_1_temp3.tfrecord'
VAL_DATASET='/home/sourish/dev/flying_object_detection/fod/dataengine/data/xplane/test/mount_evening_clear_2_temp.tfrecord'
WEIGHTS='./checkpoints/yolov3.tf'
IMG_SIZE=1280
EPOCHS=30
BATCH_SIZE=1
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
--transfer=$TRANSFER
