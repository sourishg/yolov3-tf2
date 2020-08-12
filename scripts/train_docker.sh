#!/bin/bash                                                                                                                     
export TF_FORCE_GPU_ALLOW_GROWTH=true
export NCCL_DEBUG=INFO

TRAIN_DATASET='/datasets/xplane/washington_morning_scattered_1_temp3.tfrecord'
VAL_DATASET='/datasets/xplane/washington_morning_overcast_1_temp3.tfrecord'
CHECKPOINT=None
EPOCHS=30
BATCH_SIZE=8
NUM_CLASSES=11
LEARNING_RATE=1e-5
ALPHA=0.25
GAMMA=4
DELTA=1

docker run --gpus all -it --rm -v /mnt/datasets/xplane:/datasets/xplane -v $HOME/dev/flying_object_detection/retinanet-tf2:/retinanet-tf2 -w /retinanet-tf2 sourishg/tf_retinanet:2.2.0 python train.py \
--dataset=$TRAIN_DATASET \
--val_dataset=$VAL_DATASET \
--epochs=$EPOCHS \
--batch_size=$BATCH_SIZE \
--num_classes=$NUM_CLASSES \
--learning_rate=$LEARNING_RATE \
--alpha=$ALPHA \
--gamma=$GAMMA \
--delta=$DELTA
