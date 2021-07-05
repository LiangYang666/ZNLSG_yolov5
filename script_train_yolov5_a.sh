#!/usr/bin/env bash

python \
-m torch.distributed.launch \
--nproc_per_node 4 \
train_YL.py \
--data data/znlsg.yaml \
--epochs 300 \
--weights yolov5x.pt \
-train a_all_yolov5_imgs_txts.txt \
-test a_test.txt \
--single-cls \
--img-size 960 960 \
--batch-size 16 \
--device 1,2,3 \
--data-dir ../../data/cssjj/train \
