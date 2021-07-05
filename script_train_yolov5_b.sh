#!/usr/bin/env bash

python \
-m torch.distributed.launch \
--nproc_per_node 4 \
train_YL.py \
--data data/znlsg.yaml \
--epochs 20 \
--weights /media/D_4TB/YL_4TB/Competitions/ZNLSG_21_XinYe/data/cssjj/yolov5_rundata/train11/weights/ckp_a_all_yolov5_imgs_txts_1c_epoch40.pt \
-train b_all_yolov5_imgs_txts.txt \
-test b_all_yolov5_imgs_txts.txt \
--single-cls \
--img-size 960 960 \
--batch-size 12 \
--device 0,1,2,3 \
--data-dir ../../data/cssjj/test \
--save-inter 1

