#!/usr/bin/env bash

python \
detect_multi.py \
--weights yolov5_rundata/train11/weights/ckp_a_all_yolov5_imgs_txts_1c_epoch40.pt \
--img-size 960 \
--source a_images \
--data-dir ../../data/cssjj/train \
--device 0,1,2,3 \
--batch-size 16