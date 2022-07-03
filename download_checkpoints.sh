#!/usr/bin/env bash

mkdir -p checkpoints
cd checkpoints

echo 'Downloading weights for instance segmentation model'
wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

echo 'Downloading weights for semantic segmentation model'
wget -c https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_4x4_320k_coco-stuff164k/pspnet_r50-d8_512x512_4x4_320k_coco-stuff164k_20210707_152004-be9610cc.pth
