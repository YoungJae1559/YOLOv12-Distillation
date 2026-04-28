#!/usr/bin/env bash
set -euo pipefail

# Clean launcher for Figure 1 (a): head-only KD baseline.
# This avoids interactive shell pollution that can break libtorch/MKL loading.

source /home/vip/anaconda3/bin/activate 12
cd /home/vip/harry/yolov12

unset PYTHONHOME || true
unset PYTHONPATH || true
unset LD_PRELOAD || true

export MPLCONFIGDIR=/tmp/mpl
export YOLO_CONFIG_DIR=/tmp/yolo_cfg
export LD_LIBRARY_PATH=/home/vip/anaconda3/envs/12/lib:/home/vip/anaconda3/envs/12/lib/python3.11/site-packages/torch/lib
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=1

export YOLO_TEACHER=/home/vip/harry/yolov12x.pt
export YOLO_KD_TYPE=plain_kd
export YOLO_KD_W=0.0
export YOLO_KD_HEAD_W=0.2
export YOLO_KD_HEAD_CLS_W=1.0
export YOLO_KD_HEAD_REG_W=1.0
export YOLO_KD_HEAD_TAU=2.0
export YOLO_KD_HEAD_MIN_CONF=0.1
export YOLO_KD_P3=0.0
export YOLO_KD_P4=0.0
export YOLO_KD_P5=0.0
export YOLO_PLAIN_FEAT_W=1.0

exec /home/vip/anaconda3/envs/12/bin/yolo detect train \
  model=/home/vip/harry/yolov12/runs/detect/yolov12m/weights/best.pt \
  data=/home/vip/harry/yolov12/coco.yaml \
  epochs=50 imgsz=640 batch=16 device=0 amp=True workers=0 \
  name=yolov12m_headonly_x2m
