#!/bin/bash
# 评测脚本：对 G=64 生成结果计算所有指标（含 KID）
# 用法：bash run_eval_g64.sh
set -e
source /data2/toby/qdhf/bin/activate

cd /data2/toby/OSCAR

METHODS="ourmethod dpp cads base mix pg apg"
CONCEPTS="t2i_color t2i_complex"
DEVICE="cuda:0"

python evaluation/eval_t2i.py \
  --outputs-root ./outputs \
  --methods $METHODS \
  --concepts $CONCEPTS \
  --device $DEVICE \
  --coco-dir ./datasets/coco2017/val2017 \
  --kid-subsets 100 \
  --kid-subset-size 1000

echo "=== Evaluation DONE ==="
