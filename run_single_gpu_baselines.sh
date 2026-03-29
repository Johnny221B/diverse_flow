#!/bin/bash

# 遇到错误即停止
set -e

# --- 统一参数配置 ---
G_VAL=16
GUIDANCE=5.0
STEPS=50
MODEL_DIR="./models/stable-diffusion-3.5-medium"
# 物理显卡 ID
CARD_ID=3

# --- 数据集列表 ---
# SPECS=("specs/t2i_color_mini.json" "specs/t2i_spatial_mini.json" "specs/t2i_complex_mini.json")
SPECS=("specs/t2i_human_mini.json")

# ==============================================================================
# 0. mix ode and sde (Vanilla SD3.5)
# ==============================================================================
# echo ">>>> 正在运行 mix ode and sde <<<<"
# for SPEC in "${SPECS[@]}"; do
#     CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_mix_flow_t2i.py \
#       --spec "$SPEC" \
#       --method "mix" \
#       --model-dir "$MODEL_DIR" \
#       --G $G_VAL \
#       --guidance $GUIDANCE \
#       --fp16 \
#       --device "cuda:0"
# done

# # ==============================================================================
# # 1. Base Model (Vanilla SD3.5)
# # ==============================================================================
# echo ">>>> 正在运行 Base Model (Standard ODE) <<<<"
# for SPEC in "${SPECS[@]}"; do
#     CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_base_t2i.py \
#       --spec "$SPEC" \
#       --method "base" \
#       --model-dir "$MODEL_DIR" \
#       --G $G_VAL \
#       --guidance $GUIDANCE \
#       --fp16 \
#       --device "cuda:0"
# done

# # ==============================================================================
# # 2. PG (Particle Guidance)
# # ==============================================================================
# echo ">>>> 正在运行 PG Baseline <<<<"
# for SPEC in "${SPECS[@]}"; do
#     CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_pg_t2i.py \
#       --spec "$SPEC" \
#       --method "pg" \
#       --model "$MODEL_DIR" \
#       --G $G_VAL \
#       --cfg $GUIDANCE \
#       --fp16 \
#       --device "cuda:0"
# done

# ==============================================================================
# 3. APG (Adaptive Projected Guidance)
# ==============================================================================
echo ">>>> 正在运行 APG Baseline <<<<"
for SPEC in "${SPECS[@]}"; do
    CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_apg_t2i.py \
      --spec "$SPEC" \
      --method "apg" \
      --model-dir "$MODEL_DIR" \
      --G $G_VAL \
      --guidance $GUIDANCE \
      --fp16 \
      --device "cuda:0"
done

echo "==== 单卡卡 $CARD_ID 的所有 Baseline 任务已完成 ===="