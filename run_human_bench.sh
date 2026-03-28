#!/bin/bash

# 遇到错误即停止
set -e

# --- 全局基础参数 ---
G_VAL=32
GUIDANCE=5.0
STEPS=40      # 提升至 50 步，确保面部细节和服饰纹理完美
MODEL_DIR="./models/stable-diffusion-3.5-medium"
SPEC="specs/t2i_human_mini.json"
CARD_ID=0     # 你指定的显卡 ID

# 1. Mix flow
# echo ">>>> Running mix flow on Human Spec <<<<"
# CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_mix_flow_t2i.py \
#   --spec "$SPEC" \
#   --method "mix_t2i" \
#   --model-dir "$MODEL_DIR" \
#   --G $G_VAL \
#   --guidance $GUIDANCE \
#   --steps $STEPS

# 2. Base Model (Vanilla SD3.5)
echo ">>>> Running Base Model (Vanilla ODE) on Human Spec <<<<"
CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_base_t2i.py \
  --spec "$SPEC" \
  --method "base_t2i" \
  --model-dir "$MODEL_DIR" \
  --G $G_VAL \
  --guidance $GUIDANCE \
  --steps $STEPS \
  --fp16 \
  --device "cuda:0"

# 3. APG (Adaptive Projected Guidance)
# echo ">>>> Running APG Baseline on Human Spec <<<<"
# CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_apg_t2i.py \
#   --spec "$SPEC" \
#   --method "apg_t2i" \
#   --model-dir "$MODEL_DIR" \
#   --G $G_VAL \
#   --guidance $GUIDANCE \
#   --steps $STEPS \
#   --fp16 \
#   --device "cuda:0"

# 4. PG (Particle Guidance) 
echo ">>>> Running PG Baseline on Human Spec <<<<"
CUDA_VISIBLE_DEVICES=$CARD_ID python scripts/baseline_pg_t2i.py \
  --spec "$SPEC" \
  --method "pg_t2i" \
  --model "$MODEL_DIR" \
  --G $G_VAL \
  --cfg $GUIDANCE \
  --steps $STEPS \
  --fp16 \
  --device "cuda:0"

echo "==== Human Diversity 专项任务运行完毕 ===="